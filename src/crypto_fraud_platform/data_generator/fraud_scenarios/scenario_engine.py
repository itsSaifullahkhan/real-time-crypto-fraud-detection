from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from crypto_fraud_platform.data_generator.common.config import load_yaml_file
from crypto_fraud_platform.data_generator.fraud_scenarios.scenario_models import (
    FraudScenarioDefinition,
    ScenarioRegistry,
)


DEFAULT_SCENARIO_WEIGHTS = {
    "account_takeover": 0.18,
    "high_transaction_velocity": 0.15,
    "unusual_transaction_amount": 0.17,
    "structuring": 0.20,
    "mule_account_activity": 0.15,
    "shared_suspicious_device": 0.15,
}


class ScenarioEngine:
    def __init__(
        self,
        *,
        project_root: Path,
        rng: np.random.Generator,
        optional_volatility_enabled: bool = False,
    ) -> None:
        self.project_root = project_root
        self.rng = rng
        self.registry = load_scenario_registry(project_root)
        self.scenarios = load_scenarios(project_root, self.registry)
        self.enabled_scenario_ids = [
            scenario_id
            for scenario_id in self.registry.enabled_scenarios
            if self.scenarios[scenario_id].enabled
        ]
        if optional_volatility_enabled:
            optional_id = "high_volatility_unusual_withdrawal"
            if self.scenarios[optional_id].enabled:
                self.enabled_scenario_ids.append(optional_id)

    def allocate_fraud_counts(self, target_fraud_count: int) -> dict[str, int]:
        if target_fraud_count < len(self.enabled_scenario_ids):
            raise ValueError("Target fraud count must represent every enabled scenario")
        weights = np.array(
            [DEFAULT_SCENARIO_WEIGHTS.get(scenario_id, 0.1) for scenario_id in self.enabled_scenario_ids],
            dtype=float,
        )
        weights = weights / weights.sum()
        raw_counts = np.floor(weights * target_fraud_count).astype(int)
        raw_counts = np.maximum(raw_counts, 1)
        while int(raw_counts.sum()) < target_fraud_count:
            raw_counts[int(self.rng.integers(0, len(raw_counts)))] += 1
        while int(raw_counts.sum()) > target_fraud_count:
            candidates = np.where(raw_counts > 1)[0]
            raw_counts[int(self.rng.choice(candidates))] -= 1
        return {
            scenario_id: int(count)
            for scenario_id, count in zip(self.enabled_scenario_ids, raw_counts, strict=True)
        }


def load_scenario_registry(project_root: Path) -> ScenarioRegistry:
    registry_path = project_root / "config" / "fraud-rules.yaml"
    raw = load_yaml_file(registry_path)
    target_range = raw["target_fraud_rate_range"]
    return ScenarioRegistry(
        catalogue_version=str(raw["catalogue_version"]),
        enabled_scenarios=list(raw["enabled_scenarios"]),
        optional_scenarios=list(raw["optional_scenarios"]),
        scenario_paths=dict(raw["scenario_paths"]),
        target_fraud_rate_min=float(target_range["min"]),
        target_fraud_rate_max=float(target_range["max"]),
    )


def load_scenarios(
    project_root: Path,
    registry: ScenarioRegistry | None = None,
) -> dict[str, FraudScenarioDefinition]:
    registry = registry or load_scenario_registry(project_root)
    scenarios: dict[str, FraudScenarioDefinition] = {}
    for scenario_id, relative_path in registry.scenario_paths.items():
        scenario = _scenario_from_yaml(project_root / relative_path)
        if scenario.scenario_id != scenario_id:
            raise ValueError(f"Scenario ID mismatch for {relative_path}")
        scenarios[scenario_id] = scenario
    return scenarios


def _scenario_from_yaml(path: Path) -> FraudScenarioDefinition:
    raw: dict[str, Any] = load_yaml_file(path)
    label_delay = raw.get("label_rules", {}).get("label_delay_hours") or raw["timing_rules"]["label_delay_hours"]
    return FraudScenarioDefinition(
        scenario_id=str(raw["scenario_id"]),
        scenario_version=str(raw["scenario_version"]),
        display_name=str(raw["display_name"]),
        enabled=bool(raw["enabled"]),
        fraud_type=str(raw["fraud_type"]),
        reason_codes=[str(code) for code in raw["reason_codes"]],
        label_delay_hours={
            "min": float(label_delay["min"]),
            "max": float(label_delay["max"]),
        },
        difficulty=str(raw["difficulty"]),
        raw_config=raw,
    )
