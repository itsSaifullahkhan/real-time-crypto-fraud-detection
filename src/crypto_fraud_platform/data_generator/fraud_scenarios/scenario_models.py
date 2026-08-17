from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class FraudScenarioDefinition:
    scenario_id: str
    scenario_version: str
    display_name: str
    enabled: bool
    fraud_type: str
    reason_codes: list[str]
    label_delay_hours: dict[str, int | float]
    difficulty: str
    raw_config: dict[str, Any]


@dataclass(frozen=True)
class ScenarioRegistry:
    catalogue_version: str
    enabled_scenarios: list[str]
    optional_scenarios: list[str]
    scenario_paths: dict[str, str]
    target_fraud_rate_min: float
    target_fraud_rate_max: float
