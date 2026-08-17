from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCENARIO_DIR = PROJECT_ROOT / "config" / "fraud-scenarios"
REGISTRY_PATH = PROJECT_ROOT / "config" / "fraud-rules.yaml"
SCHEMA_DIR = PROJECT_ROOT / "config" / "schemas"

EXPECTED_SCENARIO_FILES = {
    "account_takeover": "account-takeover.yaml",
    "high_transaction_velocity": "high-transaction-velocity.yaml",
    "unusual_transaction_amount": "unusual-transaction-amount.yaml",
    "structuring": "structuring.yaml",
    "mule_account_activity": "mule-account-activity.yaml",
    "shared_suspicious_device": "shared-suspicious-device.yaml",
    "high_volatility_unusual_withdrawal": "high-volatility-unusual-withdrawal.yaml",
}

REQUIRED_SCENARIO_IDS = {
    "account_takeover",
    "high_transaction_velocity",
    "unusual_transaction_amount",
    "structuring",
    "mule_account_activity",
    "shared_suspicious_device",
}

OPTIONAL_SCENARIO_ID = "high_volatility_unusual_withdrawal"

REQUIRED_TOP_LEVEL_KEYS = {
    "scenario_id",
    "scenario_version",
    "display_name",
    "description",
    "enabled",
    "fraud_type",
    "reason_codes",
    "affected_entities",
    "affected_event_types",
    "prerequisites",
    "trigger_logic",
    "event_sequence",
    "timing_rules",
    "expected_feature_signals",
    "label_rules",
    "difficulty",
    "false_positive_risk",
    "normal_control_cases",
    "validation_rules",
    "notes",
}

FORBIDDEN_MODEL_FEATURES = {
    "is_fraud",
    "fraud_type",
    "scenario_id",
    "investigation_status",
}

FORBIDDEN_DECISION_KEYS = {
    "allow_threshold",
    "review_threshold",
    "block_threshold",
    "decision_threshold",
    "fraud_threshold",
    "score_threshold",
    "production_threshold",
    "model_threshold",
}


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as file:
        data = yaml.safe_load(file)
    assert isinstance(data, dict), f"{path} must contain a YAML mapping"
    return data


def load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as file:
        return json.load(file)


@pytest.fixture(scope="session")
def registry() -> dict[str, Any]:
    return load_yaml(REGISTRY_PATH)


@pytest.fixture(scope="session")
def scenarios() -> dict[str, dict[str, Any]]:
    return {
        scenario_id: load_yaml(SCENARIO_DIR / filename)
        for scenario_id, filename in EXPECTED_SCENARIO_FILES.items()
    }


@pytest.fixture(scope="session")
def approved_fraud_types() -> set[str]:
    schema = load_json(SCHEMA_DIR / "fraud-label.schema.json")
    values = schema["properties"]["fraud_type"]["enum"]
    return {value for value in values if value is not None}


@pytest.fixture(scope="session")
def approved_reason_codes() -> set[str]:
    schema = load_json(SCHEMA_DIR / "fraud-decision.schema.json")
    return set(schema["properties"]["reason_codes"]["items"]["enum"])


def iter_mapping_keys(value: Any) -> list[str]:
    keys: list[str] = []
    if isinstance(value, dict):
        for key, child in value.items():
            keys.append(str(key))
            keys.extend(iter_mapping_keys(child))
    elif isinstance(value, list):
        for item in value:
            keys.extend(iter_mapping_keys(item))
    return keys


def test_all_expected_scenario_files_exist() -> None:
    actual_files = {path.name for path in SCENARIO_DIR.glob("*.yaml")}
    assert actual_files == set(EXPECTED_SCENARIO_FILES.values())


def test_registry_yaml_syntax_is_valid(registry: dict[str, Any]) -> None:
    assert registry["catalogue_version"] == "1.0"


@pytest.mark.parametrize("scenario_id", EXPECTED_SCENARIO_FILES)
def test_scenario_yaml_syntax_is_valid(
    scenario_id: str,
    scenarios: dict[str, dict[str, Any]],
) -> None:
    assert scenarios[scenario_id]["scenario_id"] == scenario_id


@pytest.mark.parametrize("scenario_id", EXPECTED_SCENARIO_FILES)
def test_each_scenario_has_required_top_level_keys(
    scenario_id: str,
    scenarios: dict[str, dict[str, Any]],
) -> None:
    assert REQUIRED_TOP_LEVEL_KEYS <= set(scenarios[scenario_id])


def test_scenario_ids_are_unique(scenarios: dict[str, dict[str, Any]]) -> None:
    scenario_ids = [scenario["scenario_id"] for scenario in scenarios.values()]
    assert len(scenario_ids) == len(set(scenario_ids))


@pytest.mark.parametrize("scenario_id", EXPECTED_SCENARIO_FILES)
def test_scenario_versions_are_present(
    scenario_id: str,
    scenarios: dict[str, dict[str, Any]],
) -> None:
    assert scenarios[scenario_id]["scenario_version"] == "1.0"


def test_required_six_scenarios_are_enabled(
    registry: dict[str, Any],
    scenarios: dict[str, dict[str, Any]],
) -> None:
    assert set(registry["enabled_scenarios"]) == REQUIRED_SCENARIO_IDS
    for scenario_id in REQUIRED_SCENARIO_IDS:
        assert scenarios[scenario_id]["enabled"] is True


def test_optional_volatility_scenario_is_disabled(
    registry: dict[str, Any],
    scenarios: dict[str, dict[str, Any]],
) -> None:
    assert registry["optional_scenarios"] == [OPTIONAL_SCENARIO_ID]
    assert scenarios[OPTIONAL_SCENARIO_ID]["enabled"] is False


@pytest.mark.parametrize("scenario_id", EXPECTED_SCENARIO_FILES)
def test_fraud_types_match_approved_fraud_label_schema(
    scenario_id: str,
    scenarios: dict[str, dict[str, Any]],
    approved_fraud_types: set[str],
) -> None:
    assert scenarios[scenario_id]["fraud_type"] in approved_fraud_types
    assert scenarios[scenario_id]["label_rules"]["fraud_type"] == scenarios[scenario_id]["fraud_type"]


@pytest.mark.parametrize("scenario_id", EXPECTED_SCENARIO_FILES)
def test_reason_codes_match_approved_fraud_decision_schema(
    scenario_id: str,
    scenarios: dict[str, dict[str, Any]],
    approved_reason_codes: set[str],
) -> None:
    assert set(scenarios[scenario_id]["reason_codes"]) <= approved_reason_codes


@pytest.mark.parametrize("scenario_id", EXPECTED_SCENARIO_FILES)
def test_scenarios_do_not_define_final_model_decision_thresholds(
    scenario_id: str,
    scenarios: dict[str, dict[str, Any]],
) -> None:
    all_keys = {key.lower() for key in iter_mapping_keys(scenarios[scenario_id])}
    assert FORBIDDEN_DECISION_KEYS.isdisjoint(all_keys)
    assert not any("threshold" in key for key in all_keys)


@pytest.mark.parametrize("scenario_id", EXPECTED_SCENARIO_FILES)
def test_every_scenario_defines_label_rules(
    scenario_id: str,
    scenarios: dict[str, dict[str, Any]],
) -> None:
    label_rules = scenarios[scenario_id]["label_rules"]
    assert label_rules["label_source"] == "SIMULATED_INVESTIGATION"
    assert label_rules["investigation_status"] == "CONFIRMED_FRAUD"
    assert label_rules["is_fraud"] is True
    assert "labeled_transaction_selection" in label_rules


@pytest.mark.parametrize("scenario_id", EXPECTED_SCENARIO_FILES)
def test_every_scenario_defines_expected_feature_signals(
    scenario_id: str,
    scenarios: dict[str, dict[str, Any]],
) -> None:
    features = scenarios[scenario_id]["expected_feature_signals"]
    assert isinstance(features, list)
    assert features
    assert all("name" in feature for feature in features)
    assert all("source" in feature for feature in features)
    assert all("availability" in feature for feature in features)


@pytest.mark.parametrize("scenario_id", EXPECTED_SCENARIO_FILES)
def test_every_scenario_defines_normal_control_cases(
    scenario_id: str,
    scenarios: dict[str, dict[str, Any]],
) -> None:
    controls = scenarios[scenario_id]["normal_control_cases"]
    assert isinstance(controls, list)
    assert controls


@pytest.mark.parametrize("scenario_id", EXPECTED_SCENARIO_FILES)
def test_every_scenario_defines_timing_rules(
    scenario_id: str,
    scenarios: dict[str, dict[str, Any]],
) -> None:
    timing_rules = scenarios[scenario_id]["timing_rules"]
    assert "prerequisite_history_days" in timing_rules
    assert "maximum_scenario_duration_hours" in timing_rules
    assert "label_delay_hours" in timing_rules
    assert "late_or_out_of_order_variants_later" in timing_rules
    assert "can_overlap_with_another_scenario" in timing_rules


@pytest.mark.parametrize("scenario_id", EXPECTED_SCENARIO_FILES)
def test_fraud_labels_are_separate_from_transaction_event_definitions(
    scenario_id: str,
    scenarios: dict[str, dict[str, Any]],
) -> None:
    event_sequence = scenarios[scenario_id]["event_sequence"]
    assert "fraud_label" in scenarios[scenario_id]["affected_event_types"]
    assert any(step["event_type"] == "fraud_label" for step in event_sequence)
    for step in event_sequence:
        if step["event_type"] == "customer_transaction":
            assert step["transaction_event_contains_label"] is False
            assert FORBIDDEN_MODEL_FEATURES.isdisjoint(set(step))


@pytest.mark.parametrize("scenario_id", EXPECTED_SCENARIO_FILES)
def test_no_model_facing_feature_uses_label_or_scenario_metadata(
    scenario_id: str,
    scenarios: dict[str, dict[str, Any]],
) -> None:
    feature_names = {
        feature["name"]
        for feature in scenarios[scenario_id]["expected_feature_signals"]
    }
    assert FORBIDDEN_MODEL_FEATURES.isdisjoint(feature_names)


def test_target_fraud_rate_range_remains_between_half_and_one_percent(
    registry: dict[str, Any],
) -> None:
    target_range = registry["target_fraud_rate_range"]
    assert target_range["unit"] == "fraction_of_customer_transactions"
    assert target_range["min"] == 0.005
    assert target_range["max"] == 0.01


def test_all_referenced_scenario_paths_exist(registry: dict[str, Any]) -> None:
    scenario_paths = registry["scenario_paths"]
    assert set(scenario_paths) == set(EXPECTED_SCENARIO_FILES)
    for path in scenario_paths.values():
        assert (PROJECT_ROOT / path).exists()
