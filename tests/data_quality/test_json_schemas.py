from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator, FormatChecker
from jsonschema.exceptions import ValidationError
from referencing import Registry, Resource
from referencing.jsonschema import DRAFT202012


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCHEMA_DIR = PROJECT_ROOT / "config" / "schemas"
VALID_FIXTURE_DIR = PROJECT_ROOT / "tests" / "fixtures" / "schemas" / "valid"
INVALID_FIXTURE_DIR = PROJECT_ROOT / "tests" / "fixtures" / "schemas" / "invalid"

SCHEMA_FILES = {
    "common-event": "common-event.schema.json",
    "account": "account.schema.json",
    "device": "device.schema.json",
    "wallet": "wallet.schema.json",
    "authentication-event": "authentication-event.schema.json",
    "customer-transaction": "customer-transaction.schema.json",
    "market-event": "market-event.schema.json",
    "historical-market-candle": "historical-market-candle.schema.json",
    "fraud-label": "fraud-label.schema.json",
    "fraud-decision": "fraud-decision.schema.json",
}

INVALID_FIXTURES = {
    "customer-transaction-missing-transaction-id": "customer-transaction",
    "customer-transaction-forbidden-is-fraud": "customer-transaction",
    "customer-transaction-unsupported-asset": "customer-transaction",
    "customer-transaction-negative-quantity": "customer-transaction",
    "historical-market-candle-invalid-product": "historical-market-candle",
    "historical-market-candle-negative-price": "historical-market-candle",
    "fraud-decision-risk-score-greater-than-one": "fraud-decision",
    "fraud-label-fraud-null-type": "fraud-label",
    "market-event-invalid-product-id": "market-event",
    "authentication-event-missing-common-metadata": "authentication-event",
}

EVENT_CONTRACTS = [
    "authentication-event",
    "customer-transaction",
    "market-event",
    "fraud-label",
    "fraud-decision",
]


def load_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as file:
        return json.load(file)


@pytest.fixture(scope="session")
def schemas() -> dict[str, dict]:
    return {
        schema_name: load_json(SCHEMA_DIR / schema_file)
        for schema_name, schema_file in SCHEMA_FILES.items()
    }


@pytest.fixture(scope="session")
def schema_registry(schemas: dict[str, dict]) -> Registry:
    resources = [
        (
            schema["$id"],
            Resource.from_contents(schema, default_specification=DRAFT202012),
        )
        for schema in schemas.values()
    ]
    return Registry().with_resources(resources)


@pytest.fixture(scope="session")
def format_checker() -> FormatChecker:
    return FormatChecker()


def validator_for(
    schema_name: str,
    schemas: dict[str, dict],
    schema_registry: Registry,
    format_checker: FormatChecker,
) -> Draft202012Validator:
    return Draft202012Validator(
        schemas[schema_name],
        registry=schema_registry,
        format_checker=format_checker,
    )


def assert_valid(
    schema_name: str,
    record: dict,
    schemas: dict[str, dict],
    schema_registry: Registry,
    format_checker: FormatChecker,
) -> None:
    validator_for(schema_name, schemas, schema_registry, format_checker).validate(record)


def assert_invalid(
    schema_name: str,
    record: dict,
    schemas: dict[str, dict],
    schema_registry: Registry,
    format_checker: FormatChecker,
) -> None:
    with pytest.raises(ValidationError):
        assert_valid(schema_name, record, schemas, schema_registry, format_checker)


def valid_record(contract_name: str) -> dict:
    return load_json(VALID_FIXTURE_DIR / f"{contract_name}.json")


def test_expected_schema_files_exist() -> None:
    actual_schema_files = {path.name for path in SCHEMA_DIR.glob("*.schema.json")}
    assert actual_schema_files == set(SCHEMA_FILES.values())


@pytest.mark.parametrize("schema_name", SCHEMA_FILES)
def test_json_schemas_are_valid_draft_2020_12(
    schema_name: str,
    schemas: dict[str, dict],
) -> None:
    assert schemas[schema_name]["$schema"] == "https://json-schema.org/draft/2020-12/schema"
    Draft202012Validator.check_schema(schemas[schema_name])


@pytest.mark.parametrize("schema_name", SCHEMA_FILES)
def test_valid_fixtures_pass_schema_validation(
    schema_name: str,
    schemas: dict[str, dict],
    schema_registry: Registry,
    format_checker: FormatChecker,
) -> None:
    assert_valid(schema_name, valid_record(schema_name), schemas, schema_registry, format_checker)


@pytest.mark.parametrize("fixture_name,schema_name", INVALID_FIXTURES.items())
def test_invalid_fixtures_fail_schema_validation(
    fixture_name: str,
    schema_name: str,
    schemas: dict[str, dict],
    schema_registry: Registry,
    format_checker: FormatChecker,
) -> None:
    invalid_record = load_json(INVALID_FIXTURE_DIR / f"{fixture_name}.json")
    assert_invalid(schema_name, invalid_record, schemas, schema_registry, format_checker)


def test_customer_transaction_rejects_is_fraud_field(
    schemas: dict[str, dict],
    schema_registry: Registry,
    format_checker: FormatChecker,
) -> None:
    record = valid_record("customer-transaction")
    record["is_fraud"] = False

    assert_invalid("customer-transaction", record, schemas, schema_registry, format_checker)


@pytest.mark.parametrize("risk_score", [-0.01, 1.01])
def test_fraud_decision_rejects_risk_scores_outside_zero_to_one(
    risk_score: float,
    schemas: dict[str, dict],
    schema_registry: Registry,
    format_checker: FormatChecker,
) -> None:
    record = valid_record("fraud-decision")
    record["risk_score"] = risk_score

    assert_invalid("fraud-decision", record, schemas, schema_registry, format_checker)


@pytest.mark.parametrize("event_schema_name", EVENT_CONTRACTS)
def test_event_specific_schemas_require_common_metadata(
    event_schema_name: str,
    schemas: dict[str, dict],
    schema_registry: Registry,
    format_checker: FormatChecker,
) -> None:
    record = valid_record(event_schema_name)
    record.pop("event_id")

    assert_invalid(event_schema_name, record, schemas, schema_registry, format_checker)


@pytest.mark.parametrize("asset", ["BTC", "ETH"])
def test_customer_transaction_accepts_only_supported_assets(
    asset: str,
    schemas: dict[str, dict],
    schema_registry: Registry,
    format_checker: FormatChecker,
) -> None:
    record = valid_record("customer-transaction")
    record["asset"] = asset

    assert_valid("customer-transaction", record, schemas, schema_registry, format_checker)


def test_customer_transaction_rejects_unsupported_assets(
    schemas: dict[str, dict],
    schema_registry: Registry,
    format_checker: FormatChecker,
) -> None:
    record = valid_record("customer-transaction")
    record["asset"] = "SOL"

    assert_invalid("customer-transaction", record, schemas, schema_registry, format_checker)


def test_fraud_decision_contract_does_not_store_numeric_thresholds(
    schemas: dict[str, dict],
    schema_registry: Registry,
    format_checker: FormatChecker,
) -> None:
    forbidden_threshold_fields = {
        "allow_threshold",
        "review_threshold",
        "block_threshold",
        "decision_threshold",
        "fraud_threshold",
    }
    decision_properties = set(schemas["fraud-decision"]["properties"])

    assert forbidden_threshold_fields.isdisjoint(decision_properties)

    record = copy.deepcopy(valid_record("fraud-decision"))
    record["decision_threshold"] = 0.8

    assert_invalid("fraud-decision", record, schemas, schema_registry, format_checker)
