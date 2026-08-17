from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest
import yaml
from jsonschema import Draft202012Validator, FormatChecker
from jsonschema.exceptions import ValidationError


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = PROJECT_ROOT / "config" / "historical-market-data.yaml"
SCHEMA_PATH = PROJECT_ROOT / "config" / "schemas" / "historical-market-candle.schema.json"
VALID_FIXTURE_PATH = (
    PROJECT_ROOT / "tests" / "fixtures" / "schemas" / "valid" / "historical-market-candle.json"
)
INVALID_PRODUCT_FIXTURE_PATH = (
    PROJECT_ROOT
    / "tests"
    / "fixtures"
    / "schemas"
    / "invalid"
    / "historical-market-candle-invalid-product.json"
)
NEGATIVE_PRICE_FIXTURE_PATH = (
    PROJECT_ROOT
    / "tests"
    / "fixtures"
    / "schemas"
    / "invalid"
    / "historical-market-candle-negative-price.json"
)

EXPECTED_PRODUCTS = ["BTC-USD", "ETH-USD"]
EXPECTED_START = datetime(2026, 1, 1, 0, 0, 0, tzinfo=UTC)
EXPECTED_END = datetime(2026, 6, 30, 23, 59, 59, tzinfo=UTC)
DATA_FILE_SUFFIXES = {".csv", ".json", ".jsonl", ".parquet", ".avro", ".orc", ".delta"}
FORBIDDEN_CANDLE_FIELDS = {
    "account_id",
    "confirmed_fraud",
    "device_id",
    "fraud_label",
    "fraud_type",
    "investigation_status",
    "is_fraud",
    "label_source",
    "label_timestamp",
    "login_id",
    "transaction_id",
    "wallet_id",
}
CREDENTIAL_KEY_TERMS = {"api_key", "apikey", "client_secret", "password", "secret", "token"}


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as file:
        data = yaml.safe_load(file)
    assert isinstance(data, dict)
    return data


def load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as file:
        return json.load(file)


def parse_utc(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    assert parsed.tzinfo is not None
    return parsed.astimezone(UTC)


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


@pytest.fixture(scope="session")
def config() -> dict[str, Any]:
    return load_yaml(CONFIG_PATH)


@pytest.fixture(scope="session")
def schema() -> dict[str, Any]:
    return load_json(SCHEMA_PATH)


@pytest.fixture(scope="session")
def validator(schema: dict[str, Any]) -> Draft202012Validator:
    return Draft202012Validator(schema, format_checker=FormatChecker())


def test_historical_market_configuration_exists() -> None:
    assert CONFIG_PATH.exists()


def test_only_btc_usd_and_eth_usd_are_configured(config: dict[str, Any]) -> None:
    assert config["products"] == EXPECTED_PRODUCTS
    assert config["enrichment_design"]["asset_product_mapping"] == {
        "BTC": "BTC-USD",
        "ETH": "ETH-USD",
    }


def test_date_range_is_valid_fixed_and_utc(config: dict[str, Any]) -> None:
    start = parse_utc(config["time_range"]["start"])
    end = parse_utc(config["time_range"]["end"])

    assert config["time_range"]["timezone"] == "UTC"
    assert start == EXPECTED_START
    assert end == EXPECTED_END
    assert start < end


def test_granularity_equals_sixty_seconds(config: dict[str, Any]) -> None:
    assert config["granularity"] == {"seconds": 60, "name": "one_minute"}


def test_maximum_candles_per_request_does_not_exceed_coinbase_limit(
    config: dict[str, Any],
) -> None:
    assert config["request_design"]["maximum_candles_per_request"] <= 300
    assert config["request_design"]["checkpoint_enabled"] is True
    assert config["request_design"]["retry_enabled"] is True
    assert config["request_design"]["backoff_enabled"] is True


def test_storage_format_is_parquet(config: dict[str, Any]) -> None:
    assert config["storage"]["format"] == "parquet"
    assert config["storage"]["raw_dataset_name"] == "historical_market_candles"


def test_partition_columns_include_product_id_and_event_date(config: dict[str, Any]) -> None:
    assert config["storage"]["partition_columns"] == ["product_id", "event_date"]


def test_deduplication_uses_product_id_and_candle_start(config: dict[str, Any]) -> None:
    expected_key = ["product_id", "candle_start_timestamp"]
    assert config["quality"]["deduplication_key"] == expected_key
    assert config["quality"]["natural_business_key"] == expected_key


def test_missing_intervals_are_preserved_and_flagged(config: dict[str, Any]) -> None:
    assert config["quality"]["missing_interval_policy"] == "preserve_and_flag"


def test_future_values_are_not_allowed_for_enrichment(config: dict[str, Any]) -> None:
    assert config["quality"]["future_fill_policy"] == "do_not_use_future_values"
    assert (
        config["enrichment_design"]["candle_selection_rule"]
        == "most_recent_completed_candle_at_or_before_transaction_timestamp"
    )


def test_valid_candle_fixture_passes_schema_validation(
    validator: Draft202012Validator,
) -> None:
    validator.validate(load_json(VALID_FIXTURE_PATH))


def test_invalid_product_fixture_fails_schema_validation(
    validator: Draft202012Validator,
) -> None:
    with pytest.raises(ValidationError):
        validator.validate(load_json(INVALID_PRODUCT_FIXTURE_PATH))


def test_negative_price_fixture_fails_schema_validation(
    validator: Draft202012Validator,
) -> None:
    with pytest.raises(ValidationError):
        validator.validate(load_json(NEGATIVE_PRICE_FIXTURE_PATH))


def test_historical_candle_schema_is_valid_draft_2020_12(schema: dict[str, Any]) -> None:
    assert schema["$schema"] == "https://json-schema.org/draft/2020-12/schema"
    Draft202012Validator.check_schema(schema)


def test_existing_phase_2_and_phase_3_validation_suites_are_present() -> None:
    assert (PROJECT_ROOT / "tests" / "data_quality" / "test_json_schemas.py").exists()
    assert (PROJECT_ROOT / "tests" / "data_quality" / "test_fraud_scenario_configs.py").exists()


def test_no_customer_fraud_or_label_fields_exist_in_historical_candle_schema(
    schema: dict[str, Any],
) -> None:
    assert FORBIDDEN_CANDLE_FIELDS.isdisjoint(set(schema["properties"]))


def test_no_api_credentials_are_stored_in_configuration(config: dict[str, Any]) -> None:
    all_keys = {key.lower() for key in iter_mapping_keys(config)}
    assert CREDENTIAL_KEY_TERMS.isdisjoint(all_keys)
    assert config["credential_policy"]["store_credentials_in_file"] is False


def test_no_downloaded_dataset_exists() -> None:
    data_root = PROJECT_ROOT / "data"
    approved_phase_5a_pilot_root = data_root / "generated" / "historical" / "pilot"
    dataset_files = [
        path
        for path in data_root.rglob("*")
        if path.is_file() and path.suffix.lower() in DATA_FILE_SUFFIXES
        and approved_phase_5a_pilot_root not in path.parents
    ]
    assert dataset_files == []


def test_valid_fixture_satisfies_application_level_candle_rules() -> None:
    record = load_json(VALID_FIXTURE_PATH)
    start = parse_utc(record["candle_start_timestamp"])
    end = parse_utc(record["candle_end_timestamp"])

    assert end == start + timedelta(seconds=60)
    assert record["high_price_usd"] >= record["open_price_usd"]
    assert record["high_price_usd"] >= record["close_price_usd"]
    assert record["high_price_usd"] >= record["low_price_usd"]
    assert record["low_price_usd"] <= record["open_price_usd"]
    assert record["low_price_usd"] <= record["close_price_usd"]
    assert record["low_price_usd"] <= record["high_price_usd"]
