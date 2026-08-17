from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from crypto_fraud_platform.data_generator.common.parquet_io import read_parquet_dataset
from crypto_fraud_platform.data_generator.common.schema_validation import (
    SchemaValidator,
    TRANSACTION_LEAKAGE_FIELDS,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_ROOT = PROJECT_ROOT / "data" / "generated" / "historical" / "pilot"
MANIFEST_PATH = OUTPUT_ROOT / "_metadata" / "generation_manifest.json"
QUALITY_REPORT_PATH = OUTPUT_ROOT / "_metadata" / "quality_report.json"
EXPECTED_SCENARIOS = {
    "account_takeover",
    "high_transaction_velocity",
    "unusual_transaction_amount",
    "structuring",
    "mule_account_activity",
    "shared_suspicious_device",
}


@pytest.fixture(scope="session")
def manifest() -> dict:
    if not MANIFEST_PATH.exists():
        pytest.skip("Historical pilot output has not been generated in this environment.")
    with MANIFEST_PATH.open(encoding="utf-8") as file:
        return json.load(file)


@pytest.fixture(scope="session")
def quality_report(manifest: dict) -> dict:
    assert QUALITY_REPORT_PATH.exists()
    with QUALITY_REPORT_PATH.open(encoding="utf-8") as file:
        return json.load(file)


@pytest.fixture(scope="session")
def datasets(manifest: dict) -> dict[str, pd.DataFrame]:
    return {
        "accounts": pd.read_parquet(OUTPUT_ROOT / "accounts" / "accounts.parquet", engine="pyarrow"),
        "devices": pd.read_parquet(OUTPUT_ROOT / "devices" / "devices.parquet", engine="pyarrow"),
        "wallets": pd.read_parquet(OUTPUT_ROOT / "wallets" / "wallets.parquet", engine="pyarrow"),
        "authentication_events": read_parquet_dataset(OUTPUT_ROOT / "authentication_events"),
        "customer_transactions": read_parquet_dataset(OUTPUT_ROOT / "customer_transactions"),
        "fraud_labels": read_parquet_dataset(OUTPUT_ROOT / "fraud_labels"),
        "market_candles": read_parquet_dataset(OUTPUT_ROOT / "market_candles"),
        "scenario_assignments": pd.read_parquet(
            OUTPUT_ROOT / "_audit" / "scenario_assignments.parquet",
            engine="pyarrow",
        ),
        "market_enrichment_matches": pd.read_parquet(
            OUTPUT_ROOT / "_audit" / "market_enrichment_matches.parquet",
            engine="pyarrow",
        ),
    }


def test_required_output_datasets_exist(manifest: dict) -> None:
    expected_paths = [
        OUTPUT_ROOT / "market_candles",
        OUTPUT_ROOT / "accounts" / "accounts.parquet",
        OUTPUT_ROOT / "devices" / "devices.parquet",
        OUTPUT_ROOT / "wallets" / "wallets.parquet",
        OUTPUT_ROOT / "authentication_events",
        OUTPUT_ROOT / "customer_transactions",
        OUTPUT_ROOT / "fraud_labels",
        OUTPUT_ROOT / "_audit" / "scenario_assignments.parquet",
        OUTPUT_ROOT / "_audit" / "market_enrichment_matches.parquet",
        MANIFEST_PATH,
        QUALITY_REPORT_PATH,
        OUTPUT_ROOT / "_metadata" / "market_download_checkpoint.json",
    ]

    assert manifest["status"] == "success"
    assert all(path.exists() for path in expected_paths)


def test_parquet_files_are_readable(datasets: dict[str, pd.DataFrame]) -> None:
    assert all(not dataframe.empty for dataframe in datasets.values())


def test_generated_records_match_approved_schemas(datasets: dict[str, pd.DataFrame]) -> None:
    validator = SchemaValidator(PROJECT_ROOT)
    schema_datasets = {
        "account": datasets["accounts"],
        "device": datasets["devices"],
        "wallet": datasets["wallets"],
        "authentication-event": datasets["authentication_events"],
        "customer-transaction": datasets["customer_transactions"],
        "fraud-label": datasets["fraud_labels"],
        "historical-market-candle": datasets["market_candles"],
    }

    for schema_name, dataframe in schema_datasets.items():
        validator.validate_dataframe(
            schema_name,
            dataframe.drop(columns=["event_date", "label_date"], errors="ignore"),
        )


def test_primary_keys_are_unique(datasets: dict[str, pd.DataFrame]) -> None:
    assert datasets["accounts"]["account_id"].is_unique
    assert datasets["devices"]["device_id"].is_unique
    assert datasets["wallets"]["wallet_id"].is_unique
    assert datasets["authentication_events"]["event_id"].is_unique
    assert datasets["authentication_events"]["login_id"].is_unique
    assert datasets["customer_transactions"]["event_id"].is_unique
    assert datasets["customer_transactions"]["transaction_id"].is_unique
    assert datasets["fraud_labels"]["event_id"].is_unique
    assert not datasets["market_candles"].duplicated(["product_id", "candle_start_timestamp"]).any()


def test_foreign_keys_resolve(datasets: dict[str, pd.DataFrame]) -> None:
    account_ids = set(datasets["accounts"]["account_id"])
    device_ids = set(datasets["devices"]["device_id"])
    wallet_ids = set(datasets["wallets"]["wallet_id"])
    transaction_ids = set(datasets["customer_transactions"]["transaction_id"])

    assert set(datasets["authentication_events"]["account_id"]).issubset(account_ids)
    assert set(datasets["authentication_events"]["device_id"]).issubset(device_ids)
    assert set(datasets["customer_transactions"]["account_id"]).issubset(account_ids)
    assert set(datasets["customer_transactions"]["device_id"]).issubset(device_ids)
    assert _nullable_values(datasets["customer_transactions"]["source_wallet_id"]).issubset(wallet_ids)
    assert _nullable_values(datasets["customer_transactions"]["destination_wallet_id"]).issubset(wallet_ids)
    assert set(datasets["fraud_labels"]["transaction_id"]).issubset(transaction_ids)


def test_target_counts_are_reasonable(datasets: dict[str, pd.DataFrame], manifest: dict) -> None:
    assert len(datasets["accounts"]) == 100
    assert len(datasets["customer_transactions"]) == 5000
    assert len(datasets["authentication_events"]) == 2500
    assert set(datasets["market_candles"]["product_id"]) == {"BTC-USD", "ETH-USD"}
    assert manifest["record_counts"]["customer_transactions"] == 5000


def test_all_enabled_scenarios_are_present_and_optional_is_absent(
    datasets: dict[str, pd.DataFrame],
) -> None:
    scenario_ids = set(datasets["scenario_assignments"]["scenario_id"])

    assert EXPECTED_SCENARIOS.issubset(scenario_ids)
    assert "high_volatility_unusual_withdrawal" not in scenario_ids


def test_fraud_rate_is_within_approved_range(datasets: dict[str, pd.DataFrame]) -> None:
    confirmed_count = int((datasets["fraud_labels"]["is_fraud"] == True).sum())  # noqa: E712
    fraud_rate = confirmed_count / len(datasets["customer_transactions"])

    assert 0.005 <= fraud_rate <= 0.01


def test_labels_occur_after_transactions(datasets: dict[str, pd.DataFrame]) -> None:
    joined = datasets["fraud_labels"].merge(
        datasets["customer_transactions"][["transaction_id", "event_timestamp"]],
        on="transaction_id",
        how="left",
        suffixes=("_label", "_transaction"),
    )

    assert (joined["label_timestamp"] > joined["event_timestamp_transaction"]).all()


def test_market_candle_is_not_from_future(datasets: dict[str, pd.DataFrame]) -> None:
    matches = datasets["market_enrichment_matches"]

    assert (
        matches["matched_market_candle_end_timestamp"] <= matches["transaction_event_timestamp"]
    ).all()


def test_no_model_facing_leakage_columns(datasets: dict[str, pd.DataFrame]) -> None:
    leakage_columns = TRANSACTION_LEAKAGE_FIELDS.intersection(
        datasets["customer_transactions"].columns
    )

    assert leakage_columns == set()


def test_transaction_amounts_match_quantity_times_market_price(
    datasets: dict[str, pd.DataFrame],
) -> None:
    transactions = datasets["customer_transactions"]
    expected_amount = (
        transactions["crypto_quantity"].astype(float)
        * transactions["market_price_usd"].astype(float)
    )
    delta = (transactions["transaction_amount_usd"].astype(float) - expected_amount).abs().max()

    assert delta <= 0.0001


def test_quality_report_passed(quality_report: dict) -> None:
    assert quality_report["status"] == "PASS"


def _nullable_values(series: pd.Series) -> set:
    return {value for value in series if pd.notna(value)}
