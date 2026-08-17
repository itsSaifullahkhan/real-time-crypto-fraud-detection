from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from crypto_fraud_platform.data_generator.common.config import load_historical_generator_config
from crypto_fraud_platform.data_generator.common.identifiers import (
    DeterministicIdFactory,
    deterministic_uuid,
)
from crypto_fraud_platform.data_generator.common.schema_validation import assert_no_transaction_leakage
from crypto_fraud_platform.data_generator.fraud_scenarios.scenario_engine import ScenarioEngine
from crypto_fraud_platform.data_generator.historical.entity_generator import generate_entities
from crypto_fraud_platform.data_generator.historical.label_generator import generate_fraud_labels
from crypto_fraud_platform.data_generator.historical.market_candle_downloader import (
    calculate_missing_intervals,
    generate_request_windows,
    is_retryable_status,
    normalize_coinbase_candles,
)
from crypto_fraud_platform.data_generator.historical.transaction_generator import (
    MarketPriceLookup,
    make_customer_transaction,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = PROJECT_ROOT / "config" / "historical-generator.yaml"


@pytest.fixture()
def config():
    return load_historical_generator_config(CONFIG_PATH)


def test_deterministic_uuid_generation() -> None:
    first = deterministic_uuid("account", "20260729", 1)
    second = deterministic_uuid("account", "20260729", 1)
    different = deterministic_uuid("account", "20260729", 2)

    assert first == second
    assert first != different
    assert DeterministicIdFactory(seed=7).next("login") == DeterministicIdFactory(seed=7).next("login")


def test_entity_generation_is_deterministic(config) -> None:
    kwargs = {
        "account_count": 5,
        "start_timestamp": config.time_range.start,
        "end_timestamp": config.time_range.end,
    }
    first = generate_entities(
        **kwargs,
        rng=np.random.default_rng(11),
        id_factory=DeterministicIdFactory(seed=11),
    )
    second = generate_entities(
        **kwargs,
        rng=np.random.default_rng(11),
        id_factory=DeterministicIdFactory(seed=11),
    )

    pd.testing.assert_frame_equal(first.accounts, second.accounts)
    pd.testing.assert_frame_equal(first.devices, second.devices)
    pd.testing.assert_frame_equal(first.wallets, second.wallets)


def test_request_window_generation_respects_coinbase_limit() -> None:
    start = datetime(2026, 6, 1, tzinfo=UTC)
    end = start + timedelta(minutes=301) - timedelta(seconds=1)

    windows = generate_request_windows("BTC-USD", start, end)

    assert len(windows) == 2
    assert all((window.end_exclusive - window.start).total_seconds() <= 300 * 60 for window in windows)


def test_retry_classification() -> None:
    assert is_retryable_status(429)
    assert is_retryable_status(503)
    assert not is_retryable_status(400)
    assert not is_retryable_status(404)


def test_candle_response_normalization_sorts_and_deduplicates() -> None:
    start = datetime(2026, 6, 1, 0, 0, tzinfo=UTC)
    retrieved_at = datetime(2026, 7, 1, 0, 0, tzinfo=UTC)
    payload = [
        [int((start + timedelta(minutes=1)).timestamp()), 101, 106, 102, 105, 2.5],
        [int(start.timestamp()), 99, 103, 100, 102, 1.5],
        [int(start.timestamp()), 99, 103, 100, 102, 1.5],
    ]

    records = normalize_coinbase_candles(
        payload,
        product_id="BTC-USD",
        retrieved_at=retrieved_at,
        start=start,
        end_exclusive=start + timedelta(minutes=2),
    )

    assert [record["candle_start_timestamp"] for record in records] == [
        start,
        start + timedelta(minutes=1),
    ]
    assert records[0]["source"] == "coinbase_exchange_rest_api"
    assert records[0]["candle_end_timestamp"] == start + timedelta(minutes=1)


def test_missing_interval_detection() -> None:
    start = datetime(2026, 6, 1, 0, 0, tzinfo=UTC)
    candles = _market_candles(start, minutes=3).iloc[[0, 2]]

    missing = calculate_missing_intervals(
        candles,
        product_id="BTC-USD",
        start=start,
        end=start + timedelta(minutes=3) - timedelta(seconds=1),
    )

    assert missing == [start + timedelta(minutes=1)]


def test_account_device_and_wallet_relationships(config) -> None:
    entities = generate_entities(
        account_count=12,
        start_timestamp=config.time_range.start,
        end_timestamp=config.time_range.end,
        rng=np.random.default_rng(12),
        id_factory=DeterministicIdFactory(seed=12),
    )
    account_ids = set(entities.accounts["account_id"])

    assert set(entities.devices["primary_account_id"].dropna()).issubset(account_ids)
    assert set(entities.wallets["owner_account_id"].dropna()).issubset(account_ids)
    assert len(entities.shared_device_ids) >= 1
    assert len(entities.shared_external_wallet_ids) >= 1


def test_market_candle_lookup_uses_most_recent_completed_candle() -> None:
    start = datetime(2026, 6, 1, 0, 0, tzinfo=UTC)
    lookup = MarketPriceLookup(_market_candles(start, minutes=5), {"BTC": "BTC-USD"})

    match = lookup.lookup("BTC", start + timedelta(minutes=2, seconds=30))

    assert match.candle_start_timestamp == start + timedelta(minutes=1)
    assert match.candle_end_timestamp <= start + timedelta(minutes=2, seconds=30)
    assert match.market_price_usd == 101.5


def test_transaction_usd_calculation_and_no_leakage() -> None:
    start = datetime(2026, 6, 1, 0, 0, tzinfo=UTC)
    lookup = MarketPriceLookup(_market_candles(start, minutes=5), {"BTC": "BTC-USD"})
    record, match = make_customer_transaction(
        id_factory=DeterministicIdFactory(seed=13),
        market_lookup=lookup,
        event_timestamp=start + timedelta(minutes=3),
        account_id=deterministic_uuid("account", 1),
        asset="BTC",
        desired_amount_usd=250.0,
        transaction_type="WITHDRAWAL",
        source_wallet_id=deterministic_uuid("wallet", 1),
        destination_wallet_id=deterministic_uuid("wallet", 2),
        device_id=deterministic_uuid("device", 1),
        country="US",
    )

    assert record["transaction_amount_usd"] == pytest.approx(
        record["crypto_quantity"] * record["market_price_usd"],
        abs=0.0001,
    )
    assert match["matched_market_candle_end_timestamp"] <= record["event_timestamp"]
    assert_no_transaction_leakage(pd.DataFrame([record]))
    with pytest.raises(ValueError):
        assert_no_transaction_leakage(pd.DataFrame([{**record, "is_fraud": True}]))


def test_scenario_selection_keeps_optional_volatility_disabled(config) -> None:
    engine = ScenarioEngine(
        project_root=PROJECT_ROOT,
        rng=np.random.default_rng(14),
        optional_volatility_enabled=config.generation.optional_volatility_scenario_enabled,
    )

    assert set(engine.enabled_scenario_ids) == {
        "account_takeover",
        "high_transaction_velocity",
        "unusual_transaction_amount",
        "structuring",
        "mule_account_activity",
        "shared_suspicious_device",
    }
    assert "high_volatility_unusual_withdrawal" not in engine.enabled_scenario_ids
    allocation = engine.allocate_fraud_counts(40)
    assert sum(allocation.values()) == 40
    assert all(count > 0 for count in allocation.values())


def test_delayed_label_timing(config) -> None:
    transaction_id = deterministic_uuid("transaction", 1)
    transaction_time = datetime(2026, 6, 1, 12, 0, tzinfo=UTC)
    transactions = pd.DataFrame(
        [
            {
                "transaction_id": transaction_id,
                "event_timestamp": pd.Timestamp(transaction_time),
                "transaction_status": "COMPLETED",
            }
        ]
    )
    labels = generate_fraud_labels(
        config=config,
        transactions=transactions,
        scenario_fraud_rows=[
            {
                "transaction_id": transaction_id,
                "transaction_timestamp": transaction_time,
                "scenario_id": "account_takeover",
                "fraud_type": "ACCOUNT_TAKEOVER",
                "reason_codes": ["NEW_DEVICE"],
                "label_delay_hours": {"min": 12, "max": 12},
            }
        ],
        rng=np.random.default_rng(15),
        id_factory=DeterministicIdFactory(seed=15),
    )

    assert len(labels) == 1
    assert bool(labels.iloc[0]["is_fraud"]) is True
    assert labels.iloc[0]["label_timestamp"] > pd.Timestamp(transaction_time)


def _market_candles(start: datetime, minutes: int) -> pd.DataFrame:
    records = []
    for minute in range(minutes):
        candle_start = start + timedelta(minutes=minute)
        product = "BTC-USD"
        close = 100.5 + minute
        records.append(
            {
                "schema_version": "1.0",
                "source": "coinbase_exchange_rest_api",
                "product_id": product,
                "candle_start_timestamp": candle_start,
                "candle_end_timestamp": candle_start + timedelta(minutes=1),
                "granularity_seconds": 60,
                "open_price_usd": close - 0.5,
                "high_price_usd": close + 1.0,
                "low_price_usd": close - 1.0,
                "close_price_usd": close,
                "volume": 1.0 + minute,
                "retrieved_at": start + timedelta(days=1),
            }
        )
    return pd.DataFrame(records)
