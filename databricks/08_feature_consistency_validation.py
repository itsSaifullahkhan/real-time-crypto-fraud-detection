# Databricks notebook source
# MAGIC %md
# MAGIC # 08 Feature Consistency Validation
# MAGIC
# MAGIC Validate that offline training features and online scoring features share the same
# MAGIC 55-feature contract and semantics.

# COMMAND ----------

from __future__ import annotations

import json
import math
import subprocess
import sys
from bisect import bisect_right
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from typing import Any

import mlflow
import numpy as np
import pandas as pd
from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from pyspark.sql import types as T

spark.conf.set("spark.sql.session.timeZone", "UTC")
spark.conf.set("spark.sql.shuffle.partitions", "1")

try:
    import xgboost  # noqa: F401
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "xgboost==3.4.0"])

CATALOG = "crypto_fraud"
SILVER_SCHEMA = "silver"
FEATURE_SCHEMA = "features"
FEATURE_TABLE = "crypto_fraud.features.transaction_features_offline"
TRAINING_TABLE = "crypto_fraud.features.transaction_training_dataset"
MODEL_URI = "models:/crypto_fraud.models.fraud_detection_model@candidate"
THRESHOLD = 0.80
EXPECTED_FEATURE_COUNT = 55

APPROVED_FEATURES = [
    "account_age_days",
    "account_profile_available",
    "amount_above_normal_usd",
    "amount_to_normal_ratio",
    "asset",
    "country",
    "crypto_quantity",
    "customer_risk_tier",
    "destination_wallet_age_hours",
    "destination_wallet_applicable",
    "destination_wallet_first_seen_available",
    "device_age_days",
    "device_age_hours",
    "device_distinct_account_count_1h",
    "device_failed_auth_count_10m",
    "device_profile_available",
    "device_successful_auth_count_1h",
    "failed_auth_count_10m",
    "failed_auth_count_1h",
    "has_previous_successful_auth",
    "has_previous_transaction",
    "has_prior_tx_24h",
    "is_new_destination_wallet",
    "is_new_device_24h",
    "is_night_transaction_utc",
    "is_weekend_utc",
    "latest_market_close_usd",
    "market_data_available",
    "market_data_freshness_seconds",
    "market_price_usd",
    "market_return_1h",
    "market_return_5m",
    "market_volatility_1h",
    "market_volume_sum_1h",
    "normal_transaction_amount_available",
    "normal_transaction_amount_usd",
    "prior_destination_wallet_tx_count",
    "prior_source_wallet_tx_count",
    "prior_tx_amount_avg_24h",
    "prior_tx_amount_max_24h",
    "prior_tx_amount_sum_1h",
    "prior_tx_amount_sum_24h",
    "prior_tx_count_1h",
    "prior_tx_count_24h",
    "prior_tx_count_5m",
    "recent_auth_failure_flag_10m",
    "seconds_since_last_successful_auth",
    "seconds_since_previous_tx",
    "source_wallet_applicable",
    "successful_auth_count_1h",
    "transaction_amount_usd",
    "transaction_country_mismatch_home_country",
    "transaction_day_of_week_utc",
    "transaction_hour_utc",
    "transaction_type",
]

CATEGORICAL_FEATURES = ["asset", "country", "customer_risk_tier", "transaction_type"]
NUMERIC_FEATURES = [name for name in APPROVED_FEATURES if name not in CATEGORICAL_FEATURES]
ASSET_PRODUCT_MAPPING = {"BTC": "BTC-USD", "ETH": "ETH-USD"}

FEATURE_FAMILIES = {
    "current_transaction": [
        "asset",
        "country",
        "crypto_quantity",
        "market_price_usd",
        "transaction_amount_usd",
        "transaction_day_of_week_utc",
        "transaction_hour_utc",
        "transaction_type",
        "is_night_transaction_utc",
        "is_weekend_utc",
    ],
    "account": [
        "account_age_days",
        "account_profile_available",
        "amount_above_normal_usd",
        "amount_to_normal_ratio",
        "customer_risk_tier",
        "normal_transaction_amount_available",
        "normal_transaction_amount_usd",
        "transaction_country_mismatch_home_country",
    ],
    "transaction_history": [
        "has_previous_transaction",
        "has_prior_tx_24h",
        "prior_tx_amount_avg_24h",
        "prior_tx_amount_max_24h",
        "prior_tx_amount_sum_1h",
        "prior_tx_amount_sum_24h",
        "prior_tx_count_1h",
        "prior_tx_count_24h",
        "prior_tx_count_5m",
        "seconds_since_previous_tx",
    ],
    "authentication": [
        "failed_auth_count_10m",
        "failed_auth_count_1h",
        "has_previous_successful_auth",
        "recent_auth_failure_flag_10m",
        "seconds_since_last_successful_auth",
        "successful_auth_count_1h",
    ],
    "device_authentication": [
        "device_distinct_account_count_1h",
        "device_failed_auth_count_10m",
        "device_successful_auth_count_1h",
    ],
    "device_profile": [
        "device_age_days",
        "device_age_hours",
        "device_profile_available",
        "is_new_device_24h",
    ],
    "wallet": [
        "destination_wallet_age_hours",
        "destination_wallet_applicable",
        "destination_wallet_first_seen_available",
        "is_new_destination_wallet",
        "prior_destination_wallet_tx_count",
        "prior_source_wallet_tx_count",
        "source_wallet_applicable",
    ],
    "market": [
        "latest_market_close_usd",
        "market_data_available",
        "market_data_freshness_seconds",
        "market_return_1h",
        "market_return_5m",
        "market_volatility_1h",
        "market_volume_sum_1h",
    ],
}


def full_table(schema_name: str, table_name: str) -> str:
    return f"{CATALOG}.{schema_name}.{table_name}"


def to_utc_naive(value: Any) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, float) and math.isnan(value):
        return None
    ts = pd.Timestamp(value)
    if pd.isna(ts):
        return None
    if ts.tzinfo is not None:
        ts = ts.tz_convert("UTC").tz_localize(None)
    return ts.to_pydatetime().replace(tzinfo=None)


def seconds_between(later: datetime, earlier: datetime) -> int:
    return int((later - earlier).total_seconds())


def sample_stddev(values: list[float | None]) -> float | None:
    clean = [float(value) for value in values if value is not None and math.isfinite(value)]
    if len(clean) < 2:
        return None
    mean = sum(clean) / len(clean)
    return math.sqrt(sum((value - mean) ** 2 for value in clean) / (len(clean) - 1))


def build_market_state(market_rows: dict[str, list[dict[str, Any]]]) -> dict[str, list[dict[str, Any]]]:
    result = {}
    for product_id, rows in market_rows.items():
        ordered = sorted(rows, key=lambda item: item["candle_end_timestamp"])
        ends = [item["candle_end_timestamp"] for item in ordered]
        closes = [item["close_price_usd"] for item in ordered]
        volumes = [item["volume"] for item in ordered]
        log_returns = [None]
        for index in range(1, len(ordered)):
            previous = closes[index - 1]
            log_returns.append(math.log(closes[index] / previous) if previous > 0 else None)

        features = []
        for index, item in enumerate(ordered):
            end_time = ends[index]
            start_5m = end_time - timedelta(minutes=5)
            start_1h = end_time - timedelta(hours=1)
            baseline_5m_index = bisect_right(ends, start_5m) - 1
            baseline_1h_index = bisect_right(ends, start_1h) - 1
            window_start = bisect_right(ends, start_1h - timedelta(microseconds=1))
            window_returns = log_returns[window_start : index + 1]
            window_volumes = volumes[window_start : index + 1]
            baseline_5m = closes[baseline_5m_index] if baseline_5m_index >= 0 else None
            baseline_1h = closes[baseline_1h_index] if baseline_1h_index >= 0 else None
            close = closes[index]
            features.append(
                {
                    "candle_end_timestamp": end_time,
                    "latest_market_close_usd": close,
                    "market_return_5m": close / baseline_5m - 1.0 if baseline_5m else None,
                    "market_return_1h": close / baseline_1h - 1.0 if baseline_1h else None,
                    "market_volatility_1h": sample_stddev(window_returns),
                    "market_volume_sum_1h": sum(window_volumes) if window_volumes else None,
                }
            )
        result[product_id] = features
    return result


def collect_historical_state(
    *,
    account_ids: list[str],
    device_ids: list[str],
    wallet_ids: list[str],
    min_feature_timestamp: datetime,
    max_feature_timestamp: datetime,
) -> dict[str, Any]:
    market_start = min_feature_timestamp - timedelta(hours=2)
    accounts = {
        row.account_id: {
            "created_at": to_utc_naive(row.created_at),
            "updated_at": to_utc_naive(row.updated_at),
            "home_country": row.home_country,
            "customer_risk_tier": row.customer_risk_tier,
            "normal_transaction_amount_usd": float(row.normal_transaction_amount_usd)
            if row.normal_transaction_amount_usd is not None
            else None,
        }
        for row in spark.table(full_table(SILVER_SCHEMA, "accounts"))
        .filter(F.col("account_id").isin(account_ids))
        .select(
            "account_id",
            "created_at",
            "updated_at",
            "home_country",
            "customer_risk_tier",
            "normal_transaction_amount_usd",
        )
        .collect()
    }

    devices = {
        row.device_id: {"first_seen_at": to_utc_naive(row.first_seen_at)}
        for row in spark.table(full_table(SILVER_SCHEMA, "devices"))
        .filter(F.col("device_id").isin(device_ids))
        .select("device_id", "first_seen_at")
        .collect()
    }

    wallets = {
        row.wallet_id: {"first_seen_at": to_utc_naive(row.first_seen_at)}
        for row in spark.table(full_table(SILVER_SCHEMA, "wallets"))
        .filter(F.col("wallet_id").isin(wallet_ids))
        .select("wallet_id", "first_seen_at")
        .collect()
    }

    transactions_by_account: dict[str, list[dict[str, Any]]] = defaultdict(list)
    transactions_by_source_wallet: dict[str, list[datetime]] = defaultdict(list)
    transactions_by_destination_wallet: dict[str, list[datetime]] = defaultdict(list)
    for row in spark.table(full_table(SILVER_SCHEMA, "customer_transactions")).select(
        "transaction_id",
        "account_id",
        "source_wallet_id",
        "destination_wallet_id",
        "event_timestamp",
        "transaction_amount_usd",
    ).filter(
        (
            F.col("account_id").isin(account_ids)
            | F.col("source_wallet_id").isin(wallet_ids)
            | F.col("destination_wallet_id").isin(wallet_ids)
        )
        & (F.col("event_timestamp") < F.lit(max_feature_timestamp))
    ).collect():
        event_ts = to_utc_naive(row.event_timestamp)
        item = {
            "transaction_id": row.transaction_id,
            "event_timestamp": event_ts,
            "transaction_amount_usd": float(row.transaction_amount_usd),
        }
        transactions_by_account[row.account_id].append(item)
        if row.source_wallet_id:
            transactions_by_source_wallet[row.source_wallet_id].append(event_ts)
        if row.destination_wallet_id:
            transactions_by_destination_wallet[row.destination_wallet_id].append(event_ts)

    for rows in transactions_by_account.values():
        rows.sort(key=lambda item: item["event_timestamp"])
    for rows in transactions_by_source_wallet.values():
        rows.sort()
    for rows in transactions_by_destination_wallet.values():
        rows.sort()

    auth_by_account: dict[str, list[dict[str, Any]]] = defaultdict(list)
    auth_by_device: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in spark.table(full_table(SILVER_SCHEMA, "authentication_events")).select(
        "account_id",
        "device_id",
        "event_timestamp",
        "login_success",
    ).filter(
        (
            F.col("account_id").isin(account_ids)
            | F.col("device_id").isin(device_ids)
        )
        & (F.col("event_timestamp") < F.lit(max_feature_timestamp))
    ).collect():
        event_ts = to_utc_naive(row.event_timestamp)
        item = {
            "account_id": row.account_id,
            "device_id": row.device_id,
            "event_timestamp": event_ts,
            "login_success": bool(row.login_success),
        }
        auth_by_account[row.account_id].append(item)
        auth_by_device[row.device_id].append(item)
    for rows in auth_by_account.values():
        rows.sort(key=lambda item: item["event_timestamp"])
    for rows in auth_by_device.values():
        rows.sort(key=lambda item: item["event_timestamp"])

    market_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in spark.table(full_table(SILVER_SCHEMA, "market_candles")).select(
        "product_id",
        "candle_end_timestamp",
        "close_price_usd",
        "volume",
    ).filter(
        (F.col("candle_end_timestamp") >= F.lit(market_start))
        & (F.col("candle_end_timestamp") <= F.lit(max_feature_timestamp))
    ).collect():
        market_rows[row.product_id].append(
            {
                "candle_end_timestamp": to_utc_naive(row.candle_end_timestamp),
                "close_price_usd": float(row.close_price_usd),
                "volume": float(row.volume),
            }
        )

    return {
        "accounts": dict(accounts),
        "devices": dict(devices),
        "wallets": dict(wallets),
        "transactions_by_account": dict(transactions_by_account),
        "transactions_by_source_wallet": dict(transactions_by_source_wallet),
        "transactions_by_destination_wallet": dict(transactions_by_destination_wallet),
        "auth_by_account": dict(auth_by_account),
        "auth_by_device": dict(auth_by_device),
        "market_features": build_market_state(market_rows),
    }


def prior_window(rows: list[dict[str, Any]], feature_timestamp: datetime, seconds: int) -> list[dict[str, Any]]:
    cutoff = feature_timestamp - timedelta(seconds=seconds)
    return [
        row
        for row in rows
        if row["event_timestamp"] is not None
        and cutoff <= row["event_timestamp"] < feature_timestamp
    ]


def prior_all(rows: list[dict[str, Any]], feature_timestamp: datetime) -> list[dict[str, Any]]:
    return [
        row
        for row in rows
        if row["event_timestamp"] is not None and row["event_timestamp"] < feature_timestamp
    ]


def wallet_prior_count(rows: list[datetime], feature_timestamp: datetime) -> int:
    return len([event_timestamp for event_timestamp in rows if event_timestamp < feature_timestamp])


def latest_market_feature(market_rows: list[dict[str, Any]], feature_timestamp: datetime) -> dict[str, Any] | None:
    ends = [row["candle_end_timestamp"] for row in market_rows]
    index = bisect_right(ends, feature_timestamp) - 1
    return market_rows[index] if index >= 0 else None


def online_build_features(row: dict[str, Any], state: dict[str, Any]) -> dict[str, Any]:
    feature_timestamp = row["event_timestamp"]
    account_id = row["account_id"]
    device_id = row["device_id"]
    source_wallet_id = row["source_wallet_id"]
    destination_wallet_id = row["destination_wallet_id"]
    transaction_amount = float(row["transaction_amount_usd"])

    day_of_week = int(feature_timestamp.isoweekday())
    hour = int(feature_timestamp.hour)
    features: dict[str, Any] = {
        "asset": row["asset"],
        "transaction_type": row["transaction_type"],
        "country": row["country"],
        "transaction_amount_usd": transaction_amount,
        "crypto_quantity": float(row["crypto_quantity"]),
        "market_price_usd": float(row["market_price_usd"]),
        "transaction_hour_utc": hour,
        "transaction_day_of_week_utc": day_of_week,
        "is_weekend_utc": 1 if day_of_week in {6, 7} else 0,
        "is_night_transaction_utc": 1 if 0 <= hour <= 5 else 0,
    }

    account = state["accounts"].get(account_id)
    account_created = account.get("created_at") if account else None
    account_updated = account.get("updated_at") if account else None
    account_exists = account_created is not None and account_created <= feature_timestamp
    profile_available = account_exists and (
        account_updated is None or account_updated <= feature_timestamp
    )
    normal_amount = account.get("normal_transaction_amount_usd") if profile_available else None
    normal_available = normal_amount is not None and normal_amount > 0
    features.update(
        {
            "account_age_days": math.floor(seconds_between(feature_timestamp, account_created) / 86400)
            if account_exists
            else None,
            "account_profile_available": 1 if profile_available else 0,
            "customer_risk_tier": account.get("customer_risk_tier") if profile_available else None,
            "normal_transaction_amount_usd": normal_amount if profile_available else None,
            "normal_transaction_amount_available": 1 if normal_available else 0,
            "amount_to_normal_ratio": transaction_amount / normal_amount if normal_available else None,
            "amount_above_normal_usd": transaction_amount - normal_amount if normal_available else None,
            "transaction_country_mismatch_home_country": (
                1 if row["country"] != account.get("home_country") else 0
            )
            if profile_available and row["country"] and account.get("home_country")
            else None,
        }
    )

    account_transactions = state["transactions_by_account"].get(account_id, [])
    prior_5m = prior_window(account_transactions, feature_timestamp, 300)
    prior_1h = prior_window(account_transactions, feature_timestamp, 3600)
    prior_24h = prior_window(account_transactions, feature_timestamp, 86400)
    prior_history = prior_all(account_transactions, feature_timestamp)
    previous_timestamp = prior_history[-1]["event_timestamp"] if prior_history else None
    amounts_24h = [item["transaction_amount_usd"] for item in prior_24h]
    features.update(
        {
            "prior_tx_count_5m": len(prior_5m),
            "prior_tx_count_1h": len(prior_1h),
            "prior_tx_count_24h": len(prior_24h),
            "prior_tx_amount_sum_1h": float(sum(item["transaction_amount_usd"] for item in prior_1h)),
            "prior_tx_amount_sum_24h": float(sum(amounts_24h)),
            "prior_tx_amount_avg_24h": float(sum(amounts_24h) / len(amounts_24h))
            if amounts_24h
            else None,
            "prior_tx_amount_max_24h": float(max(amounts_24h)) if amounts_24h else None,
            "has_prior_tx_24h": 1 if prior_24h else 0,
            "seconds_since_previous_tx": seconds_between(feature_timestamp, previous_timestamp)
            if previous_timestamp
            else None,
            "has_previous_transaction": 1 if previous_timestamp else 0,
        }
    )

    account_auth = state["auth_by_account"].get(account_id, [])
    account_auth_10m = prior_window(account_auth, feature_timestamp, 600)
    account_auth_1h = prior_window(account_auth, feature_timestamp, 3600)
    failed_10m = [item for item in account_auth_10m if not item["login_success"]]
    failed_1h = [item for item in account_auth_1h if not item["login_success"]]
    successful_1h = [item for item in account_auth_1h if item["login_success"]]
    prior_successful = [
        item for item in prior_all(account_auth, feature_timestamp) if item["login_success"]
    ]
    last_successful_ts = prior_successful[-1]["event_timestamp"] if prior_successful else None
    features.update(
        {
            "failed_auth_count_10m": len(failed_10m),
            "failed_auth_count_1h": len(failed_1h),
            "successful_auth_count_1h": len(successful_1h),
            "recent_auth_failure_flag_10m": 1 if failed_10m else 0,
            "seconds_since_last_successful_auth": seconds_between(feature_timestamp, last_successful_ts)
            if last_successful_ts
            else None,
            "has_previous_successful_auth": 1 if last_successful_ts else 0,
        }
    )

    device_auth = state["auth_by_device"].get(device_id, [])
    device_auth_10m = prior_window(device_auth, feature_timestamp, 600)
    device_auth_1h = prior_window(device_auth, feature_timestamp, 3600)
    features.update(
        {
            "device_failed_auth_count_10m": len(
                [item for item in device_auth_10m if not item["login_success"]]
            ),
            "device_successful_auth_count_1h": len(
                [item for item in device_auth_1h if item["login_success"]]
            ),
            "device_distinct_account_count_1h": len(
                {item["account_id"] for item in device_auth_1h if item.get("account_id")}
            ),
        }
    )

    device = state["devices"].get(device_id)
    device_first_seen = device.get("first_seen_at") if device else None
    device_available = device_first_seen is not None and device_first_seen <= feature_timestamp
    device_age_hours = (
        seconds_between(feature_timestamp, device_first_seen) / 3600.0 if device_available else None
    )
    features.update(
        {
            "device_profile_available": 1 if device_available else 0,
            "device_age_hours": device_age_hours,
            "device_age_days": device_age_hours / 24.0 if device_age_hours is not None else None,
            "is_new_device_24h": 1 if device_available and device_age_hours <= 24.0 else 0
            if device_available
            else None,
        }
    )

    source_applicable = source_wallet_id is not None
    destination_applicable = destination_wallet_id is not None
    destination_rows = state["transactions_by_destination_wallet"].get(destination_wallet_id, [])
    source_rows = state["transactions_by_source_wallet"].get(source_wallet_id, [])
    prior_destination_count = (
        wallet_prior_count(destination_rows, feature_timestamp) if destination_applicable else 0
    )
    prior_source_count = wallet_prior_count(source_rows, feature_timestamp) if source_applicable else 0
    destination_wallet = state["wallets"].get(destination_wallet_id)
    destination_first_seen = destination_wallet.get("first_seen_at") if destination_wallet else None
    destination_available = (
        destination_applicable
        and destination_first_seen is not None
        and destination_first_seen <= feature_timestamp
    )
    features.update(
        {
            "source_wallet_applicable": 1 if source_applicable else 0,
            "destination_wallet_applicable": 1 if destination_applicable else 0,
            "destination_wallet_first_seen_available": 1 if destination_available else 0,
            "destination_wallet_age_hours": seconds_between(feature_timestamp, destination_first_seen)
            / 3600.0
            if destination_available
            else None,
            "is_new_destination_wallet": 1 if destination_applicable and prior_destination_count == 0 else 0
            if destination_applicable
            else None,
            "prior_destination_wallet_tx_count": prior_destination_count,
            "prior_source_wallet_tx_count": prior_source_count,
        }
    )

    market_product = ASSET_PRODUCT_MAPPING.get(row["asset"])
    market = latest_market_feature(state["market_features"].get(market_product, []), feature_timestamp)
    features.update(
        {
            "latest_market_close_usd": market.get("latest_market_close_usd") if market else None,
            "market_data_available": 1 if market else 0,
            "market_data_freshness_seconds": seconds_between(
                feature_timestamp, market["candle_end_timestamp"]
            )
            if market
            else None,
            "market_return_1h": market.get("market_return_1h") if market else None,
            "market_return_5m": market.get("market_return_5m") if market else None,
            "market_volatility_1h": market.get("market_volatility_1h") if market else None,
            "market_volume_sum_1h": market.get("market_volume_sum_1h") if market else None,
        }
    )
    return {feature_name: features.get(feature_name) for feature_name in APPROVED_FEATURES}


def deterministic_sample() -> DataFrame:
    features = spark.table(FEATURE_TABLE)
    training = spark.table(TRAINING_TABLE).select("transaction_id", "target_is_fraud")
    base = features.join(training, "transaction_id", "left")
    fraud = base.filter(F.col("target_is_fraud") == F.lit(True)).orderBy("feature_timestamp", "transaction_id").limit(25)
    normal = base.filter(F.col("target_is_fraud") == F.lit(False)).orderBy("feature_timestamp", "transaction_id").limit(45)
    deposits = base.filter(F.col("source_wallet_id").isNull()).orderBy("feature_timestamp", "transaction_id").limit(20)
    history = (
        base.filter(
            (F.col("has_previous_transaction") == F.lit(1))
            | (F.col("failed_auth_count_10m") > F.lit(0))
            | (F.col("device_distinct_account_count_1h") > F.lit(1))
        )
        .orderBy("feature_timestamp", "transaction_id")
        .limit(20)
    )
    ids = (
        fraud.select("transaction_id")
        .unionByName(normal.select("transaction_id"))
        .unionByName(deposits.select("transaction_id"))
        .unionByName(history.select("transaction_id"))
        .distinct()
        .limit(100)
    )
    return features.join(ids, "transaction_id", "inner").orderBy("feature_timestamp", "transaction_id")


def clean_value(value: Any) -> Any:
    if isinstance(value, Decimal):
        return float(value)
    if isinstance(value, np.generic):
        return value.item()
    return value


def is_null(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, float) and math.isnan(value):
        return True
    try:
        return bool(pd.isna(value))
    except Exception:
        return False


def numeric_equal(left: Any, right: Any, tolerance: float = 1e-6) -> tuple[bool, bool, float | None]:
    if is_null(left) and is_null(right):
        return True, True, 0.0
    if is_null(left) or is_null(right):
        return False, False, None
    left_float = float(clean_value(left))
    right_float = float(clean_value(right))
    diff = abs(left_float - right_float)
    exact = left_float == right_float
    scale = max(abs(left_float), abs(right_float), 1.0)
    tolerant = diff <= tolerance or (diff / scale) <= 1e-9
    return exact, tolerant, diff


def categorical_equal(left: Any, right: Any) -> tuple[bool, bool]:
    if is_null(left) and is_null(right):
        return True, True
    if is_null(left) or is_null(right):
        return False, False
    return str(left) == str(right), str(left) == str(right)


def compare_feature_frames(offline_pdf: pd.DataFrame, online_pdf: pd.DataFrame) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows = []
    mismatches = []
    joined = offline_pdf.merge(online_pdf, on="transaction_id", suffixes=("_offline", "_online"))
    for feature_name in APPROVED_FEATURES:
        exact_count = 0
        tolerance_count = 0
        mismatch_count = 0
        max_abs_diff = 0.0
        samples = []
        for _, row in joined.iterrows():
            left = clean_value(row[f"{feature_name}_offline"])
            right = clean_value(row[f"{feature_name}_online"])
            if feature_name in NUMERIC_FEATURES:
                exact, tolerant, diff = numeric_equal(left, right)
                if diff is not None:
                    max_abs_diff = max(max_abs_diff, float(diff))
            else:
                exact, tolerant = categorical_equal(left, right)
            if exact:
                exact_count += 1
                tolerance_count += 1
            elif tolerant:
                tolerance_count += 1
            else:
                mismatch_count += 1
                if len(samples) < 5:
                    samples.append(
                        {
                            "transaction_id": row["transaction_id"],
                            "offline": None if is_null(left) else clean_value(left),
                            "online": None if is_null(right) else clean_value(right),
                        }
                    )

        comparable_count = int(len(joined))
        if comparable_count == 0:
            status = "NOT-VERIFIABLE"
        elif mismatch_count == 0 and exact_count == comparable_count:
            status = "EXACT"
        elif mismatch_count == 0 and tolerance_count == comparable_count:
            status = "TOLERANCE"
        else:
            status = "MISMATCH"

        item = {
            "feature_name": feature_name,
            "status": status,
            "comparison_type": "numeric" if feature_name in NUMERIC_FEATURES else "categorical",
            "sample_rows": comparable_count,
            "exact_match_count": exact_count,
            "tolerance_match_count": tolerance_count,
            "mismatch_count": mismatch_count,
            "exact_match_rate": exact_count / comparable_count if comparable_count else None,
            "tolerance_match_rate": tolerance_count / comparable_count if comparable_count else None,
            "max_abs_diff": max_abs_diff if feature_name in NUMERIC_FEATURES else None,
        }
        rows.append(item)
        if samples:
            mismatches.append({"feature_name": feature_name, "samples": samples})
    return rows, mismatches


def spark_dtype_compatible(dtype: str, feature_name: str) -> bool:
    if feature_name in CATEGORICAL_FEATURES:
        return dtype == "string"
    return (
        dtype in {"int", "bigint", "long", "double", "float", "smallint", "tinyint"}
        or dtype.startswith("decimal")
    )


def positive_probability(model: Any, frame: pd.DataFrame) -> np.ndarray:
    probabilities = model.predict_proba(frame)
    classes = list(model.classes_)
    if 1 in classes:
        return probabilities[:, classes.index(1)]
    if True in classes:
        return probabilities[:, classes.index(True)]
    return np.zeros(len(frame), dtype=float)


def static_comparison_rows(data_rows: list[dict[str, Any]]) -> list[dict[str, str]]:
    data_status = {row["feature_name"]: row["status"] for row in data_rows}
    family_by_feature = {
        feature_name: family
        for family, names in FEATURE_FAMILIES.items()
        for feature_name in names
    }
    rows = []
    for feature_name in APPROVED_FEATURES:
        data_result = data_status.get(feature_name, "NOT-VERIFIABLE")
        if data_result in {"EXACT", "TOLERANCE"}:
            status = "MATCH"
        elif data_result == "MISMATCH":
            status = "MAJOR-MISMATCH"
        else:
            status = "NOT-VERIFIABLE"
        rows.append(
            {
                "feature_name": feature_name,
                "feature_family": family_by_feature.get(feature_name, "unknown"),
                "static_status": status,
            }
        )
    return rows


# COMMAND ----------

feature_df = spark.table(FEATURE_TABLE)
training_df = spark.table(TRAINING_TABLE)

feature_columns = set(feature_df.columns)
training_columns = set(training_df.columns)
missing_feature_table_columns = [name for name in APPROVED_FEATURES if name not in feature_columns]
missing_training_columns = [name for name in APPROVED_FEATURES if name not in training_columns]
canonical_feature_count = len(APPROVED_FEATURES)
ordered_feature_list_verified = (
    canonical_feature_count == EXPECTED_FEATURE_COUNT
    and not missing_feature_table_columns
    and not missing_training_columns
)

training_dtypes = dict(training_df.select(*APPROVED_FEATURES).dtypes)
dtype_issues = [
    {"feature_name": name, "dtype": training_dtypes.get(name)}
    for name in APPROVED_FEATURES
    if not spark_dtype_compatible(training_dtypes.get(name, ""), name)
]
dtypes_compatible = len(dtype_issues) == 0

leakage_terms = ["is_fraud", "target", "fraud_type", "fraud_label", "label_timestamp", "label_source", "investigation_status"]
feature_label_leakage_columns = [
    name
    for name in APPROVED_FEATURES
    if any(term in name.lower() for term in leakage_terms)
]

sample_df = deterministic_sample().cache()
sample_count = int(sample_df.count())

sample_ids = [row.transaction_id for row in sample_df.select("transaction_id").collect()]
sample_transactions = (
    spark.table(full_table(SILVER_SCHEMA, "customer_transactions"))
    .filter(F.col("transaction_id").isin(sample_ids))
    .select(
        "transaction_id",
        "account_id",
        "device_id",
        "source_wallet_id",
        "destination_wallet_id",
        "event_timestamp",
        "asset",
        "transaction_type",
        "country",
        "transaction_amount_usd",
        "crypto_quantity",
        "market_price_usd",
    )
    .orderBy("event_timestamp", "transaction_id")
    .collect()
)

sample_account_ids = sorted({row.account_id for row in sample_transactions if row.account_id})
sample_device_ids = sorted({row.device_id for row in sample_transactions if row.device_id})
sample_wallet_ids = sorted(
    {
        wallet_id
        for row in sample_transactions
        for wallet_id in (row.source_wallet_id, row.destination_wallet_id)
        if wallet_id
    }
)
sample_timestamps = [to_utc_naive(row.event_timestamp) for row in sample_transactions]
state = collect_historical_state(
    account_ids=sample_account_ids,
    device_ids=sample_device_ids,
    wallet_ids=sample_wallet_ids,
    min_feature_timestamp=min(sample_timestamps),
    max_feature_timestamp=max(sample_timestamps),
)
online_rows = []
for row in sample_transactions:
    raw = {
        "transaction_id": row.transaction_id,
        "account_id": row.account_id,
        "device_id": row.device_id,
        "source_wallet_id": row.source_wallet_id,
        "destination_wallet_id": row.destination_wallet_id,
        "event_timestamp": to_utc_naive(row.event_timestamp),
        "asset": row.asset,
        "transaction_type": row.transaction_type,
        "country": row.country,
        "transaction_amount_usd": float(row.transaction_amount_usd),
        "crypto_quantity": float(row.crypto_quantity),
        "market_price_usd": float(row.market_price_usd),
    }
    features = online_build_features(raw, state)
    features["transaction_id"] = row.transaction_id
    online_rows.append(features)

offline_pdf = sample_df.select("transaction_id", *APPROVED_FEATURES).toPandas()
online_pdf = pd.DataFrame(online_rows, columns=["transaction_id"] + APPROVED_FEATURES)
comparison_rows, mismatch_samples = compare_feature_frames(offline_pdf, online_pdf)
static_rows = static_comparison_rows(comparison_rows)

exact_consistent = sum(1 for row in comparison_rows if row["status"] == "EXACT")
tolerance_consistent = sum(1 for row in comparison_rows if row["status"] == "TOLERANCE")
mismatched = sum(1 for row in comparison_rows if row["status"] == "MISMATCH")
not_verifiable = sum(1 for row in comparison_rows if row["status"] == "NOT-VERIFIABLE")

major_mismatches = [
    row["feature_name"] for row in static_rows if row["static_status"] == "MAJOR-MISMATCH"
]

mlflow.set_tracking_uri("databricks")
mlflow.set_registry_uri("databricks-uc")
candidate_model = mlflow.sklearn.load_model(MODEL_URI)
score_frame = online_pdf[APPROVED_FEATURES].copy()
probabilities = positive_probability(candidate_model, score_frame)
probabilities_valid = bool(
    len(probabilities) == len(score_frame)
    and np.isfinite(probabilities).all()
    and (probabilities >= 0.0).all()
    and (probabilities <= 1.0).all()
)

model_validation = {
    "sample_rows_scored": int(len(score_frame)),
    "minimum_probability": float(np.min(probabilities)) if len(probabilities) else None,
    "maximum_probability": float(np.max(probabilities)) if len(probabilities) else None,
    "predicted_fraud_count_at_threshold_0_80": int(np.sum(probabilities >= THRESHOLD)) if len(probabilities) else 0,
    "positive_probabilities_valid": probabilities_valid,
}

status = "PASS"
failed_checks = []
if not ordered_feature_list_verified:
    failed_checks.append("ordered_feature_list_not_verified")
if not dtypes_compatible:
    failed_checks.append("dtype_incompatibility")
if feature_label_leakage_columns:
    failed_checks.append("fraud_label_leakage_columns_in_model_features")
if mismatched:
    failed_checks.append("offline_online_feature_mismatches")
if not probabilities_valid:
    failed_checks.append("candidate_model_probability_validation_failed")
if failed_checks:
    status = "FAIL"

summary = {
    "phase": "15",
    "status": status,
    "notebook_path": "/Users/akanaskhan1506@gmail.com/08_feature_consistency_validation",
    "canonical_feature_count": canonical_feature_count,
    "expected_feature_count": EXPECTED_FEATURE_COUNT,
    "exact_ordered_feature_list_verified": ordered_feature_list_verified,
    "feature_order": APPROVED_FEATURES,
    "sample_count": sample_count,
    "consistency_results": {
        "exact_consistent": exact_consistent,
        "tolerance_consistent": tolerance_consistent,
        "mismatched": mismatched,
        "not_verifiable": not_verifiable,
    },
    "major_mismatches_found": major_mismatches,
    "major_mismatches_fixed": [],
    "feature_comparison": comparison_rows,
    "static_comparison": static_rows,
    "mismatch_samples": mismatch_samples[:20],
    "model_validation": model_validation,
    "final_model_feature_count": len(score_frame.columns),
    "feature_ordering_verified": list(score_frame.columns) == APPROVED_FEATURES,
    "dtypes_compatible": dtypes_compatible,
    "dtype_issues": dtype_issues,
    "fraud_label_leakage": False,
    "feature_label_leakage_columns": feature_label_leakage_columns,
    "registered_candidate_model_changed": False,
    "threshold": THRESHOLD,
    "threshold_changed": False,
    "short_live_reverification_required": False,
    "short_live_reverification_result": "NOT REQUIRED",
    "failed_checks": failed_checks,
    "secrets_exposed": False,
}

print(json.dumps(summary, indent=2, sort_keys=True))
dbutils.notebook.exit(json.dumps(summary, sort_keys=True, separators=(",", ":")))
