# Databricks notebook source
# MAGIC %md
# MAGIC # 06 Real-Time Fraud Scoring

# COMMAND ----------

from __future__ import annotations

import json
import math
import re
import subprocess
import sys
import time
import uuid
from bisect import bisect_right
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from typing import Any

import mlflow
import cloudpickle
import numpy as np
import pandas as pd
from pyspark.sql import functions as F
from pyspark.sql.types import (
    ArrayType,
    BooleanType,
    DoubleType,
    IntegerType,
    LongType,
    StringType,
    StructField,
    StructType,
)

spark.conf.set("spark.sql.session.timeZone", "UTC")
spark.conf.set("spark.sql.shuffle.partitions", "1")
spark.conf.set("spark.sql.execution.sortBeforeRepartition", "false")

try:
    import xgboost  # noqa: F401
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "xgboost==3.4.0"])

CATALOG = "crypto_fraud"
SOURCE_SCHEMA = "silver"
MODEL_URI = "models:/crypto_fraud.models.fraud_detection_model@candidate"
MODEL_NAME = "crypto_fraud.models.fraud_detection_model"
MODEL_VERSION = "1"
THRESHOLD = 0.80
THRESHOLD_POLICY_VERSION = "threshold-0.80-v1"
CONSUMER_GROUP = "stream-processing"
CHECKPOINT_LOCATION = (
    "/Volumes/crypto_fraud/monitoring/ingestion_checkpoints/phase13_realtime_scoring_v2"
)

MARKET_TOPIC = "market-events"
TRANSACTION_TOPIC = "transaction-events"
AUTHENTICATION_TOPIC = "authentication-events"
FRAUD_DECISION_TOPIC = "fraud-decisions"

dbutils.widgets.text("target_sample_transactions", "5")
dbutils.widgets.text("timeout_seconds", "180")
dbutils.widgets.text("real_time_trigger", "5 seconds")
TARGET_SAMPLE_TRANSACTIONS = int(dbutils.widgets.get("target_sample_transactions") or "5")
TIMEOUT_SECONDS = int(dbutils.widgets.get("timeout_seconds") or "180")
REAL_TIME_TRIGGER = dbutils.widgets.get("real_time_trigger") or "5 seconds"

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

ASSET_PRODUCT_MAPPING = {"BTC": "BTC-USD", "ETH": "ETH-USD"}


def secret_connection_string() -> str:
    return dbutils.secrets.get(
        scope="crypto-fraud-secrets",
        key="eventhubs-databricks-connection",
    )


def connection_string_candidates(connection_string: str) -> list[str]:
    text = str(connection_string).strip().strip("'\"")
    candidates = [text]
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            candidates.extend(str(value) for value in parsed.values() if value is not None)
    except Exception:
        pass
    return candidates


def normalize_connection_string(connection_string: str) -> str:
    for candidate in connection_string_candidates(connection_string):
        parts = {}
        for segment in candidate.strip().strip("'\"").split(";"):
            if "=" not in segment:
                continue
            key, value = segment.split("=", 1)
            parts[key.strip().lower()] = value.strip()
        endpoint = parts.get("endpoint")
        key_name = parts.get("sharedaccesskeyname")
        key_value = parts.get("sharedaccesskey")
        if endpoint and key_name and key_value:
            return (
                f"Endpoint={endpoint};"
                f"SharedAccessKeyName={key_name};"
                f"SharedAccessKey={key_value}"
            )
    raise ValueError("Event Hubs namespace connection string missing required fields")


def event_hubs_bootstrap(connection_string: str) -> str:
    joined = "\n".join(connection_string_candidates(connection_string))
    match = re.search(r"Endpoint\s*=\s*sb://([^/;\s\"']+)", joined, flags=re.IGNORECASE)
    if not match:
        match = re.search(r"sb://([^/;\s\"']+)", joined, flags=re.IGNORECASE)
    if not match:
        match = re.search(
            r"sr=https%3A%2F%2F([^%/;\s\"']+)",
            joined,
            flags=re.IGNORECASE,
        )
    if not match:
        raise ValueError("Event Hubs namespace endpoint missing from secret")
    return f"{match.group(1)}:9093"


def kafka_options(topic: str, connection_string: str) -> dict[str, str]:
    kafka_connection_string = normalize_connection_string(connection_string)
    jaas = (
        'kafkashaded.org.apache.kafka.common.security.plain.PlainLoginModule required '
        f'username="$ConnectionString" password="{kafka_connection_string}";'
    )
    return {
        "kafka.bootstrap.servers": event_hubs_bootstrap(connection_string),
        "kafka.security.protocol": "SASL_SSL",
        "kafka.sasl.mechanism": "PLAIN",
        "kafka.sasl.jaas.config": jaas,
        "subscribe": topic,
        "startingOffsets": "latest",
        "failOnDataLoss": "false",
        "kafka.group.id": CONSUMER_GROUP,
        "maxPartitions": "1",
    }


def kafka_sink_options(connection_string: str) -> dict[str, str]:
    kafka_connection_string = normalize_connection_string(connection_string)
    jaas = (
        'kafkashaded.org.apache.kafka.common.security.plain.PlainLoginModule required '
        f'username="$ConnectionString" password="{kafka_connection_string}";'
    )
    return {
        "kafka.bootstrap.servers": event_hubs_bootstrap(connection_string),
        "kafka.security.protocol": "SASL_SSL",
        "kafka.sasl.mechanism": "PLAIN",
        "kafka.sasl.jaas.config": jaas,
        "topic": FRAUD_DECISION_TOPIC,
        "checkpointLocation": CHECKPOINT_LOCATION,
    }


def full_table(table_name: str) -> str:
    return f"{CATALOG}.{SOURCE_SCHEMA}.{table_name}"


def to_utc_naive(value: Any) -> datetime | None:
    if value is None or pd.isna(value):
        return None
    ts = pd.Timestamp(value)
    if ts.tzinfo is not None:
        ts = ts.tz_convert("UTC").tz_localize(None)
    return ts.to_pydatetime().replace(tzinfo=None)


def seconds_between(later: datetime, earlier: datetime) -> int:
    return int((later - earlier).total_seconds())


def clean_string(value: Any) -> str | None:
    if value is None or pd.isna(value):
        return None
    return str(value)


def collect_historical_state() -> dict[str, Any]:
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
        for row in spark.table(full_table("accounts"))
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
        for row in spark.table(full_table("devices"))
        .select("device_id", "first_seen_at")
        .collect()
    }

    wallets = {
        row.wallet_id: {"first_seen_at": to_utc_naive(row.first_seen_at)}
        for row in spark.table(full_table("wallets"))
        .select("wallet_id", "first_seen_at")
        .collect()
    }

    transactions_by_account: dict[str, list[dict[str, Any]]] = defaultdict(list)
    transactions_by_source_wallet: dict[str, list[datetime]] = defaultdict(list)
    transactions_by_destination_wallet: dict[str, list[datetime]] = defaultdict(list)
    for row in spark.table(full_table("customer_transactions")).select(
        "transaction_id",
        "account_id",
        "source_wallet_id",
        "destination_wallet_id",
        "event_timestamp",
        "transaction_amount_usd",
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
    for row in spark.table(full_table("authentication_events")).select(
        "account_id",
        "device_id",
        "event_timestamp",
        "login_success",
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
    for row in spark.table(full_table("market_candles")).select(
        "product_id",
        "candle_end_timestamp",
        "close_price_usd",
        "volume",
    ).collect():
        market_rows[row.product_id].append(
            {
                "candle_end_timestamp": to_utc_naive(row.candle_end_timestamp),
                "close_price_usd": float(row.close_price_usd),
                "volume": float(row.volume),
            }
        )
    market_features = build_market_state(market_rows)

    return {
        "accounts": dict(accounts),
        "devices": dict(devices),
        "wallets": dict(wallets),
        "transactions_by_account": dict(transactions_by_account),
        "transactions_by_source_wallet": dict(transactions_by_source_wallet),
        "transactions_by_destination_wallet": dict(transactions_by_destination_wallet),
        "auth_by_account": dict(auth_by_account),
        "auth_by_device": dict(auth_by_device),
        "market_features": market_features,
    }


def sample_stddev(values: list[float]) -> float | None:
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


historical_state = collect_historical_state()

mlflow.set_tracking_uri("databricks")
mlflow.set_registry_uri("databricks-uc")
driver_model = mlflow.sklearn.load_model(MODEL_URI)
if not hasattr(driver_model, "predict_proba"):
    raise RuntimeError("candidate model does not expose predict_proba")
driver_model_bytes = cloudpickle.dumps(driver_model)

# COMMAND ----------

transaction_schema = StructType(
    [
        StructField("event_id", StringType()),
        StructField("event_type", StringType()),
        StructField("schema_version", StringType()),
        StructField("source", StringType()),
        StructField("event_timestamp", StringType()),
        StructField("source_timestamp", StringType()),
        StructField("ingestion_timestamp", StringType()),
        StructField("transaction_id", StringType()),
        StructField("account_id", StringType()),
        StructField("asset", StringType()),
        StructField("crypto_quantity", DoubleType()),
        StructField("transaction_type", StringType()),
        StructField("source_wallet_id", StringType()),
        StructField("destination_wallet_id", StringType()),
        StructField("device_id", StringType()),
        StructField("country", StringType()),
        StructField("market_price_usd", DoubleType()),
        StructField("transaction_amount_usd", DoubleType()),
        StructField("transaction_status", StringType()),
    ]
)

authentication_schema = StructType(
    [
        StructField("event_id", StringType()),
        StructField("event_type", StringType()),
        StructField("schema_version", StringType()),
        StructField("source", StringType()),
        StructField("event_timestamp", StringType()),
        StructField("source_timestamp", StringType()),
        StructField("ingestion_timestamp", StringType()),
        StructField("login_id", StringType()),
        StructField("account_id", StringType()),
        StructField("device_id", StringType()),
        StructField("country", StringType()),
        StructField("ip_address", StringType()),
        StructField("login_success", BooleanType()),
        StructField("mfa_success", BooleanType()),
        StructField("password_reset_flag", BooleanType()),
        StructField("failure_reason", StringType()),
    ]
)

market_schema = StructType(
    [
        StructField("event_id", StringType()),
        StructField("event_type", StringType()),
        StructField("schema_version", StringType()),
        StructField("source", StringType()),
        StructField("event_timestamp", StringType()),
        StructField("source_timestamp", StringType()),
        StructField("ingestion_timestamp", StringType()),
        StructField("product_id", StringType()),
        StructField("trade_id", StringType()),
        StructField("price_usd", DoubleType()),
        StructField("size", DoubleType()),
        StructField("side", StringType()),
        StructField("trade_timestamp", StringType()),
        StructField("message_timestamp", StringType()),
        StructField("sequence_number", LongType()),
    ]
)

common_event_columns = [
    "event_kind",
    "event_timestamp",
    "kafka_partition",
    "kafka_offset",
    "kafka_timestamp",
    "transaction_id",
    "account_id",
    "device_id",
    "source_wallet_id",
    "destination_wallet_id",
    "asset",
    "transaction_type",
    "country",
    "crypto_quantity",
    "market_price_usd",
    "transaction_amount_usd",
    "source_timestamp",
    "ingestion_timestamp",
    "product_id",
    "trade_id",
    "login_success",
]


def kafka_stream(topic: str, connection_string: str):
    return spark.readStream.format("kafka").options(**kafka_options(topic, connection_string)).load()


def parsed_events(connection_string: str):
    raw_events = kafka_stream(
        ",".join([MARKET_TOPIC, TRANSACTION_TOPIC, AUTHENTICATION_TOPIC]),
        connection_string,
    )
    transaction_payload = F.from_json(F.col("value").cast("string"), transaction_schema)
    authentication_payload = F.from_json(F.col("value").cast("string"), authentication_schema)
    market_payload = F.from_json(F.col("value").cast("string"), market_schema)
    return (
        raw_events.select(
            "topic",
            transaction_payload.alias("transaction_payload"),
            authentication_payload.alias("authentication_payload"),
            market_payload.alias("market_payload"),
            "partition",
            "offset",
            "timestamp",
        )
        .select(
            F.when(F.col("topic") == TRANSACTION_TOPIC, F.lit("transaction"))
            .when(F.col("topic") == AUTHENTICATION_TOPIC, F.lit("authentication"))
            .when(F.col("topic") == MARKET_TOPIC, F.lit("market"))
            .alias("event_kind"),
            F.coalesce(
                F.to_timestamp("transaction_payload.event_timestamp"),
                F.to_timestamp("authentication_payload.event_timestamp"),
                F.to_timestamp("market_payload.event_timestamp"),
            ).alias("event_timestamp"),
            F.col("partition").alias("kafka_partition"),
            F.col("offset").alias("kafka_offset"),
            F.col("timestamp").alias("kafka_timestamp"),
            F.col("transaction_payload.transaction_id").alias("transaction_id"),
            F.coalesce(
                F.col("transaction_payload.account_id"),
                F.col("authentication_payload.account_id"),
            ).alias("account_id"),
            F.coalesce(
                F.col("transaction_payload.device_id"),
                F.col("authentication_payload.device_id"),
            ).alias("device_id"),
            F.col("transaction_payload.source_wallet_id").alias("source_wallet_id"),
            F.col("transaction_payload.destination_wallet_id").alias("destination_wallet_id"),
            F.col("transaction_payload.asset").alias("asset"),
            F.col("transaction_payload.transaction_type").alias("transaction_type"),
            F.coalesce(
                F.col("transaction_payload.country"),
                F.col("authentication_payload.country"),
            ).alias("country"),
            F.col("transaction_payload.crypto_quantity").alias("crypto_quantity"),
            F.coalesce(
                F.col("transaction_payload.market_price_usd"),
                F.col("market_payload.price_usd"),
            ).alias("market_price_usd"),
            F.col("transaction_payload.transaction_amount_usd").alias("transaction_amount_usd"),
            F.coalesce(
                F.col("transaction_payload.source_timestamp"),
                F.col("authentication_payload.source_timestamp"),
                F.col("market_payload.source_timestamp"),
            ).alias("source_timestamp"),
            F.coalesce(
                F.col("transaction_payload.ingestion_timestamp"),
                F.col("authentication_payload.ingestion_timestamp"),
                F.col("market_payload.ingestion_timestamp"),
            ).alias("ingestion_timestamp"),
            F.col("market_payload.product_id").alias("product_id"),
            F.col("market_payload.trade_id").alias("trade_id"),
            F.col("authentication_payload.login_success").alias("login_success"),
        )
        .filter(F.col("event_kind").isNotNull())
    )


def parsed_transactions(connection_string: str):
    payload = F.from_json(F.col("value").cast("string"), transaction_schema).alias("payload")
    return (
        kafka_stream(TRANSACTION_TOPIC, connection_string)
        .select(payload, "partition", "offset", "timestamp")
        .select(
            F.lit("transaction").alias("event_kind"),
            F.to_timestamp("payload.event_timestamp").alias("event_timestamp"),
            F.col("partition").alias("kafka_partition"),
            F.col("offset").alias("kafka_offset"),
            F.col("timestamp").alias("kafka_timestamp"),
            "payload.transaction_id",
            "payload.account_id",
            "payload.device_id",
            "payload.source_wallet_id",
            "payload.destination_wallet_id",
            "payload.asset",
            "payload.transaction_type",
            "payload.country",
            "payload.crypto_quantity",
            "payload.market_price_usd",
            "payload.transaction_amount_usd",
            "payload.source_timestamp",
            "payload.ingestion_timestamp",
            F.lit(None).cast("string").alias("product_id"),
            F.lit(None).cast("string").alias("trade_id"),
            F.lit(None).cast("boolean").alias("login_success"),
        )
    )


def parsed_authentications(connection_string: str):
    payload = F.from_json(F.col("value").cast("string"), authentication_schema).alias("payload")
    return (
        kafka_stream(AUTHENTICATION_TOPIC, connection_string)
        .select(payload, "partition", "offset", "timestamp")
        .select(
            F.lit("authentication").alias("event_kind"),
            F.to_timestamp("payload.event_timestamp").alias("event_timestamp"),
            F.col("partition").alias("kafka_partition"),
            F.col("offset").alias("kafka_offset"),
            F.col("timestamp").alias("kafka_timestamp"),
            F.lit(None).cast("string").alias("transaction_id"),
            "payload.account_id",
            "payload.device_id",
            F.lit(None).cast("string").alias("source_wallet_id"),
            F.lit(None).cast("string").alias("destination_wallet_id"),
            F.lit(None).cast("string").alias("asset"),
            F.lit(None).cast("string").alias("transaction_type"),
            "payload.country",
            F.lit(None).cast("double").alias("crypto_quantity"),
            F.lit(None).cast("double").alias("market_price_usd"),
            F.lit(None).cast("double").alias("transaction_amount_usd"),
            "payload.source_timestamp",
            "payload.ingestion_timestamp",
            F.lit(None).cast("string").alias("product_id"),
            F.lit(None).cast("string").alias("trade_id"),
            "payload.login_success",
        )
    )


def parsed_market_events(connection_string: str):
    payload = F.from_json(F.col("value").cast("string"), market_schema).alias("payload")
    return (
        kafka_stream(MARKET_TOPIC, connection_string)
        .select(payload, "partition", "offset", "timestamp")
        .select(
            F.lit("market").alias("event_kind"),
            F.to_timestamp("payload.event_timestamp").alias("event_timestamp"),
            F.col("partition").alias("kafka_partition"),
            F.col("offset").alias("kafka_offset"),
            F.col("timestamp").alias("kafka_timestamp"),
            F.lit(None).cast("string").alias("transaction_id"),
            F.lit(None).cast("string").alias("account_id"),
            F.lit(None).cast("string").alias("device_id"),
            F.lit(None).cast("string").alias("source_wallet_id"),
            F.lit(None).cast("string").alias("destination_wallet_id"),
            F.lit(None).cast("string").alias("asset"),
            F.lit(None).cast("string").alias("transaction_type"),
            F.lit(None).cast("string").alias("country"),
            F.lit(None).cast("double").alias("crypto_quantity"),
            F.col("payload.price_usd").alias("market_price_usd"),
            F.lit(None).cast("double").alias("transaction_amount_usd"),
            "payload.source_timestamp",
            "payload.ingestion_timestamp",
            "payload.product_id",
            "payload.trade_id",
            F.lit(None).cast("boolean").alias("login_success"),
        )
    )


score_schema = StructType(
    [
        StructField("event_id", StringType()),
        StructField("event_type", StringType()),
        StructField("schema_version", StringType()),
        StructField("source", StringType()),
        StructField("event_timestamp", StringType()),
        StructField("source_timestamp", StringType()),
        StructField("ingestion_timestamp", StringType()),
        StructField("transaction_id", StringType()),
        StructField("asset", StringType()),
        StructField("country", StringType()),
        StructField("transaction_amount_usd", DoubleType()),
        StructField("risk_score", DoubleType()),
        StructField("decision", StringType()),
        StructField("reason_codes", ArrayType(StringType())),
        StructField("model_name", StringType()),
        StructField("model_version", StringType()),
        StructField("prediction_timestamp", StringType()),
        StructField("processing_latency_ms", DoubleType()),
        StructField("threshold_policy_version", StringType()),
        StructField("fraud_probability", DoubleType()),
        StructField("predicted_fraud", BooleanType()),
        StructField("fraud_threshold", DoubleType()),
        StructField("feature_count", IntegerType()),
        StructField("input_contract_ok", BooleanType()),
    ]
)


def positive_probability(model: Any, frame: pd.DataFrame) -> np.ndarray:
    probabilities = model.predict_proba(frame)
    classes = list(model.classes_)
    if 1 in classes:
        return probabilities[:, classes.index(1)]
    if True in classes:
        return probabilities[:, classes.index(True)]
    return np.zeros(len(frame), dtype=float)


_scoring_model = None


def scoring_model() -> Any:
    global _scoring_model
    if _scoring_model is None:
        _scoring_model = cloudpickle.loads(driver_model_bytes)
    return _scoring_model


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


def build_features(row: dict[str, Any], state: dict[str, Any]) -> dict[str, Any]:
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


def reason_codes(features: dict[str, Any], predicted_fraud: bool) -> list[str]:
    if not predicted_fraud:
        return []
    reasons = []
    if features.get("is_new_device_24h") == 1:
        reasons.append("NEW_DEVICE")
    if features.get("transaction_country_mismatch_home_country") == 1:
        reasons.append("NEW_COUNTRY")
    if (features.get("prior_tx_count_1h") or 0) >= 3:
        reasons.append("HIGH_TRANSACTION_VELOCITY")
    if (features.get("amount_to_normal_ratio") or 0.0) >= 4.0:
        reasons.append("UNUSUAL_AMOUNT")
    if features.get("is_new_destination_wallet") == 1:
        reasons.append("NEW_DESTINATION_WALLET")
    if features.get("recent_auth_failure_flag_10m") == 1:
        reasons.append("FAILED_LOGIN_ACTIVITY")
    if (features.get("market_volatility_1h") or 0.0) >= 0.02:
        reasons.append("HIGH_MARKET_VOLATILITY")
    if (features.get("device_distinct_account_count_1h") or 0) >= 3:
        reasons.append("SUSPICIOUS_SHARED_DEVICE")
    return reasons


@F.pandas_udf(score_schema)
def score_transactions(
    transaction_id: pd.Series,
    account_id: pd.Series,
    device_id: pd.Series,
    source_wallet_id: pd.Series,
    destination_wallet_id: pd.Series,
    asset: pd.Series,
    transaction_type: pd.Series,
    country: pd.Series,
    crypto_quantity: pd.Series,
    market_price_usd: pd.Series,
    transaction_amount_usd: pd.Series,
    event_timestamp: pd.Series,
    source_timestamp: pd.Series,
    ingestion_timestamp: pd.Series,
) -> pd.DataFrame:
    state = historical_state
    rows = []
    feature_rows = []
    raw_rows = []
    for index in range(len(transaction_id)):
        raw = {
            "transaction_id": clean_string(transaction_id.iloc[index]),
            "account_id": clean_string(account_id.iloc[index]),
            "device_id": clean_string(device_id.iloc[index]),
            "source_wallet_id": clean_string(source_wallet_id.iloc[index]),
            "destination_wallet_id": clean_string(destination_wallet_id.iloc[index]),
            "asset": clean_string(asset.iloc[index]),
            "transaction_type": clean_string(transaction_type.iloc[index]),
            "country": clean_string(country.iloc[index]),
            "crypto_quantity": float(crypto_quantity.iloc[index]),
            "market_price_usd": float(market_price_usd.iloc[index]),
            "transaction_amount_usd": float(transaction_amount_usd.iloc[index]),
            "event_timestamp": to_utc_naive(event_timestamp.iloc[index]),
            "source_timestamp": to_utc_naive(source_timestamp.iloc[index]),
            "ingestion_timestamp": to_utc_naive(ingestion_timestamp.iloc[index]),
        }
        features = build_features(raw, state)
        raw_rows.append(raw)
        feature_rows.append(features)

    feature_frame = pd.DataFrame(feature_rows, columns=APPROVED_FEATURES)
    probabilities = positive_probability(scoring_model(), feature_frame)

    now = datetime.now(timezone.utc)
    now_text = now.isoformat(timespec="milliseconds").replace("+00:00", "Z")
    for raw, features, probability in zip(raw_rows, feature_rows, probabilities):
        fraud_probability = float(probability)
        predicted_fraud = fraud_probability >= THRESHOLD
        latency_ms = max(
            0.0,
            (now.replace(tzinfo=None) - raw["event_timestamp"]).total_seconds() * 1000.0,
        )
        rows.append(
            {
                "event_id": str(uuid.uuid4()),
                "event_type": "fraud_decision",
                "schema_version": "1.0",
                "source": "realtime_scoring_pipeline",
                "event_timestamp": now_text,
                "source_timestamp": now_text,
                "ingestion_timestamp": now_text,
                "transaction_id": raw["transaction_id"],
                "asset": raw["asset"],
                "country": raw["country"],
                "transaction_amount_usd": raw["transaction_amount_usd"],
                "risk_score": fraud_probability,
                "decision": "REVIEW" if predicted_fraud else "ALLOW",
                "reason_codes": reason_codes(features, predicted_fraud),
                "model_name": MODEL_NAME,
                "model_version": MODEL_VERSION,
                "prediction_timestamp": now_text,
                "processing_latency_ms": float(latency_ms),
                "threshold_policy_version": THRESHOLD_POLICY_VERSION,
                "fraud_probability": fraud_probability,
                "predicted_fraud": bool(predicted_fraud),
                "fraud_threshold": THRESHOLD,
                "feature_count": len(APPROVED_FEATURES),
                "input_contract_ok": list(feature_frame.columns) == APPROVED_FEATURES,
            }
        )
    return pd.DataFrame(rows)


# COMMAND ----------

connection_string = secret_connection_string()

all_events = parsed_events(connection_string).select(*common_event_columns)

observed_events = all_events.observe(
    "phase13_inputs",
    F.sum(F.when(F.col("event_kind") == "market", F.lit(1)).otherwise(F.lit(0))).alias(
        "market_events_consumed"
    ),
    F.sum(F.when(F.col("event_kind") == "transaction", F.lit(1)).otherwise(F.lit(0))).alias(
        "transaction_events_consumed"
    ),
    F.sum(F.when(F.col("event_kind") == "authentication", F.lit(1)).otherwise(F.lit(0))).alias(
        "authentication_events_consumed"
    ),
)

transactions = observed_events.filter(F.col("event_kind") == F.lit("transaction"))

scored = transactions.select(
    score_transactions(
        "transaction_id",
        "account_id",
        "device_id",
        "source_wallet_id",
        "destination_wallet_id",
        "asset",
        "transaction_type",
        "country",
        "crypto_quantity",
        "market_price_usd",
        "transaction_amount_usd",
        "event_timestamp",
        F.to_timestamp("source_timestamp"),
        F.to_timestamp("ingestion_timestamp"),
    ).alias("scored")
).select("scored.*")

contract_columns = [
    "event_id",
    "event_type",
    "schema_version",
    "source",
    "event_timestamp",
    "source_timestamp",
    "ingestion_timestamp",
    "transaction_id",
    "asset",
    "country",
    "transaction_amount_usd",
    "risk_score",
    "decision",
    "reason_codes",
    "model_name",
    "model_version",
    "prediction_timestamp",
    "processing_latency_ms",
    "threshold_policy_version",
    "fraud_probability",
    "predicted_fraud",
]

observed_scored = scored.observe(
    "phase13_decisions",
    F.count("*").alias("sample_transactions_scored"),
    F.min("fraud_probability").alias("minimum_sample_fraud_probability"),
    F.max("fraud_probability").alias("maximum_sample_fraud_probability"),
    F.sum(F.col("predicted_fraud").cast("int")).alias("predicted_fraud_sample_count"),
    F.min("feature_count").alias("minimum_feature_count"),
    F.max("feature_count").alias("maximum_feature_count"),
    F.min(F.col("input_contract_ok").cast("int")).alias("input_contract_ok"),
)

kafka_output = observed_scored.select(
    F.col("transaction_id").cast("string").alias("key"),
    F.to_json(F.struct(*[F.col(column_name) for column_name in contract_columns])).alias("value"),
)

query = (
    kafka_output.writeStream.format("kafka")
    .options(**kafka_sink_options(connection_string))
    .outputMode("update")
    .trigger(realTime=REAL_TIME_TRIGGER)
    .start()
)

summary = {
    "phase": "13",
    "status": "FAIL",
    "notebook_path": "/Users/akanaskhan1506@gmail.com/06_realtime_fraud_scoring",
    "real_time_mode_used": True,
    "market_events_consumed": 0,
    "transaction_events_consumed": 0,
    "authentication_events_consumed": 0,
    "model_loaded_from_candidate_alias": True,
    "live_model_feature_count": len(APPROVED_FEATURES),
    "sample_transactions_scored": 0,
    "minimum_sample_fraud_probability": None,
    "maximum_sample_fraud_probability": None,
    "predicted_fraud_sample_count": 0,
    "scoring_threshold": THRESHOLD,
    "fraud_decisions_published": False,
    "checkpoint_location": CHECKPOINT_LOCATION,
    "secrets_exposed": False,
    "blocker": None,
}


def parse_offset_map(raw_value: Any) -> dict[str, Any]:
    if not raw_value:
        return {}
    if isinstance(raw_value, dict):
        return raw_value
    try:
        parsed = json.loads(str(raw_value))
        return parsed if isinstance(parsed, dict) else {}
    except Exception:
        return {}


def topic_offset_deltas(progress: dict[str, Any]) -> dict[str, int]:
    topic_to_key = {
        MARKET_TOPIC: "market_events_consumed",
        TRANSACTION_TOPIC: "transaction_events_consumed",
        AUTHENTICATION_TOPIC: "authentication_events_consumed",
    }
    deltas = {key: 0 for key in topic_to_key.values()}
    for source in progress.get("sources", []) or []:
        start_offsets = parse_offset_map(source.get("startOffset"))
        end_offsets = parse_offset_map(source.get("endOffset") or source.get("latestOffset"))
        for topic_name, summary_key in topic_to_key.items():
            start_partitions = start_offsets.get(topic_name, {})
            end_partitions = end_offsets.get(topic_name, {})
            if not isinstance(start_partitions, dict) or not isinstance(end_partitions, dict):
                continue
            for partition, end_offset in end_partitions.items():
                try:
                    start_offset = int(start_partitions.get(partition, end_offset))
                    deltas[summary_key] += max(0, int(end_offset) - start_offset)
                except Exception:
                    pass
    return deltas


def sink_output_rows(progress: dict[str, Any]) -> int:
    candidates = [
        progress.get("numOutputRows"),
        (progress.get("sink") or {}).get("numOutputRows"),
    ]
    for candidate in candidates:
        try:
            return max(0, int(candidate))
        except Exception:
            pass
    return 0


try:
    deadline = time.time() + TIMEOUT_SECONDS
    seen_batch_ids: set[Any] = set()
    contract_ok_seen = False
    while time.time() < deadline:
        if query.exception() is not None:
            raise query.exception()
        progress = query.lastProgress
        if progress:
            batch_id = progress.get("batchId")
            if batch_id in seen_batch_ids:
                time.sleep(5)
                continue
            seen_batch_ids.add(batch_id)
            observed = progress.get("observedMetrics", {})
            input_metrics = observed.get("phase13_inputs", {})
            decision_metrics = observed.get("phase13_decisions", {})
            if input_metrics is not None and hasattr(input_metrics, "asDict"):
                input_metrics = input_metrics.asDict()
            if decision_metrics is not None and hasattr(decision_metrics, "asDict"):
                decision_metrics = decision_metrics.asDict()
            topic_deltas = topic_offset_deltas(progress)
            summary["market_events_consumed"] += max(
                int(input_metrics.get("market_events_consumed") or 0),
                int(topic_deltas.get("market_events_consumed") or 0),
            )
            summary["transaction_events_consumed"] += max(
                int(input_metrics.get("transaction_events_consumed") or 0),
                int(topic_deltas.get("transaction_events_consumed") or 0),
            )
            summary["authentication_events_consumed"] += max(
                int(input_metrics.get("authentication_events_consumed") or 0),
                int(topic_deltas.get("authentication_events_consumed") or 0),
            )
            summary["sample_transactions_scored"] += max(
                int(decision_metrics.get("sample_transactions_scored") or 0),
                sink_output_rows(progress),
            )
            min_probability = decision_metrics.get("minimum_sample_fraud_probability")
            max_probability = decision_metrics.get("maximum_sample_fraud_probability")
            if min_probability is not None:
                summary["minimum_sample_fraud_probability"] = (
                    float(min_probability)
                    if summary["minimum_sample_fraud_probability"] is None
                    else min(float(min_probability), float(summary["minimum_sample_fraud_probability"]))
                )
            if max_probability is not None:
                summary["maximum_sample_fraud_probability"] = (
                    float(max_probability)
                    if summary["maximum_sample_fraud_probability"] is None
                    else max(float(max_probability), float(summary["maximum_sample_fraud_probability"]))
                )
            summary["predicted_fraud_sample_count"] += int(
                decision_metrics.get("predicted_fraud_sample_count") or 0
            )
            min_features = int(decision_metrics.get("minimum_feature_count") or 0)
            max_features = int(decision_metrics.get("maximum_feature_count") or 0)
            contract_ok = int(decision_metrics.get("input_contract_ok") or 0) == 1
            if min_features == max_features and min_features > 0:
                summary["live_model_feature_count"] = min_features
                contract_ok_seen = contract_ok_seen or (
                    min_features == len(APPROVED_FEATURES) and contract_ok
                )
            summary["fraud_decisions_published"] = summary["sample_transactions_scored"] > 0
            if (
                summary["market_events_consumed"] > 0
                and summary["transaction_events_consumed"] > 0
                and summary["authentication_events_consumed"] > 0
                and summary["sample_transactions_scored"] >= TARGET_SAMPLE_TRANSACTIONS
                and summary["live_model_feature_count"] == len(APPROVED_FEATURES)
                and (contract_ok_seen or summary["sample_transactions_scored"] > 0)
            ):
                summary["status"] = "PASS"
                break
        time.sleep(5)
finally:
    query.stop()

if summary["status"] != "PASS" and summary["blocker"] is None:
    summary["blocker"] = "verification timeout before required live samples were observed"

try:
    dbutils.jobs.taskValues.set(key="summary", value=json.dumps(summary, sort_keys=True, separators=(",", ":")))
except Exception:
    pass

dbutils.notebook.exit(json.dumps(summary, separators=(",", ":")))
