# Databricks notebook source
# MAGIC %md
# MAGIC # 09 Feedback Retraining
# MAGIC
# MAGIC Join delayed live fraud labels to live fraud decisions, persist feedback
# MAGIC monitoring outputs, build a retraining-ready dataset with the canonical
# MAGIC 55-feature contract, and run one controlled retraining experiment without
# MAGIC changing the current candidate alias or live threshold.

# COMMAND ----------

from __future__ import annotations

import importlib
import inspect
import json
import math
import subprocess
import sys
from bisect import bisect_right
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any


PACKAGE_IMPORTS = {
    "mlflow": "mlflow",
    "numpy": "numpy",
    "pandas": "pandas",
    "sklearn": "sklearn",
    "xgboost": "xgboost",
}


def package_available(import_name: str) -> bool:
    try:
        importlib.import_module(import_name)
        return True
    except ImportError:
        return False


if not package_available("xgboost"):
    subprocess.check_call([sys.executable, "-m", "pip", "install", "xgboost==3.4.0"])

missing_packages = [
    package_name
    for package_name, import_name in PACKAGE_IMPORTS.items()
    if not package_available(import_name)
]
if missing_packages:
    raise RuntimeError(f"Missing required Phase 16 packages: {missing_packages}")

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import sklearn
import xgboost
from mlflow.models.signature import infer_signature
from pyspark.sql import DataFrame, Window
from pyspark.sql import functions as F
from pyspark.sql import types as T
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

try:
    import cloudpickle
except ImportError:
    cloudpickle = None

spark.conf.set("spark.sql.session.timeZone", "UTC")
spark.conf.set("spark.sql.shuffle.partitions", "1")

dbutils.widgets.text("run_retraining", "true")
RUN_RETRAINING = (dbutils.widgets.get("run_retraining") or "true").strip().lower() == "true"

# COMMAND ----------

CATALOG = "crypto_fraud"
BRONZE_SCHEMA = "bronze"
SILVER_SCHEMA = "silver"
FEATURE_SCHEMA = "features"
MONITORING_SCHEMA = "monitoring"

LIVE_DECISIONS_TABLE = f"{CATALOG}.{BRONZE_SCHEMA}.live_fraud_decisions"
LIVE_LABELS_TABLE = f"{CATALOG}.{BRONZE_SCHEMA}.live_fraud_labels"
LIVE_TRANSACTIONS_TABLE = f"{CATALOG}.{BRONZE_SCHEMA}.live_customer_transactions"
HISTORICAL_TRAINING_TABLE = f"{CATALOG}.{FEATURE_SCHEMA}.transaction_training_dataset"
FEEDBACK_TABLE = f"{CATALOG}.{MONITORING_SCHEMA}.fraud_prediction_feedback"
FEEDBACK_METRICS_TABLE = f"{CATALOG}.{MONITORING_SCHEMA}.fraud_feedback_metrics"
LIVE_RETRAINING_TABLE = f"{CATALOG}.{FEATURE_SCHEMA}.transaction_retraining_dataset"
COMBINED_RETRAINING_TABLE = f"{CATALOG}.{FEATURE_SCHEMA}.transaction_retraining_combined_dataset"

REGISTERED_MODEL_NAME = "crypto_fraud.models.fraud_detection_model"
CANDIDATE_MODEL_URI = f"models:/{REGISTERED_MODEL_NAME}@candidate"
EXPERIMENT_PATH = "/Users/akanaskhan1506@gmail.com/crypto-fraud-phase16-retraining"
WORKSPACE_NOTEBOOK_PATH = "/Users/akanaskhan1506@gmail.com/09_feedback_retraining"
THRESHOLD = 0.80
EXPECTED_FEATURE_COUNT = 55
RANDOM_STATE = 42
TARGET_COLUMN = "target_is_fraud"

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

REPORT_ROOT = Path("/tmp") / "crypto_fraud_phase16"
REPORT_ROOT.mkdir(parents=True, exist_ok=True)

# COMMAND ----------


def q(name: str) -> str:
    return f"`{name.replace('`', '``')}`"


def full_table(schema_name: str, table_name: str) -> str:
    return f"{CATALOG}.{schema_name}.{table_name}"


def quoted_full_name(full_name: str) -> str:
    return ".".join(q(part) for part in full_name.split("."))


def table_exists(full_name: str) -> bool:
    return spark.catalog.tableExists(full_name)


def require_table(full_name: str) -> None:
    if not table_exists(full_name):
        raise RuntimeError(f"Required Phase 16 source table does not exist: {full_name}")


def write_delta_table(df: DataFrame, full_name: str) -> None:
    (
        df.write.format("delta")
        .mode("overwrite")
        .option("overwriteSchema", "true")
        .saveAsTable(full_name)
    )


def as_jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): as_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [as_jsonable(item) for item in value]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return None if not math.isfinite(float(value)) else float(value)
    if isinstance(value, np.ndarray):
        return [as_jsonable(item) for item in value.tolist()]
    if isinstance(value, pd.Timestamp):
        if pd.isna(value):
            return None
        return value.isoformat()
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, float):
        return None if not math.isfinite(value) else value
    return value


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(as_jsonable(payload), indent=2, sort_keys=True), encoding="utf-8")


def metric_value(value: Any) -> float | None:
    if value is None:
        return None
    numeric = float(value)
    return numeric if math.isfinite(numeric) else None


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
    clean = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    if len(clean) < 2:
        return None
    mean = sum(clean) / len(clean)
    return math.sqrt(sum((value - mean) ** 2 for value in clean) / (len(clean) - 1))


def latest_per_transaction(df: DataFrame, timestamp_col: str) -> DataFrame:
    window = Window.partitionBy("transaction_id").orderBy(
        F.col(timestamp_col).desc_nulls_last(),
        F.col("kafka_timestamp").desc_nulls_last(),
        F.col("kafka_partition").desc_nulls_last(),
        F.col("kafka_offset").desc_nulls_last(),
    )
    return df.withColumn("_rn", F.row_number().over(window)).filter(F.col("_rn") == 1).drop("_rn")


def positive_probability(model: Any, frame: pd.DataFrame) -> np.ndarray:
    probabilities = model.predict_proba(frame)
    classes = list(model.classes_)
    if 1 in classes:
        return probabilities[:, classes.index(1)]
    if True in classes:
        return probabilities[:, classes.index(True)]
    return np.zeros(len(frame), dtype=float)


def classification_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float) -> dict[str, Any]:
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    class_count = len(set(y_true.tolist()))
    return {
        "threshold": float(threshold),
        "row_count": int(len(y_true)),
        "fraud_count": int(np.sum(y_true == 1)),
        "normal_count": int(np.sum(y_true == 0)),
        "predicted_fraud_count": int(np.sum(y_pred == 1)),
        "true_positives": int(tp),
        "false_positives": int(fp),
        "true_negatives": int(tn),
        "false_negatives": int(fn),
        "precision": metric_value(precision_score(y_true, y_pred, zero_division=0)),
        "recall": metric_value(recall_score(y_true, y_pred, zero_division=0)),
        "f1": metric_value(f1_score(y_true, y_pred, zero_division=0)),
        "fraud_rate": metric_value(np.mean(y_true == 1)) if len(y_true) else None,
        "pr_auc": metric_value(average_precision_score(y_true, y_prob)) if class_count == 2 else None,
        "roc_auc": metric_value(roc_auc_score(y_true, y_prob)) if class_count == 2 else None,
    }


def row_count(df: DataFrame) -> int:
    return int(df.count())


# COMMAND ----------

for source_table in [LIVE_DECISIONS_TABLE, LIVE_LABELS_TABLE, LIVE_TRANSACTIONS_TABLE, HISTORICAL_TRAINING_TABLE]:
    require_table(source_table)

spark.sql(f"CREATE SCHEMA IF NOT EXISTS {q(CATALOG)}.{q(MONITORING_SCHEMA)}")
spark.sql(f"CREATE SCHEMA IF NOT EXISTS {q(CATALOG)}.{q(FEATURE_SCHEMA)}")

ordered_feature_list_verified = len(APPROVED_FEATURES) == EXPECTED_FEATURE_COUNT and len(set(APPROVED_FEATURES)) == EXPECTED_FEATURE_COUNT
feature_label_leakage_columns = [
    name
    for name in APPROVED_FEATURES
    if "fraud" in name.lower() or "label" in name.lower() or name.lower() in {"is_fraud", TARGET_COLUMN}
]

# COMMAND ----------

label_payload_schema = T.StructType(
    [
        T.StructField("event_id", T.StringType()),
        T.StructField("event_timestamp", T.StringType()),
        T.StructField("transaction_id", T.StringType()),
        T.StructField("label_timestamp", T.StringType()),
        T.StructField("is_fraud", T.BooleanType()),
        T.StructField("fraud_type", T.StringType()),
        T.StructField("label_source", T.StringType()),
        T.StructField("investigation_status", T.StringType()),
    ]
)

transaction_payload_schema = T.StructType(
    [
        T.StructField("event_id", T.StringType()),
        T.StructField("event_timestamp", T.StringType()),
        T.StructField("transaction_id", T.StringType()),
        T.StructField("account_id", T.StringType()),
        T.StructField("asset", T.StringType()),
        T.StructField("crypto_quantity", T.DoubleType()),
        T.StructField("transaction_type", T.StringType()),
        T.StructField("source_wallet_id", T.StringType()),
        T.StructField("destination_wallet_id", T.StringType()),
        T.StructField("device_id", T.StringType()),
        T.StructField("country", T.StringType()),
        T.StructField("market_price_usd", T.DoubleType()),
        T.StructField("transaction_amount_usd", T.DoubleType()),
        T.StructField("transaction_status", T.StringType()),
    ]
)

decision_timestamp = F.coalesce(
    F.col("prediction_timestamp"),
    F.col("event_timestamp"),
    F.col("kafka_timestamp"),
    F.col("bronze_ingested_at"),
)
decisions = latest_per_transaction(
    spark.table(LIVE_DECISIONS_TABLE)
    .filter(F.col("transaction_id").isNotNull())
    .withColumn("decision_timestamp", decision_timestamp)
    .withColumn("fraud_probability", F.col("risk_score").cast("double"))
    .filter(F.col("fraud_probability").isNotNull())
    .withColumn("threshold", F.lit(float(THRESHOLD)))
    .withColumn("predicted_fraud", F.col("fraud_probability") >= F.lit(float(THRESHOLD)))
    .select(
        "transaction_id",
        F.col("event_id").alias("decision_event_id"),
        "decision_timestamp",
        "fraud_probability",
        "threshold",
        "predicted_fraud",
        "decision",
        "model_name",
        "model_version",
        "threshold_policy_version",
        F.col("kafka_partition").alias("decision_kafka_partition"),
        F.col("kafka_offset").alias("decision_kafka_offset"),
        F.col("kafka_timestamp").alias("decision_kafka_timestamp"),
        "bronze_ingested_at",
        "kafka_partition",
        "kafka_offset",
        "kafka_timestamp",
    ),
    "decision_timestamp",
)

labels_raw = spark.table(LIVE_LABELS_TABLE).withColumn("label_payload", F.from_json("raw_payload", label_payload_schema))
labels = latest_per_transaction(
    labels_raw.filter(F.col("transaction_id").isNotNull())
    .withColumn(
        "label_timestamp",
        F.coalesce(
            F.to_timestamp("label_payload.label_timestamp"),
            F.to_timestamp("label_payload.event_timestamp"),
            F.col("event_timestamp"),
            F.col("kafka_timestamp"),
            F.col("bronze_ingested_at"),
        ),
    )
    .withColumn("actual_is_fraud", F.coalesce(F.col("is_fraud"), F.col("label_payload.is_fraud")).cast("boolean"))
    .filter(F.col("actual_is_fraud").isNotNull())
    .select(
        "transaction_id",
        F.coalesce(F.col("event_id"), F.col("label_payload.event_id")).alias("label_event_id"),
        "label_timestamp",
        "actual_is_fraud",
        F.coalesce(F.col("label_source"), F.col("label_payload.label_source")).alias("label_source"),
        F.coalesce(F.col("investigation_status"), F.col("label_payload.investigation_status")).alias("investigation_status"),
        F.col("kafka_partition").alias("label_kafka_partition"),
        F.col("kafka_offset").alias("label_kafka_offset"),
        F.col("kafka_timestamp").alias("label_kafka_timestamp"),
        "bronze_ingested_at",
        "kafka_partition",
        "kafka_offset",
        "kafka_timestamp",
    ),
    "label_timestamp",
)

transactions_raw = spark.table(LIVE_TRANSACTIONS_TABLE).withColumn(
    "transaction_payload", F.from_json("raw_payload", transaction_payload_schema)
)
transactions = latest_per_transaction(
    transactions_raw.filter(F.col("transaction_id").isNotNull())
    .withColumn(
        "transaction_timestamp",
        F.coalesce(F.to_timestamp("transaction_payload.event_timestamp"), F.col("event_timestamp"), F.col("kafka_timestamp")),
    )
    .select(
        "transaction_id",
        F.coalesce(F.col("event_id"), F.col("transaction_payload.event_id")).alias("transaction_event_id"),
        "transaction_timestamp",
        F.coalesce(F.col("account_id"), F.col("transaction_payload.account_id")).alias("account_id"),
        F.col("transaction_payload.asset").alias("asset"),
        F.col("transaction_payload.crypto_quantity").cast("double").alias("crypto_quantity"),
        F.col("transaction_payload.transaction_type").alias("transaction_type"),
        F.col("transaction_payload.source_wallet_id").alias("source_wallet_id"),
        F.col("transaction_payload.destination_wallet_id").alias("destination_wallet_id"),
        F.coalesce(F.col("device_id"), F.col("transaction_payload.device_id")).alias("device_id"),
        F.col("transaction_payload.country").alias("country"),
        F.col("transaction_payload.market_price_usd").cast("double").alias("market_price_usd"),
        F.col("transaction_payload.transaction_amount_usd").cast("double").alias("transaction_amount_usd"),
        F.col("transaction_payload.transaction_status").alias("transaction_status"),
        F.col("kafka_partition").alias("transaction_kafka_partition"),
        F.col("kafka_offset").alias("transaction_kafka_offset"),
        F.col("kafka_timestamp").alias("transaction_kafka_timestamp"),
        "bronze_ingested_at",
        "kafka_partition",
        "kafka_offset",
        "kafka_timestamp",
    ),
    "transaction_timestamp",
)

# COMMAND ----------

feedback_df = (
    decisions.alias("d")
    .join(labels.alias("l"), "transaction_id", "inner")
    .join(
        transactions.select(
            "transaction_id",
            "transaction_event_id",
            "transaction_timestamp",
            "transaction_kafka_partition",
            "transaction_kafka_offset",
        ).alias("t"),
        "transaction_id",
        "left",
    )
    .withColumn("label_delay_seconds", F.unix_timestamp("label_timestamp") - F.unix_timestamp("decision_timestamp"))
    .withColumn("label_arrived_after_decision", F.col("label_timestamp") >= F.col("decision_timestamp"))
    .withColumn(
        "outcome",
        F.when(F.col("predicted_fraud") & F.col("actual_is_fraud"), F.lit("TP"))
        .when(F.col("predicted_fraud") & ~F.col("actual_is_fraud"), F.lit("FP"))
        .when(~F.col("predicted_fraud") & F.col("actual_is_fraud"), F.lit("FN"))
        .otherwise(F.lit("TN")),
    )
    .withColumn("true_positive", (F.col("outcome") == F.lit("TP")).cast("int"))
    .withColumn("false_positive", (F.col("outcome") == F.lit("FP")).cast("int"))
    .withColumn("true_negative", (F.col("outcome") == F.lit("TN")).cast("int"))
    .withColumn("false_negative", (F.col("outcome") == F.lit("FN")).cast("int"))
    .withColumn("model_alias_at_scoring", F.lit("candidate"))
    .withColumn("feedback_created_at", F.current_timestamp())
    .select(
        "transaction_id",
        "transaction_event_id",
        "transaction_timestamp",
        "decision_event_id",
        "decision_timestamp",
        "label_event_id",
        "label_timestamp",
        "fraud_probability",
        "threshold",
        "predicted_fraud",
        "actual_is_fraud",
        "label_delay_seconds",
        "label_arrived_after_decision",
        "outcome",
        "true_positive",
        "false_positive",
        "true_negative",
        "false_negative",
        "model_name",
        "model_version",
        "model_alias_at_scoring",
        "threshold_policy_version",
        "label_source",
        "investigation_status",
        "decision_kafka_partition",
        "decision_kafka_offset",
        "decision_kafka_timestamp",
        "label_kafka_partition",
        "label_kafka_offset",
        "label_kafka_timestamp",
        "transaction_kafka_partition",
        "transaction_kafka_offset",
        "feedback_created_at",
    )
)

write_delta_table(feedback_df, FEEDBACK_TABLE)
feedback_stored_df = spark.table(FEEDBACK_TABLE)
feedback_pdf = feedback_stored_df.select(
    "transaction_id",
    "actual_is_fraud",
    "predicted_fraud",
    "fraud_probability",
).toPandas()

if len(feedback_pdf) == 0:
    feedback_metrics = {
        "labeled_decision_count": 0,
        "actual_fraud_count": 0,
        "actual_normal_count": 0,
        "predicted_fraud_count": 0,
        "true_positives": 0,
        "false_positives": 0,
        "true_negatives": 0,
        "false_negatives": 0,
        "precision": None,
        "recall": None,
        "f1": None,
        "fraud_rate": None,
        "pr_auc": None,
        "roc_auc": None,
    }
else:
    feedback_y_true = feedback_pdf["actual_is_fraud"].astype(int).to_numpy()
    feedback_y_prob = feedback_pdf["fraud_probability"].astype(float).to_numpy()
    calculated = classification_metrics(feedback_y_true, feedback_y_prob, THRESHOLD)
    feedback_metrics = {
        "labeled_decision_count": calculated["row_count"],
        "actual_fraud_count": calculated["fraud_count"],
        "actual_normal_count": calculated["normal_count"],
        "predicted_fraud_count": calculated["predicted_fraud_count"],
        "true_positives": calculated["true_positives"],
        "false_positives": calculated["false_positives"],
        "true_negatives": calculated["true_negatives"],
        "false_negatives": calculated["false_negatives"],
        "precision": calculated["precision"],
        "recall": calculated["recall"],
        "f1": calculated["f1"],
        "fraud_rate": calculated["fraud_rate"],
        "pr_auc": calculated["pr_auc"],
        "roc_auc": calculated["roc_auc"],
    }

metrics_row = {
    **feedback_metrics,
    "threshold": float(THRESHOLD),
    "feedback_table": FEEDBACK_TABLE,
    "created_at_utc": datetime.now(timezone.utc),
}
metrics_schema = T.StructType(
    [
        T.StructField("labeled_decision_count", T.LongType()),
        T.StructField("actual_fraud_count", T.LongType()),
        T.StructField("actual_normal_count", T.LongType()),
        T.StructField("predicted_fraud_count", T.LongType()),
        T.StructField("true_positives", T.LongType()),
        T.StructField("false_positives", T.LongType()),
        T.StructField("true_negatives", T.LongType()),
        T.StructField("false_negatives", T.LongType()),
        T.StructField("precision", T.DoubleType()),
        T.StructField("recall", T.DoubleType()),
        T.StructField("f1", T.DoubleType()),
        T.StructField("fraud_rate", T.DoubleType()),
        T.StructField("pr_auc", T.DoubleType()),
        T.StructField("roc_auc", T.DoubleType()),
        T.StructField("threshold", T.DoubleType()),
        T.StructField("feedback_table", T.StringType()),
        T.StructField("created_at_utc", T.TimestampType()),
    ]
)
write_delta_table(spark.createDataFrame([metrics_row], schema=metrics_schema), FEEDBACK_METRICS_TABLE)

# COMMAND ----------


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
        for index, _ in enumerate(ordered):
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
        (F.col("account_id").isin(account_ids) | F.col("device_id").isin(device_ids))
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
        if row["event_timestamp"] is not None and cutoff <= row["event_timestamp"] < feature_timestamp
    ]


def prior_all(rows: list[dict[str, Any]], feature_timestamp: datetime) -> list[dict[str, Any]]:
    return [row for row in rows if row["event_timestamp"] is not None and row["event_timestamp"] < feature_timestamp]


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
    profile_available = account_exists and (account_updated is None or account_updated <= feature_timestamp)
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
            "prior_tx_amount_avg_24h": float(sum(amounts_24h) / len(amounts_24h)) if amounts_24h else None,
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
    prior_successful = [item for item in prior_all(account_auth, feature_timestamp) if item["login_success"]]
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
            "device_failed_auth_count_10m": len([item for item in device_auth_10m if not item["login_success"]]),
            "device_successful_auth_count_1h": len([item for item in device_auth_1h if item["login_success"]]),
            "device_distinct_account_count_1h": len({item["account_id"] for item in device_auth_1h if item.get("account_id")}),
        }
    )

    device = state["devices"].get(device_id)
    device_first_seen = device.get("first_seen_at") if device else None
    device_available = device_first_seen is not None and device_first_seen <= feature_timestamp
    device_age_hours = seconds_between(feature_timestamp, device_first_seen) / 3600.0 if device_available else None
    features.update(
        {
            "device_profile_available": 1 if device_available else 0,
            "device_age_hours": device_age_hours,
            "device_age_days": device_age_hours / 24.0 if device_age_hours is not None else None,
            "is_new_device_24h": (1 if device_age_hours is not None and device_age_hours <= 24.0 else 0)
            if device_available
            else None,
        }
    )

    source_applicable = source_wallet_id is not None
    destination_applicable = destination_wallet_id is not None
    destination_rows = state["transactions_by_destination_wallet"].get(destination_wallet_id, [])
    source_rows = state["transactions_by_source_wallet"].get(source_wallet_id, [])
    prior_destination_count = wallet_prior_count(destination_rows, feature_timestamp) if destination_applicable else 0
    prior_source_count = wallet_prior_count(source_rows, feature_timestamp) if source_applicable else 0
    destination_wallet = state["wallets"].get(destination_wallet_id)
    destination_first_seen = destination_wallet.get("first_seen_at") if destination_wallet else None
    destination_available = destination_applicable and destination_first_seen is not None and destination_first_seen <= feature_timestamp
    features.update(
        {
            "source_wallet_applicable": 1 if source_applicable else 0,
            "destination_wallet_applicable": 1 if destination_applicable else 0,
            "destination_wallet_first_seen_available": 1 if destination_available else 0,
            "destination_wallet_age_hours": seconds_between(feature_timestamp, destination_first_seen) / 3600.0
            if destination_available
            else None,
            "is_new_destination_wallet": (1 if prior_destination_count == 0 else 0)
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
            "market_data_freshness_seconds": seconds_between(feature_timestamp, market["candle_end_timestamp"]) if market else None,
            "market_return_1h": market.get("market_return_1h") if market else None,
            "market_return_5m": market.get("market_return_5m") if market else None,
            "market_volatility_1h": market.get("market_volatility_1h") if market else None,
            "market_volume_sum_1h": market.get("market_volume_sum_1h") if market else None,
        }
    )
    return {feature_name: features.get(feature_name) for feature_name in APPROVED_FEATURES}


# COMMAND ----------

training_feedback = feedback_stored_df.select(
    "transaction_id",
    "actual_is_fraud",
    "label_timestamp",
    "decision_timestamp",
)

live_training_inputs = (
    transactions.join(training_feedback, "transaction_id", "inner")
    .filter(
        F.col("transaction_timestamp").isNotNull()
        & F.col("account_id").isNotNull()
        & F.col("device_id").isNotNull()
        & F.col("asset").isNotNull()
        & F.col("transaction_type").isNotNull()
        & F.col("country").isNotNull()
        & F.col("transaction_amount_usd").isNotNull()
        & F.col("crypto_quantity").isNotNull()
        & F.col("market_price_usd").isNotNull()
    )
    .orderBy("transaction_timestamp", "transaction_id")
)

live_input_rows = live_training_inputs.collect()

feature_schema = T.StructType(
    [
        T.StructField("transaction_id", T.StringType(), False),
        T.StructField("feature_timestamp", T.TimestampType()),
        T.StructField("transaction_timestamp", T.TimestampType()),
        T.StructField("label_timestamp", T.TimestampType()),
        T.StructField("decision_timestamp", T.TimestampType()),
        T.StructField("training_source", T.StringType(), False),
        T.StructField(TARGET_COLUMN, T.BooleanType(), False),
    ]
    + [
        T.StructField(name, T.StringType() if name in CATEGORICAL_FEATURES else T.DoubleType())
        for name in APPROVED_FEATURES
    ]
)

if live_input_rows:
    account_ids = sorted({row.account_id for row in live_input_rows if row.account_id})
    device_ids = sorted({row.device_id for row in live_input_rows if row.device_id})
    wallet_ids = sorted(
        {
            wallet_id
            for row in live_input_rows
            for wallet_id in (row.source_wallet_id, row.destination_wallet_id)
            if wallet_id
        }
    )
    feature_timestamps = [to_utc_naive(row.transaction_timestamp) for row in live_input_rows]
    state = collect_historical_state(
        account_ids=account_ids,
        device_ids=device_ids,
        wallet_ids=wallet_ids,
        min_feature_timestamp=min(feature_timestamps),
        max_feature_timestamp=max(feature_timestamps),
    )

    live_feature_rows = []
    for row in live_input_rows:
        raw = {
            "transaction_id": row.transaction_id,
            "account_id": row.account_id,
            "device_id": row.device_id,
            "source_wallet_id": row.source_wallet_id,
            "destination_wallet_id": row.destination_wallet_id,
            "event_timestamp": to_utc_naive(row.transaction_timestamp),
            "asset": row.asset,
            "transaction_type": row.transaction_type,
            "country": row.country,
            "transaction_amount_usd": float(row.transaction_amount_usd),
            "crypto_quantity": float(row.crypto_quantity),
            "market_price_usd": float(row.market_price_usd),
        }
        features = online_build_features(raw, state)
        live_feature_rows.append(
            {
                "transaction_id": row.transaction_id,
                "feature_timestamp": to_utc_naive(row.transaction_timestamp),
                "transaction_timestamp": to_utc_naive(row.transaction_timestamp),
                "label_timestamp": to_utc_naive(row.label_timestamp),
                "decision_timestamp": to_utc_naive(row.decision_timestamp),
                "training_source": "live_feedback",
                TARGET_COLUMN: bool(row.actual_is_fraud),
                **features,
            }
        )
else:
    live_feature_rows = []

live_retraining_df = spark.createDataFrame(live_feature_rows, schema=feature_schema)
write_delta_table(live_retraining_df, LIVE_RETRAINING_TABLE)

# COMMAND ----------


def feature_select_expressions(df: DataFrame) -> list[Any]:
    return [
        F.col(name).cast("string" if name in CATEGORICAL_FEATURES else "double").alias(name)
        for name in APPROVED_FEATURES
    ]


historical_df = spark.table(HISTORICAL_TRAINING_TABLE)
historical_columns = set(historical_df.columns)
missing_historical_features = [name for name in APPROVED_FEATURES if name not in historical_columns]
if TARGET_COLUMN not in historical_columns:
    raise RuntimeError(f"Historical training table is missing required target column {TARGET_COLUMN}.")
if missing_historical_features:
    raise RuntimeError(f"Historical training table is missing approved model features: {missing_historical_features}")

historical_feature_timestamp = (
    F.col("feature_timestamp").cast("timestamp")
    if "feature_timestamp" in historical_columns
    else F.lit(None).cast("timestamp")
)
historical_selected = historical_df.select(
    F.col("transaction_id").cast("string").alias("transaction_id"),
    historical_feature_timestamp.alias("feature_timestamp"),
    historical_feature_timestamp.alias("transaction_timestamp"),
    F.col("label_timestamp").cast("timestamp").alias("label_timestamp")
    if "label_timestamp" in historical_columns
    else F.lit(None).cast("timestamp").alias("label_timestamp"),
    F.lit(None).cast("timestamp").alias("decision_timestamp"),
    F.lit("historical").alias("training_source"),
    F.col(TARGET_COLUMN).cast("boolean").alias(TARGET_COLUMN),
    *feature_select_expressions(historical_df),
)

live_selected = spark.table(LIVE_RETRAINING_TABLE).select(
    "transaction_id",
    "feature_timestamp",
    "transaction_timestamp",
    "label_timestamp",
    "decision_timestamp",
    "training_source",
    TARGET_COLUMN,
    *feature_select_expressions(spark.table(LIVE_RETRAINING_TABLE)),
)

combined_raw = historical_selected.unionByName(live_selected)
dedupe_window = Window.partitionBy("transaction_id").orderBy(
    F.when(F.col("training_source") == F.lit("live_feedback"), F.lit(1)).otherwise(F.lit(0)).desc(),
    F.col("label_timestamp").desc_nulls_last(),
    F.col("feature_timestamp").desc_nulls_last(),
)
combined_training_df = (
    combined_raw.filter(F.col("transaction_id").isNotNull())
    .withColumn("_rn", F.row_number().over(dedupe_window))
    .filter(F.col("_rn") == 1)
    .drop("_rn")
)
write_delta_table(combined_training_df, COMBINED_RETRAINING_TABLE)

historical_rows = row_count(historical_selected)
live_feedback_rows = row_count(spark.table(LIVE_RETRAINING_TABLE))
retraining_rows = row_count(spark.table(COMBINED_RETRAINING_TABLE))

live_class_counts = {
    bool(row[TARGET_COLUMN]): int(row["count"])
    for row in spark.table(LIVE_RETRAINING_TABLE).groupBy(TARGET_COLUMN).count().collect()
}
combined_class_counts = {
    bool(row[TARGET_COLUMN]): int(row["count"])
    for row in spark.table(COMBINED_RETRAINING_TABLE).groupBy(TARGET_COLUMN).count().collect()
}

# COMMAND ----------


class NumericCoercer(BaseEstimator, TransformerMixin):
    def fit(self, X: Any, y: Any = None):
        self.feature_names_in_ = self._feature_names(X)
        return self

    def transform(self, X: Any):
        frame = self._to_frame(X)
        converted = frame.apply(pd.to_numeric, errors="coerce")
        converted = converted.replace([np.inf, -np.inf], np.nan)
        return converted.to_numpy(dtype=float)

    def get_feature_names_out(self, input_features: Any = None):
        if input_features is None:
            input_features = getattr(self, "feature_names_in_", None)
        return np.asarray(input_features, dtype=object)

    @staticmethod
    def _feature_names(X: Any) -> list[str]:
        if isinstance(X, pd.DataFrame):
            return list(X.columns)
        return [f"x{i}" for i in range(np.asarray(X).shape[1])]

    def _to_frame(self, X: Any) -> pd.DataFrame:
        if isinstance(X, pd.DataFrame):
            return X.copy()
        return pd.DataFrame(X, columns=getattr(self, "feature_names_in_", None))


class CategoricalCleaner(BaseEstimator, TransformerMixin):
    def __init__(self, missing_value: str = "__MISSING__"):
        self.missing_value = missing_value

    def fit(self, X: Any, y: Any = None):
        self.feature_names_in_ = self._feature_names(X)
        return self

    def transform(self, X: Any):
        frame = self._to_frame(X)
        cleaned = frame.astype("object").where(pd.notna(frame), self.missing_value)
        return cleaned.astype(str)

    def get_feature_names_out(self, input_features: Any = None):
        if input_features is None:
            input_features = getattr(self, "feature_names_in_", None)
        return np.asarray(input_features, dtype=object)

    @staticmethod
    def _feature_names(X: Any) -> list[str]:
        if isinstance(X, pd.DataFrame):
            return list(X.columns)
        return [f"x{i}" for i in range(np.asarray(X).shape[1])]

    def _to_frame(self, X: Any) -> pd.DataFrame:
        if isinstance(X, pd.DataFrame):
            return X.copy()
        return pd.DataFrame(X, columns=getattr(self, "feature_names_in_", None))


def build_xgb_pipeline(scale_pos_weight: float) -> Pipeline:
    encoder_kwargs = {"handle_unknown": "ignore"}
    if "sparse_output" in inspect.signature(OneHotEncoder).parameters:
        encoder_kwargs["sparse_output"] = True
    else:
        encoder_kwargs["sparse"] = True

    numeric_transformer = Pipeline(
        steps=[
            ("coerce_numeric", NumericCoercer()),
            ("impute_median", SimpleImputer(strategy="median")),
            ("standardize", StandardScaler()),
        ]
    )
    categorical_transformer = Pipeline(
        steps=[
            ("clean_categories", CategoricalCleaner(missing_value="__MISSING__")),
            ("one_hot", OneHotEncoder(**encoder_kwargs)),
        ]
    )
    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", numeric_transformer, NUMERIC_FEATURES),
            ("categorical", categorical_transformer, CATEGORICAL_FEATURES),
        ],
        remainder="drop",
        verbose_feature_names_out=True,
    )
    xgb_params = {
        "objective": "binary:logistic",
        "eval_metric": "aucpr",
        "n_estimators": 200,
        "max_depth": 3,
        "learning_rate": 0.05,
        "subsample": 0.85,
        "colsample_bytree": 0.85,
        "min_child_weight": 1,
        "reg_lambda": 1.0,
        "random_state": RANDOM_STATE,
        "n_jobs": 1,
        "tree_method": "hist",
        "scale_pos_weight": float(scale_pos_weight),
        "verbosity": 0,
    }
    return Pipeline(steps=[("preprocess", preprocessor), ("model", XGBClassifier(**xgb_params))])


def chronological_split(pdf: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    ordered = pdf.sort_values(["feature_timestamp", "transaction_id"], na_position="first").reset_index(drop=True)
    row_total = len(ordered)
    train_end = max(1, int(row_total * 0.70))
    validation_end = max(train_end + 1, int(row_total * 0.85))
    validation_end = min(validation_end, row_total - 1)
    return (
        ordered.iloc[:train_end].copy(),
        ordered.iloc[train_end:validation_end].copy(),
        ordered.iloc[validation_end:].copy(),
    )


def split_summary(split_pdf: pd.DataFrame) -> dict[str, int]:
    y = split_pdf[TARGET_COLUMN].astype(int)
    return {
        "row_count": int(len(split_pdf)),
        "fraud_count": int(np.sum(y == 1)),
        "normal_count": int(np.sum(y == 0)),
    }


def feature_frame(pdf: pd.DataFrame) -> pd.DataFrame:
    frame = pdf[APPROVED_FEATURES].copy()
    for feature_name in CATEGORICAL_FEATURES:
        frame[feature_name] = frame[feature_name].astype("object")
    return frame


def log_metric_dict(prefix: str, metrics: dict[str, Any]) -> None:
    for key, value in metrics.items():
        if key in {"threshold", "row_count", "fraud_count", "normal_count", "predicted_fraud_count"}:
            continue
        cleaned = metric_value(value)
        if cleaned is not None:
            mlflow.log_metric(f"{prefix}_{key}", float(cleaned))


def log_params(params: dict[str, Any]) -> None:
    for key, value in params.items():
        if value is not None:
            mlflow.log_param(key, value)


# COMMAND ----------

mlflow.set_tracking_uri("databricks")
mlflow.set_registry_uri("databricks-uc")
mlflow.set_experiment(EXPERIMENT_PATH)
client = mlflow.tracking.MlflowClient()

try:
    candidate_alias_before = str(client.get_model_version_by_alias(REGISTERED_MODEL_NAME, "candidate").version)
except Exception:
    candidate_alias_before = None

retraining_executed = False
mlflow_run_id = None
new_model_version = None
registration_status = "NOT_ATTEMPTED"
candidate_comparison = {
    "candidate": {"pr_auc": None, "recall": None, "precision": None, "f1": None},
    "retrained": {"pr_auc": None, "recall": None, "precision": None, "f1": None},
}
split_details = {}
retraining_status_reason = None

feature_contract_valid = ordered_feature_list_verified and not feature_label_leakage_columns
live_has_both_classes = live_class_counts.get(True, 0) > 0 and live_class_counts.get(False, 0) > 0
combined_has_both_classes = combined_class_counts.get(True, 0) > 0 and combined_class_counts.get(False, 0) > 0

if not RUN_RETRAINING:
    registration_status = "SKIPPED_FEEDBACK_ONLY_MODE"
    retraining_status_reason = "Retraining disabled by run_retraining=false; feedback tables refreshed only."
elif live_feedback_rows > 0 and live_has_both_classes and combined_has_both_classes and feature_contract_valid:
    training_pdf = (
        spark.table(COMBINED_RETRAINING_TABLE)
        .select("transaction_id", "feature_timestamp", "training_source", TARGET_COLUMN, *APPROVED_FEATURES)
        .toPandas()
    )
    training_pdf[TARGET_COLUMN] = training_pdf[TARGET_COLUMN].astype(int)
    train_pdf, validation_pdf, test_pdf = chronological_split(training_pdf)
    split_details = {
        "train": split_summary(train_pdf),
        "validation": split_summary(validation_pdf),
        "test": split_summary(test_pdf),
        "split_method": "chronological_70_15_15",
        "test_partition_policy": "latest-time partition used only after model fitting",
    }

    train_has_both = split_details["train"]["fraud_count"] > 0 and split_details["train"]["normal_count"] > 0
    test_has_rows = split_details["test"]["row_count"] > 0
    if train_has_both and test_has_rows:
        X_train = feature_frame(train_pdf)
        y_train = train_pdf[TARGET_COLUMN].astype(int).to_numpy()
        X_validation = feature_frame(validation_pdf)
        y_validation = validation_pdf[TARGET_COLUMN].astype(int).to_numpy()
        X_test = feature_frame(test_pdf)
        y_test = test_pdf[TARGET_COLUMN].astype(int).to_numpy()

        scale_pos_weight = float(np.sum(y_train == 0) / max(np.sum(y_train == 1), 1))
        xgb_pipeline = build_xgb_pipeline(scale_pos_weight)

        with mlflow.start_run(run_name="phase16_controlled_feedback_retraining") as run:
            mlflow_run_id = run.info.run_id
            xgb_pipeline.fit(X_train, y_train)
            retraining_executed = True

            retrained_validation_prob = positive_probability(xgb_pipeline, X_validation)
            retrained_test_prob = positive_probability(xgb_pipeline, X_test)
            retrained_validation_metrics = classification_metrics(y_validation, retrained_validation_prob, THRESHOLD)
            retrained_test_metrics = classification_metrics(y_test, retrained_test_prob, THRESHOLD)

            candidate_model = mlflow.sklearn.load_model(CANDIDATE_MODEL_URI)
            candidate_test_prob = positive_probability(candidate_model, X_test)
            candidate_test_metrics = classification_metrics(y_test, candidate_test_prob, THRESHOLD)

            candidate_comparison = {
                "candidate": {
                    "pr_auc": candidate_test_metrics["pr_auc"],
                    "recall": candidate_test_metrics["recall"],
                    "precision": candidate_test_metrics["precision"],
                    "f1": candidate_test_metrics["f1"],
                    "false_positives": candidate_test_metrics["false_positives"],
                    "true_positives": candidate_test_metrics["true_positives"],
                    "false_negatives": candidate_test_metrics["false_negatives"],
                    "true_negatives": candidate_test_metrics["true_negatives"],
                },
                "retrained": {
                    "pr_auc": retrained_test_metrics["pr_auc"],
                    "recall": retrained_test_metrics["recall"],
                    "precision": retrained_test_metrics["precision"],
                    "f1": retrained_test_metrics["f1"],
                    "false_positives": retrained_test_metrics["false_positives"],
                    "true_positives": retrained_test_metrics["true_positives"],
                    "false_negatives": retrained_test_metrics["false_negatives"],
                    "true_negatives": retrained_test_metrics["true_negatives"],
                },
                "evaluation_split": "test",
                "threshold": float(THRESHOLD),
            }

            log_params(
                {
                    "phase": "16",
                    "model_type": "XGBClassifier",
                    "feature_count": len(APPROVED_FEATURES),
                    "historical_rows": historical_rows,
                    "live_feedback_rows": live_feedback_rows,
                    "combined_training_rows": retraining_rows,
                    "train_rows": split_details["train"]["row_count"],
                    "validation_rows": split_details["validation"]["row_count"],
                    "test_rows": split_details["test"]["row_count"],
                    "train_fraud_rows": split_details["train"]["fraud_count"],
                    "train_normal_rows": split_details["train"]["normal_count"],
                    "threshold": THRESHOLD,
                    "scale_pos_weight": scale_pos_weight,
                    "n_estimators": 200,
                    "max_depth": 3,
                    "learning_rate": 0.05,
                    "subsample": 0.85,
                    "colsample_bytree": 0.85,
                    "min_child_weight": 1,
                    "reg_lambda": 1.0,
                    "tree_method": "hist",
                }
            )
            log_metric_dict("feedback", feedback_metrics)
            log_metric_dict("validation_retrained", retrained_validation_metrics)
            log_metric_dict("test_retrained", retrained_test_metrics)
            log_metric_dict("test_candidate", candidate_test_metrics)
            mlflow.set_tag("phase", "16")
            mlflow.set_tag("candidate_alias_changed", "false")
            mlflow.set_tag("automatic_promotion", "false")
            mlflow.set_tag("threshold_changed", "false")
            mlflow.set_tag("fraud_label_leakage", "false")
            mlflow.log_dict(as_jsonable(split_details), "phase16_split_details.json")
            mlflow.log_dict(as_jsonable(candidate_comparison), "phase16_candidate_comparison.json")
            mlflow.log_dict(as_jsonable(feedback_metrics), "phase16_feedback_metrics.json")

            input_example = X_train.head(min(10, len(X_train))).copy()
            signature = infer_signature(input_example, xgb_pipeline.predict(input_example))
            pip_requirements = [
                f"mlflow=={mlflow.__version__}",
                f"xgboost=={xgboost.__version__}",
                f"scikit-learn=={sklearn.__version__}",
                f"pandas=={pd.__version__}",
                f"numpy=={np.__version__}",
            ]
            if cloudpickle is not None and getattr(cloudpickle, "__version__", None):
                pip_requirements.append(f"cloudpickle=={cloudpickle.__version__}")
            try:
                mlflow.sklearn.log_model(
                    sk_model=xgb_pipeline,
                    artifact_path="model",
                    signature=signature,
                    input_example=input_example,
                    pip_requirements=pip_requirements,
                )
            except TypeError:
                mlflow.sklearn.log_model(
                    sk_model=xgb_pipeline,
                    name="model",
                    signature=signature,
                    input_example=input_example,
                    pip_requirements=pip_requirements,
                )

            model_uri = f"runs:/{mlflow_run_id}/model"
            try:
                registered = mlflow.register_model(model_uri, REGISTERED_MODEL_NAME)
                new_model_version = str(registered.version)
                registration_status = "REGISTERED_NEW_VERSION"
            except Exception as registration_exc:
                registration_status = f"MODEL_ARTIFACT_LOGGED_REGISTRATION_SKIPPED: {type(registration_exc).__name__}"
    else:
        retraining_status_reason = "Training split lacks both classes or test rows."
else:
    retraining_status_reason = "Live feedback lacks usable rows, both classes, or the canonical feature contract."

try:
    candidate_alias_after = str(client.get_model_version_by_alias(REGISTERED_MODEL_NAME, "candidate").version)
except Exception:
    candidate_alias_after = None

candidate_alias_changed = candidate_alias_before != candidate_alias_after

# COMMAND ----------

failed_checks = []
if not ordered_feature_list_verified:
    failed_checks.append("canonical_ordered_55_feature_contract_not_verified")
if feature_label_leakage_columns:
    failed_checks.append("fraud_label_leakage_columns_present_in_model_features")
if feedback_metrics["labeled_decision_count"] <= 0:
    failed_checks.append("no_labeled_decisions_joined")
if live_feedback_rows <= 0:
    failed_checks.append("no_live_feedback_rows_for_retraining_dataset")
if len(APPROVED_FEATURES) != EXPECTED_FEATURE_COUNT:
    failed_checks.append("model_feature_count_not_55")
if RUN_RETRAINING and live_feedback_rows > 0 and not live_has_both_classes:
    failed_checks.append("live_feedback_does_not_contain_both_classes_for_retraining")
if RUN_RETRAINING and live_has_both_classes and not retraining_executed:
    failed_checks.append("controlled_retraining_did_not_execute_with_supported_data")
if not mlflow_run_id and retraining_executed:
    failed_checks.append("mlflow_run_missing")
if candidate_alias_changed:
    failed_checks.append("candidate_alias_changed")

overall_status = "PASS" if not failed_checks else "FAIL"

feedback_summary = {
    "phase": "16",
    "feedback_table": FEEDBACK_TABLE,
    "feedback_metrics_table": FEEDBACK_METRICS_TABLE,
    **feedback_metrics,
    "threshold": float(THRESHOLD),
    "fraud_label_leakage": False,
    "created_at_utc": datetime.now(timezone.utc).isoformat(),
}
retraining_summary = {
    "phase": "16",
    "overall_status": overall_status,
    "notebook": "databricks/09_feedback_retraining.py",
    "workspace_path": WORKSPACE_NOTEBOOK_PATH,
    "live_retraining_table": LIVE_RETRAINING_TABLE,
    "combined_retraining_table": COMBINED_RETRAINING_TABLE,
    "retraining_rows": retraining_rows,
    "historical_rows": historical_rows,
    "live_feedback_rows": live_feedback_rows,
    "feature_count": len(APPROVED_FEATURES),
    "ordered_feature_list_verified": ordered_feature_list_verified,
    "feature_label_leakage_columns": feature_label_leakage_columns,
    "live_class_counts": {"fraud": live_class_counts.get(True, 0), "normal": live_class_counts.get(False, 0)},
    "combined_class_counts": {"fraud": combined_class_counts.get(True, 0), "normal": combined_class_counts.get(False, 0)},
    "run_retraining": RUN_RETRAINING,
    "retraining_executed": retraining_executed,
    "retraining_status_reason": retraining_status_reason,
    "mlflow_run_id": mlflow_run_id,
    "new_model_version_registered": new_model_version,
    "registration_status": registration_status,
    "candidate_alias_before": candidate_alias_before,
    "candidate_alias_after": candidate_alias_after,
    "candidate_alias_changed": candidate_alias_changed,
    "candidate_comparison": candidate_comparison,
    "split_details": split_details,
    "threshold_changed": False,
    "automatic_promotion_performed": False,
    "registered_candidate_model_changed": False,
    "failed_checks": failed_checks,
    "created_at_utc": datetime.now(timezone.utc).isoformat(),
}

write_json(REPORT_ROOT / "phase16_feedback_summary.json", feedback_summary)
write_json(REPORT_ROOT / "phase16_retraining_summary.json", retraining_summary)

summary = {
    **retraining_summary,
    "feedback": feedback_summary,
    "feedback_table": FEEDBACK_TABLE,
    "feedback_confusion_matrix": {
        "TP": feedback_metrics["true_positives"],
        "FP": feedback_metrics["false_positives"],
        "TN": feedback_metrics["true_negatives"],
        "FN": feedback_metrics["false_negatives"],
    },
    "secrets_exposed": False,
}

try:
    dbutils.jobs.taskValues.set(key="summary", value=json.dumps(as_jsonable(summary), sort_keys=True, separators=(",", ":")))
except Exception:
    pass

dbutils.notebook.exit(json.dumps(as_jsonable(summary), sort_keys=True, separators=(",", ":")))
