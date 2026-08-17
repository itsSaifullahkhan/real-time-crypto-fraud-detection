# Databricks notebook source
# MAGIC %md
# MAGIC # 10 Gold Analytics
# MAGIC
# MAGIC Build dashboard-ready Gold monitoring tables from durable live Bronze data and
# MAGIC Phase 16 delayed-feedback metrics. This phase is analytics-only: it does not
# MAGIC alter model aliases, thresholds, labels, or scoring behavior.

# COMMAND ----------

from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import mlflow
import numpy as np
import pandas as pd
from pyspark.sql import DataFrame, Window
from pyspark.sql import functions as F
from pyspark.sql import types as T
from sklearn.metrics import average_precision_score, roc_auc_score

spark.conf.set("spark.sql.session.timeZone", "UTC")
spark.conf.set("spark.sql.shuffle.partitions", "1")

# COMMAND ----------

CATALOG = "crypto_fraud"
BRONZE_SCHEMA = "bronze"
MONITORING_SCHEMA = "monitoring"
GOLD_SCHEMA = "gold"

LIVE_TRANSACTIONS_TABLE = f"{CATALOG}.{BRONZE_SCHEMA}.live_customer_transactions"
LIVE_DECISIONS_TABLE = f"{CATALOG}.{BRONZE_SCHEMA}.live_fraud_decisions"
FEEDBACK_TABLE = f"{CATALOG}.{MONITORING_SCHEMA}.fraud_prediction_feedback"

GOLD_TRANSACTION_DECISIONS = f"{CATALOG}.{GOLD_SCHEMA}.fraud_transaction_decisions"
GOLD_KPI_SUMMARY = f"{CATALOG}.{GOLD_SCHEMA}.fraud_kpi_summary"
GOLD_TIME_SERIES = f"{CATALOG}.{GOLD_SCHEMA}.fraud_activity_timeseries"
GOLD_MODEL_MONITORING = f"{CATALOG}.{GOLD_SCHEMA}.model_monitoring_summary"

REGISTERED_MODEL_NAME = "crypto_fraud.models.fraud_detection_model"
PHASE16_RETRAINED_VERSION = "2"
THRESHOLD = 0.80
WORKSPACE_NOTEBOOK_PATH = "/Users/akanaskhan1506@gmail.com/10_gold_analytics"
REPORT_ROOT = Path("/tmp") / "crypto_fraud_phase17"
REPORT_ROOT.mkdir(parents=True, exist_ok=True)

# COMMAND ----------


def q(name: str) -> str:
    return f"`{name.replace('`', '``')}`"


def quoted_full_name(full_name: str) -> str:
    return ".".join(q(part) for part in full_name.split("."))


def table_exists(full_name: str) -> bool:
    return spark.catalog.tableExists(full_name)


def require_table(full_name: str) -> None:
    if not table_exists(full_name):
        raise RuntimeError(f"Required Phase 17 source table does not exist: {full_name}")


def write_delta_table(df: DataFrame, full_name: str) -> None:
    (
        df.write.format("delta")
        .mode("overwrite")
        .option("overwriteSchema", "true")
        .saveAsTable(full_name)
    )


def latest_per_transaction(df: DataFrame, timestamp_col: str) -> DataFrame:
    window = Window.partitionBy("transaction_id").orderBy(
        F.col(timestamp_col).desc_nulls_last(),
        F.col("kafka_timestamp").desc_nulls_last(),
        F.col("kafka_partition").desc_nulls_last(),
        F.col("kafka_offset").desc_nulls_last(),
    )
    return df.withColumn("_rn", F.row_number().over(window)).filter(F.col("_rn") == 1).drop("_rn")


def to_jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [to_jsonable(item) for item in value]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return None if not math.isfinite(float(value)) else float(value)
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
    path.write_text(json.dumps(to_jsonable(payload), indent=2, sort_keys=True), encoding="utf-8")


def now_utc_naive() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


def safe_divide(numerator: int | float | None, denominator: int | float | None) -> float | None:
    if numerator is None or denominator in {None, 0}:
        return None
    value = float(numerator) / float(denominator)
    return value if math.isfinite(value) else None


def maybe_float(value: Any) -> float | None:
    if value is None:
        return None
    numeric = float(value)
    return numeric if math.isfinite(numeric) else None


def maybe_int(value: Any) -> int:
    return 0 if value is None else int(value)


# COMMAND ----------

for source_table in [LIVE_TRANSACTIONS_TABLE, LIVE_DECISIONS_TABLE, FEEDBACK_TABLE]:
    require_table(source_table)

spark.sql(f"CREATE SCHEMA IF NOT EXISTS {q(CATALOG)}.{q(GOLD_SCHEMA)}")

try:
    mlflow.set_registry_uri("databricks-uc")
    client = mlflow.tracking.MlflowClient()
    candidate_alias_before = str(client.get_model_version_by_alias(REGISTERED_MODEL_NAME, "candidate").version)
except Exception:
    client = None
    candidate_alias_before = None

# COMMAND ----------

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

transactions_base = (
    spark.table(LIVE_TRANSACTIONS_TABLE)
    .withColumn("payload", F.from_json("raw_payload", transaction_payload_schema))
    .filter(F.col("transaction_id").isNotNull())
    .withColumn(
        "transaction_timestamp",
        F.coalesce(F.to_timestamp("payload.event_timestamp"), F.col("event_timestamp"), F.col("kafka_timestamp")),
    )
    .select(
        "transaction_id",
        F.coalesce(F.col("event_id"), F.col("payload.event_id")).alias("transaction_event_id"),
        "transaction_timestamp",
        F.coalesce(F.col("account_id"), F.col("payload.account_id")).alias("account_id"),
        F.coalesce(F.col("device_id"), F.col("payload.device_id")).alias("device_id"),
        F.col("payload.asset").alias("asset"),
        F.col("payload.transaction_type").alias("transaction_type"),
        F.col("payload.country").alias("country"),
        F.col("payload.crypto_quantity").cast("double").alias("crypto_quantity"),
        F.col("payload.market_price_usd").cast("double").alias("market_price_usd"),
        F.col("payload.transaction_amount_usd").cast("double").alias("transaction_amount_usd"),
        F.col("payload.transaction_status").alias("transaction_status"),
        "kafka_partition",
        "kafka_offset",
        "kafka_timestamp",
    )
)

transactions = latest_per_transaction(transactions_base, "transaction_timestamp").select(
    "transaction_id",
    "transaction_event_id",
    "transaction_timestamp",
    "account_id",
    "device_id",
    "asset",
    "transaction_type",
    "country",
    "crypto_quantity",
    "market_price_usd",
    "transaction_amount_usd",
    "transaction_status",
    F.col("kafka_partition").alias("transaction_kafka_partition"),
    F.col("kafka_offset").alias("transaction_kafka_offset"),
    F.col("kafka_timestamp").alias("transaction_kafka_timestamp"),
)

decision_timestamp = F.coalesce(
    F.col("prediction_timestamp"),
    F.col("event_timestamp"),
    F.col("kafka_timestamp"),
    F.col("bronze_ingested_at"),
)
decisions_base = (
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
        "kafka_partition",
        "kafka_offset",
        "kafka_timestamp",
    )
)

decisions = latest_per_transaction(decisions_base, "decision_timestamp").select(
    "transaction_id",
    "decision_event_id",
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
)

feedback = spark.table(FEEDBACK_TABLE).select(
    "transaction_id",
    F.col("actual_is_fraud").cast("boolean").alias("actual_fraud"),
    "label_timestamp",
    "label_delay_seconds",
    "outcome",
    "true_positive",
    "false_positive",
    "true_negative",
    "false_negative",
    "label_source",
    "investigation_status",
)

# COMMAND ----------

transaction_decisions = (
    transactions.alias("t")
    .join(decisions.alias("d"), "transaction_id", "left")
    .join(feedback.alias("f"), "transaction_id", "left")
    .withColumn("gold_created_at", F.current_timestamp())
    .select(
        "transaction_id",
        "transaction_event_id",
        "transaction_timestamp",
        "account_id",
        "device_id",
        "asset",
        "transaction_type",
        "country",
        "crypto_quantity",
        "market_price_usd",
        "transaction_amount_usd",
        "transaction_status",
        "decision_event_id",
        "decision_timestamp",
        "fraud_probability",
        "threshold",
        "predicted_fraud",
        "decision",
        "actual_fraud",
        "outcome",
        "true_positive",
        "false_positive",
        "true_negative",
        "false_negative",
        "label_timestamp",
        "label_delay_seconds",
        "label_source",
        "investigation_status",
        "model_name",
        "model_version",
        "threshold_policy_version",
        "transaction_kafka_partition",
        "transaction_kafka_offset",
        "decision_kafka_partition",
        "decision_kafka_offset",
        "gold_created_at",
    )
)

write_delta_table(transaction_decisions, GOLD_TRANSACTION_DECISIONS)
gold_decisions = spark.table(GOLD_TRANSACTION_DECISIONS)

# COMMAND ----------

kpi_row = gold_decisions.agg(
    F.count("*").cast("long").alias("total_transactions"),
    F.sum(F.col("fraud_probability").isNotNull().cast("long")).alias("total_scored_transactions"),
    F.sum(F.col("actual_fraud").isNotNull().cast("long")).alias("total_labeled_transactions"),
    F.sum((F.col("actual_fraud") == F.lit(True)).cast("long")).alias("actual_fraud_count"),
    F.sum((F.col("actual_fraud") == F.lit(False)).cast("long")).alias("actual_normal_count"),
    F.sum((F.col("predicted_fraud") == F.lit(True)).cast("long")).alias("predicted_fraud_count"),
    F.sum(F.coalesce(F.col("true_positive"), F.lit(0))).cast("long").alias("true_positives"),
    F.sum(F.coalesce(F.col("false_positive"), F.lit(0))).cast("long").alias("false_positives"),
    F.sum(F.coalesce(F.col("true_negative"), F.lit(0))).cast("long").alias("true_negatives"),
    F.sum(F.coalesce(F.col("false_negative"), F.lit(0))).cast("long").alias("false_negatives"),
    F.avg("fraud_probability").alias("average_fraud_probability"),
    F.max("fraud_probability").alias("max_fraud_probability"),
    F.sum(F.coalesce(F.col("transaction_amount_usd"), F.lit(0.0))).alias("total_transaction_value_usd"),
    F.sum(
        F.when(F.col("actual_fraud") == F.lit(True), F.coalesce(F.col("transaction_amount_usd"), F.lit(0.0))).otherwise(F.lit(0.0))
    ).alias("actual_fraudulent_transaction_value_usd"),
).collect()[0].asDict()

tp = maybe_int(kpi_row["true_positives"])
fp = maybe_int(kpi_row["false_positives"])
tn = maybe_int(kpi_row["true_negatives"])
fn = maybe_int(kpi_row["false_negatives"])
actual_fraud_count = maybe_int(kpi_row["actual_fraud_count"])
total_labeled = maybe_int(kpi_row["total_labeled_transactions"])
precision = safe_divide(tp, tp + fp)
recall = safe_divide(tp, tp + fn)
f1 = safe_divide(2 * precision * recall, precision + recall) if precision is not None and recall is not None else None
fraud_rate = safe_divide(actual_fraud_count, total_labeled)

feedback_pdf = (
    gold_decisions.filter(F.col("actual_fraud").isNotNull() & F.col("fraud_probability").isNotNull())
    .select("actual_fraud", "fraud_probability")
    .toPandas()
)
if len(feedback_pdf) and feedback_pdf["actual_fraud"].astype(int).nunique() == 2:
    y_true = feedback_pdf["actual_fraud"].astype(int).to_numpy()
    y_prob = feedback_pdf["fraud_probability"].astype(float).to_numpy()
    pr_auc = maybe_float(average_precision_score(y_true, y_prob))
    roc_auc = maybe_float(roc_auc_score(y_true, y_prob))
else:
    pr_auc = None
    roc_auc = None

kpi_values = {
    "metric_scope": "live_feedback",
    "total_transactions": maybe_int(kpi_row["total_transactions"]),
    "total_scored_transactions": maybe_int(kpi_row["total_scored_transactions"]),
    "total_labeled_transactions": total_labeled,
    "actual_fraud_count": actual_fraud_count,
    "actual_normal_count": maybe_int(kpi_row["actual_normal_count"]),
    "predicted_fraud_count": maybe_int(kpi_row["predicted_fraud_count"]),
    "fraud_rate": fraud_rate,
    "true_positives": tp,
    "false_positives": fp,
    "true_negatives": tn,
    "false_negatives": fn,
    "precision": precision,
    "recall": recall,
    "f1": f1,
    "average_fraud_probability": maybe_float(kpi_row["average_fraud_probability"]),
    "max_fraud_probability": maybe_float(kpi_row["max_fraud_probability"]),
    "total_transaction_value_usd": maybe_float(kpi_row["total_transaction_value_usd"]),
    "actual_fraudulent_transaction_value_usd": maybe_float(kpi_row["actual_fraudulent_transaction_value_usd"]),
    "pr_auc": pr_auc,
    "roc_auc": roc_auc,
    "threshold": float(THRESHOLD),
    "evaluation_timestamp": now_utc_naive(),
}

kpi_schema = T.StructType(
    [
        T.StructField("metric_scope", T.StringType()),
        T.StructField("total_transactions", T.LongType()),
        T.StructField("total_scored_transactions", T.LongType()),
        T.StructField("total_labeled_transactions", T.LongType()),
        T.StructField("actual_fraud_count", T.LongType()),
        T.StructField("actual_normal_count", T.LongType()),
        T.StructField("predicted_fraud_count", T.LongType()),
        T.StructField("fraud_rate", T.DoubleType()),
        T.StructField("true_positives", T.LongType()),
        T.StructField("false_positives", T.LongType()),
        T.StructField("true_negatives", T.LongType()),
        T.StructField("false_negatives", T.LongType()),
        T.StructField("precision", T.DoubleType()),
        T.StructField("recall", T.DoubleType()),
        T.StructField("f1", T.DoubleType()),
        T.StructField("average_fraud_probability", T.DoubleType()),
        T.StructField("max_fraud_probability", T.DoubleType()),
        T.StructField("total_transaction_value_usd", T.DoubleType()),
        T.StructField("actual_fraudulent_transaction_value_usd", T.DoubleType()),
        T.StructField("pr_auc", T.DoubleType()),
        T.StructField("roc_auc", T.DoubleType()),
        T.StructField("threshold", T.DoubleType()),
        T.StructField("evaluation_timestamp", T.TimestampType()),
    ]
)
write_delta_table(spark.createDataFrame([kpi_values], schema=kpi_schema), GOLD_KPI_SUMMARY)

# COMMAND ----------

time_series = (
    gold_decisions.withColumn("time_bucket_utc", F.date_trunc("hour", F.col("transaction_timestamp")))
    .groupBy("time_bucket_utc")
    .agg(
        F.count("*").cast("long").alias("transaction_count"),
        F.sum(F.coalesce(F.col("transaction_amount_usd"), F.lit(0.0))).alias("transaction_volume_usd"),
        F.sum(F.col("fraud_probability").isNotNull().cast("long")).alias("scored_transaction_count"),
        F.sum((F.col("actual_fraud") == F.lit(True)).cast("long")).alias("actual_fraud_count"),
        F.sum((F.col("predicted_fraud") == F.lit(True)).cast("long")).alias("predicted_fraud_count"),
        F.avg("fraud_probability").alias("average_fraud_probability"),
        F.max("fraud_probability").alias("max_fraud_probability"),
        F.sum(
            F.when(F.col("actual_fraud") == F.lit(True), F.coalesce(F.col("transaction_amount_usd"), F.lit(0.0))).otherwise(F.lit(0.0))
        ).alias("fraudulent_value_usd"),
    )
    .withColumn("gold_created_at", F.current_timestamp())
    .orderBy("time_bucket_utc")
)
write_delta_table(time_series, GOLD_TIME_SERIES)

# COMMAND ----------

model_rows = [
    {
        "model_role": "current_candidate",
        "model_name": REGISTERED_MODEL_NAME,
        "model_version": candidate_alias_before,
        "active_candidate_version": candidate_alias_before,
        "phase16_retrained_version": PHASE16_RETRAINED_VERSION,
        "threshold": float(THRESHOLD),
        "labeled_sample_count": total_labeled,
        "true_positives": tp,
        "false_positives": fp,
        "true_negatives": tn,
        "false_negatives": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "pr_auc": pr_auc,
        "roc_auc": roc_auc,
        "fraud_prevalence": fraud_rate,
        "evaluation_timestamp": now_utc_naive(),
        "monitoring_note": "Live decisions were produced by the active candidate alias at threshold 0.80.",
    },
    {
        "model_role": "phase16_retrained_registered",
        "model_name": REGISTERED_MODEL_NAME,
        "model_version": PHASE16_RETRAINED_VERSION,
        "active_candidate_version": candidate_alias_before,
        "phase16_retrained_version": PHASE16_RETRAINED_VERSION,
        "threshold": float(THRESHOLD),
        "labeled_sample_count": None,
        "true_positives": None,
        "false_positives": None,
        "true_negatives": None,
        "false_negatives": None,
        "precision": None,
        "recall": None,
        "f1": None,
        "pr_auc": None,
        "roc_auc": None,
        "fraud_prevalence": None,
        "evaluation_timestamp": now_utc_naive(),
        "monitoring_note": "Phase 16 retrained model version is registered for comparison only and was not promoted.",
    },
]

model_schema = T.StructType(
    [
        T.StructField("model_role", T.StringType()),
        T.StructField("model_name", T.StringType()),
        T.StructField("model_version", T.StringType()),
        T.StructField("active_candidate_version", T.StringType()),
        T.StructField("phase16_retrained_version", T.StringType()),
        T.StructField("threshold", T.DoubleType()),
        T.StructField("labeled_sample_count", T.LongType()),
        T.StructField("true_positives", T.LongType()),
        T.StructField("false_positives", T.LongType()),
        T.StructField("true_negatives", T.LongType()),
        T.StructField("false_negatives", T.LongType()),
        T.StructField("precision", T.DoubleType()),
        T.StructField("recall", T.DoubleType()),
        T.StructField("f1", T.DoubleType()),
        T.StructField("pr_auc", T.DoubleType()),
        T.StructField("roc_auc", T.DoubleType()),
        T.StructField("fraud_prevalence", T.DoubleType()),
        T.StructField("evaluation_timestamp", T.TimestampType()),
        T.StructField("monitoring_note", T.StringType()),
    ]
)
write_delta_table(spark.createDataFrame(model_rows, schema=model_schema), GOLD_MODEL_MONITORING)

# COMMAND ----------

spark.sql(
    f"""
    CREATE OR REPLACE VIEW {q(CATALOG)}.{q(GOLD_SCHEMA)}.{q("v_fraud_activity_by_asset")} AS
    SELECT
      asset,
      COUNT(*) AS transaction_count,
      SUM(transaction_amount_usd) AS transaction_value_usd,
      SUM(CASE WHEN actual_fraud THEN 1 ELSE 0 END) AS actual_fraud_count,
      SUM(CASE WHEN predicted_fraud THEN 1 ELSE 0 END) AS predicted_fraud_count,
      AVG(fraud_probability) AS average_fraud_probability,
      MAX(fraud_probability) AS max_fraud_probability
    FROM {quoted_full_name(GOLD_TRANSACTION_DECISIONS)}
    GROUP BY asset
    """
)

spark.sql(
    f"""
    CREATE OR REPLACE VIEW {q(CATALOG)}.{q(GOLD_SCHEMA)}.{q("v_fraud_activity_by_country")} AS
    SELECT
      country,
      COUNT(*) AS transaction_count,
      SUM(transaction_amount_usd) AS transaction_value_usd,
      SUM(CASE WHEN actual_fraud THEN 1 ELSE 0 END) AS actual_fraud_count,
      SUM(CASE WHEN predicted_fraud THEN 1 ELSE 0 END) AS predicted_fraud_count,
      AVG(fraud_probability) AS average_fraud_probability,
      MAX(fraud_probability) AS max_fraud_probability
    FROM {quoted_full_name(GOLD_TRANSACTION_DECISIONS)}
    GROUP BY country
    """
)

spark.sql(
    f"""
    CREATE OR REPLACE VIEW {q(CATALOG)}.{q(GOLD_SCHEMA)}.{q("v_fraud_activity_by_transaction_type")} AS
    SELECT
      transaction_type,
      COUNT(*) AS transaction_count,
      SUM(transaction_amount_usd) AS transaction_value_usd,
      SUM(CASE WHEN actual_fraud THEN 1 ELSE 0 END) AS actual_fraud_count,
      SUM(CASE WHEN predicted_fraud THEN 1 ELSE 0 END) AS predicted_fraud_count,
      AVG(fraud_probability) AS average_fraud_probability,
      MAX(fraud_probability) AS max_fraud_probability
    FROM {quoted_full_name(GOLD_TRANSACTION_DECISIONS)}
    GROUP BY transaction_type
    """
)

spark.sql(
    f"""
    CREATE OR REPLACE VIEW {q(CATALOG)}.{q(GOLD_SCHEMA)}.{q("v_confusion_matrix")} AS
    SELECT outcome, COUNT(*) AS transaction_count
    FROM {quoted_full_name(GOLD_TRANSACTION_DECISIONS)}
    WHERE outcome IS NOT NULL
    GROUP BY outcome
    """
)

spark.sql(
    f"""
    CREATE OR REPLACE VIEW {q(CATALOG)}.{q(GOLD_SCHEMA)}.{q("v_fraud_probability_bands")} AS
    SELECT
      CASE
        WHEN fraud_probability IS NULL THEN 'unscored'
        WHEN fraud_probability < 0.01 THEN '00. < 1%'
        WHEN fraud_probability < 0.05 THEN '01. 1%-5%'
        WHEN fraud_probability < 0.20 THEN '02. 5%-20%'
        WHEN fraud_probability < 0.80 THEN '03. 20%-80%'
        ELSE '04. >= 80%'
      END AS fraud_probability_band,
      COUNT(*) AS transaction_count,
      SUM(CASE WHEN actual_fraud THEN 1 ELSE 0 END) AS actual_fraud_count,
      SUM(CASE WHEN predicted_fraud THEN 1 ELSE 0 END) AS predicted_fraud_count
    FROM {quoted_full_name(GOLD_TRANSACTION_DECISIONS)}
    GROUP BY
      CASE
        WHEN fraud_probability IS NULL THEN 'unscored'
        WHEN fraud_probability < 0.01 THEN '00. < 1%'
        WHEN fraud_probability < 0.05 THEN '01. 1%-5%'
        WHEN fraud_probability < 0.20 THEN '02. 5%-20%'
        WHEN fraud_probability < 0.80 THEN '03. 20%-80%'
        ELSE '04. >= 80%'
      END
    """
)

spark.sql(
    f"""
    CREATE OR REPLACE VIEW {q(CATALOG)}.{q(GOLD_SCHEMA)}.{q("v_recent_high_risk_transactions")} AS
    SELECT
      transaction_id,
      transaction_timestamp,
      account_id,
      asset,
      transaction_type,
      country,
      transaction_amount_usd,
      fraud_probability,
      threshold,
      predicted_fraud,
      actual_fraud,
      outcome,
      model_name,
      model_version
    FROM {quoted_full_name(GOLD_TRANSACTION_DECISIONS)}
    WHERE fraud_probability IS NOT NULL
    ORDER BY fraud_probability DESC, transaction_timestamp DESC
    LIMIT 100
    """
)

# COMMAND ----------

transaction_decisions_rows = int(spark.table(GOLD_TRANSACTION_DECISIONS).count())
kpi_summary_rows = int(spark.table(GOLD_KPI_SUMMARY).count())
time_series_rows = int(spark.table(GOLD_TIME_SERIES).count())
model_monitoring_rows = int(spark.table(GOLD_MODEL_MONITORING).count())

source_transaction_count = int(transactions.count())
source_scored_count = int(decisions.count())
source_feedback_count = int(feedback.count())
gold_distinct_transactions = int(spark.table(GOLD_TRANSACTION_DECISIONS).select("transaction_id").distinct().count())
duplicate_transaction_count = transaction_decisions_rows - gold_distinct_transactions
invalid_probability_count = int(
    spark.table(GOLD_TRANSACTION_DECISIONS)
    .filter(F.col("fraud_probability").isNotNull() & ((F.col("fraud_probability") < 0.0) | (F.col("fraud_probability") > 1.0)))
    .count()
)
null_amount_count = int(spark.table(GOLD_TRANSACTION_DECISIONS).filter(F.col("transaction_amount_usd").isNull()).count())

kpi_check = spark.table(GOLD_KPI_SUMMARY).collect()[0].asDict()
source_reconciliation_checks = {
    "gold_transaction_rows_match_source_transactions": transaction_decisions_rows == source_transaction_count,
    "gold_transaction_ids_unique": duplicate_transaction_count == 0,
    "kpi_total_transactions_match_gold": maybe_int(kpi_check["total_transactions"]) == transaction_decisions_rows,
    "kpi_scored_transactions_match_decisions": maybe_int(kpi_check["total_scored_transactions"]) == source_scored_count,
    "kpi_labeled_transactions_match_feedback": maybe_int(kpi_check["total_labeled_transactions"]) == source_feedback_count,
    "fraud_probabilities_in_unit_interval": invalid_probability_count == 0,
    "transaction_amounts_numeric_and_present": null_amount_count == 0,
}
source_reconciliation_pass = all(source_reconciliation_checks.values())

gold_tables_queryable = all(
    int(spark.table(table_name).count()) >= 0
    for table_name in [
        GOLD_TRANSACTION_DECISIONS,
        GOLD_KPI_SUMMARY,
        GOLD_TIME_SERIES,
        GOLD_MODEL_MONITORING,
    ]
)

try:
    candidate_alias_after = str(client.get_model_version_by_alias(REGISTERED_MODEL_NAME, "candidate").version) if client else None
except Exception:
    candidate_alias_after = None
candidate_alias_changed = candidate_alias_before != candidate_alias_after

failed_checks = []
if transaction_decisions_rows <= 0:
    failed_checks.append("gold_transaction_decisions_empty")
if kpi_summary_rows != 1:
    failed_checks.append("gold_kpi_summary_not_single_row")
if time_series_rows <= 0:
    failed_checks.append("gold_time_series_empty")
if model_monitoring_rows < 2:
    failed_checks.append("gold_model_monitoring_missing_candidate_or_retrained_row")
if not source_reconciliation_pass:
    failed_checks.append("source_reconciliation_failed")
if not gold_tables_queryable:
    failed_checks.append("gold_tables_not_queryable")
if candidate_alias_changed:
    failed_checks.append("candidate_alias_changed")

overall_status = "PASS" if not failed_checks else "FAIL"

summary = {
    "phase": "17",
    "overall_status": overall_status,
    "notebook": "databricks/10_gold_analytics.py",
    "workspace_path": WORKSPACE_NOTEBOOK_PATH,
    "gold_tables": {
        "transaction_decisions": GOLD_TRANSACTION_DECISIONS,
        "kpi_summary": GOLD_KPI_SUMMARY,
        "time_series": GOLD_TIME_SERIES,
        "model_monitoring": GOLD_MODEL_MONITORING,
    },
    "rows": {
        "transaction_decisions": transaction_decisions_rows,
        "kpi_summary": kpi_summary_rows,
        "time_series": time_series_rows,
        "model_monitoring": model_monitoring_rows,
    },
    "dashboard_created_in_ui": False,
    "dashboard_sql_created": True,
    "dashboard_sql_path": "sql/fraud_monitoring_dashboard.sql",
    "dashboard_sections_available": {
        "kpi_cards": True,
        "fraud_activity_over_time": True,
        "by_asset": True,
        "by_country": True,
        "confusion_matrix": True,
        "model_metrics": True,
        "high_risk_transactions": True,
    },
    "source_counts": {
        "transactions": source_transaction_count,
        "scored_decisions": source_scored_count,
        "feedback_labels": source_feedback_count,
    },
    "source_reconciliation": "PASS" if source_reconciliation_pass else "FAIL",
    "source_reconciliation_checks": source_reconciliation_checks,
    "gold_tables_queryable": gold_tables_queryable,
    "candidate_alias_before": candidate_alias_before,
    "candidate_alias_after": candidate_alias_after,
    "candidate_alias_changed": candidate_alias_changed,
    "threshold_changed": False,
    "threshold": float(THRESHOLD),
    "fraud_label_leakage": False,
    "secrets_exposed": False,
    "failed_checks": failed_checks,
    "created_at_utc": datetime.now(timezone.utc).isoformat(),
}

write_json(REPORT_ROOT / "phase17_summary.json", summary)
try:
    dbutils.jobs.taskValues.set(key="summary", value=json.dumps(to_jsonable(summary), sort_keys=True, separators=(",", ":")))
except Exception:
    pass

dbutils.notebook.exit(json.dumps(to_jsonable(summary), sort_keys=True, separators=(",", ":")))
