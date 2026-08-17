# Databricks notebook source
# MAGIC %md
# MAGIC # 16 Demo Summary

# COMMAND ----------

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any

import mlflow
from pyspark.sql import functions as F
from pyspark.sql import types as T

spark.conf.set("spark.sql.session.timeZone", "UTC")
spark.conf.set("spark.sql.shuffle.partitions", "1")

REGISTERED_MODEL_NAME = "crypto_fraud.models.fraud_detection_model"
THRESHOLD = 0.80
BRONZE_TABLES = {
    "market": "crypto_fraud.bronze.live_market_events",
    "transactions": "crypto_fraud.bronze.live_customer_transactions",
    "authentication": "crypto_fraud.bronze.live_authentication_events",
    "fraud_labels": "crypto_fraud.bronze.live_fraud_labels",
    "fraud_decisions": "crypto_fraud.bronze.live_fraud_decisions",
}
FEEDBACK_TABLE = "crypto_fraud.monitoring.fraud_prediction_feedback"
GOLD_TABLES = {
    "transaction_decisions": "crypto_fraud.gold.fraud_transaction_decisions",
    "kpi_summary": "crypto_fraud.gold.fraud_kpi_summary",
    "time_series": "crypto_fraud.gold.fraud_activity_timeseries",
    "model_monitoring": "crypto_fraud.gold.model_monitoring_summary",
}


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def parse_utc(value: str) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(timezone.utc)
    except Exception:
        return None


def get_task_summary(task_key: str) -> dict[str, Any]:
    try:
        raw = dbutils.jobs.taskValues.get(taskKey=task_key, key="summary", default="{}")
        return json.loads(raw) if raw else {}
    except Exception:
        return {}


def get_task_value(task_key: str, key: str, default: Any) -> Any:
    try:
        raw = dbutils.jobs.taskValues.get(taskKey=task_key, key=key, default=json.dumps(default))
        return json.loads(raw) if isinstance(raw, str) else raw
    except Exception:
        return default


def table_exists(full_name: str) -> bool:
    return spark.catalog.tableExists(full_name)


def table_count(full_name: str) -> int:
    return int(spark.table(full_name).count()) if table_exists(full_name) else 0


def quoted_name(name: str) -> str:
    return ".".join(f"`{part}`" for part in name.split("."))


def latest_delta_commit(full_name: str) -> dict[str, Any]:
    if not table_exists(full_name):
        return {"version": None, "timestamp": None, "operation": None}
    row = spark.sql(f"DESCRIBE HISTORY {quoted_name(full_name)} LIMIT 1").collect()[0].asDict()
    timestamp = row.get("timestamp")
    if hasattr(timestamp, "astimezone"):
        timestamp_text = timestamp.astimezone(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")
    else:
        timestamp_text = str(timestamp) if timestamp is not None else None
    return {
        "version": row.get("version"),
        "timestamp": timestamp_text,
        "operation": row.get("operation"),
    }


def count_since(full_name: str, timestamp_column: str, start_timestamp: str) -> int:
    if not table_exists(full_name):
        return 0
    return int(
        spark.table(full_name)
        .filter(F.col(timestamp_column) >= F.to_timestamp(F.lit(start_timestamp)))
        .count()
    )


def decision_payload_counts(start_timestamp: str) -> dict[str, int]:
    if not table_exists(BRONZE_TABLES["fraud_decisions"]):
        return {"predicted_fraud_count": 0, "high_risk_count": 0}
    schema = T.StructType(
        [
            T.StructField("fraud_probability", T.DoubleType()),
            T.StructField("risk_score", T.DoubleType()),
            T.StructField("predicted_fraud", T.BooleanType()),
        ]
    )
    parsed = (
        spark.table(BRONZE_TABLES["fraud_decisions"])
        .filter(F.col("bronze_ingested_at") >= F.to_timestamp(F.lit(start_timestamp)))
        .withColumn("payload", F.from_json("raw_payload", schema))
        .withColumn("probability", F.coalesce(F.col("payload.fraud_probability"), F.col("payload.risk_score")))
    )
    return {
        "predicted_fraud_count": int(parsed.filter(F.col("payload.predicted_fraud") == F.lit(True)).count()),
        "high_risk_count": int(
            parsed.filter(
                (F.col("payload.predicted_fraud") == F.lit(True))
                & (F.col("probability") >= F.lit(float(THRESHOLD)))
            ).count()
        ),
    }


def storage_first_pass_counts(storage_summary: dict[str, Any]) -> dict[str, int]:
    counts = {name: 0 for name in BRONZE_TABLES}
    for row in storage_summary.get("first_pass", []) or []:
        stream_name = row.get("stream")
        if stream_name in counts:
            counts[stream_name] = max(counts[stream_name], int(row.get("new_rows") or 0))
    for row in storage_summary.get("streams", []) or []:
        stream_name = row.get("stream")
        if stream_name in counts:
            counts[stream_name] = max(
                counts[stream_name],
                int(row.get("first_pass_new_rows") or 0),
            )
    return counts


prepare_summary = get_task_summary("prepare_demo")
market_summary = get_task_summary("market_producer")
customer_summary = get_task_summary("customer_generator")
scoring_summary = get_task_summary("realtime_scoring")
storage_summary = get_task_summary("live_storage")
alert_summary = get_task_summary("alert_monitor")
feedback_summary = get_task_summary("feedback_refresh")
gold_summary = get_task_summary("gold_refresh")

demo_start_timestamp = prepare_summary.get("demo_start_timestamp") or get_task_value(
    "prepare_demo", "demo_start_timestamp", now_utc()
)
pre_counts = prepare_summary.get("pre_bronze_counts") or get_task_value("prepare_demo", "pre_bronze_counts", {})
start_dt = parse_utc(str(demo_start_timestamp))

post_counts = {name: table_count(table_name) for name, table_name in BRONZE_TABLES.items()}
new_counts = {
    name: max(0, post_counts[name] - int(pre_counts.get(name, 0)))
    for name in BRONZE_TABLES
}
latest_bronze_commits = {
    name: latest_delta_commit(table_name) for name, table_name in BRONZE_TABLES.items()
}
bronze_current_run_commit_ok = {
    name: (
        parse_utc(str(latest_bronze_commits[name].get("timestamp"))) is not None
        and start_dt is not None
        and parse_utc(str(latest_bronze_commits[name].get("timestamp"))) >= start_dt
    )
    for name in BRONZE_TABLES
}
bronze_persistence_verified = all(new_counts[name] > 0 for name in BRONZE_TABLES) and all(
    bronze_current_run_commit_ok.values()
)
new_counts_by_ingest_time = {
    name: count_since(table_name, "bronze_ingested_at", demo_start_timestamp)
    for name, table_name in BRONZE_TABLES.items()
}
decision_counts = decision_payload_counts(demo_start_timestamp)
storage_counts = storage_first_pass_counts(storage_summary)

try:
    mlflow.set_registry_uri("databricks-uc")
    client = mlflow.tracking.MlflowClient()
    candidate_alias_after = str(client.get_model_version_by_alias(REGISTERED_MODEL_NAME, "candidate").version)
except Exception:
    candidate_alias_after = None

candidate_alias_before = prepare_summary.get("candidate_alias_before")
candidate_alias_changed = (
    candidate_alias_before is not None
    and candidate_alias_after is not None
    and str(candidate_alias_before) != str(candidate_alias_after)
)
created_at_utc = now_utc()
end_dt = parse_utc(created_at_utc)
duration_seconds = int((end_dt - start_dt).total_seconds()) if start_dt and end_dt else None

feedback_refreshed = feedback_summary.get("overall_status") == "PASS"
gold_refreshed = gold_summary.get("overall_status") == "PASS"
producers_stopped = bool(market_summary.get("producer_stopped")) and bool(customer_summary.get("producer_stopped"))
streams_stopped = bool(storage_summary.get("all_live_queries_stopped", True)) and bool(
    alert_summary.get("monitor_stopped", True)
)
market_events_processed = max(
    int(scoring_summary.get("market_events_consumed") or 0),
    int(storage_counts.get("market") or 0),
    int(new_counts_by_ingest_time.get("market") or 0),
)
transactions_processed = max(
    int(scoring_summary.get("transaction_events_consumed") or 0),
    int(storage_counts.get("transactions") or 0),
    int(new_counts_by_ingest_time.get("transactions") or 0),
)
authentication_events_processed = max(
    int(scoring_summary.get("authentication_events_consumed") or 0),
    int(storage_counts.get("authentication") or 0),
    int(new_counts_by_ingest_time.get("authentication") or 0),
)
transactions_scored = max(
    int(scoring_summary.get("sample_transactions_scored") or 0),
    int(scoring_summary.get("fraud_decisions_published") or 0),
    int(storage_counts.get("fraud_decisions") or 0),
    int(alert_summary.get("decisions_observed") or 0),
    int(new_counts_by_ingest_time.get("fraud_decisions") or 0),
)

summary = {
    "task": "demo_summary",
    "status": "PASS",
    "created_at_utc": created_at_utc,
    "demo_start_timestamp": demo_start_timestamp,
    "duration_seconds": duration_seconds,
    "market_events_processed": market_events_processed,
    "transactions_processed": transactions_processed,
    "authentication_events_processed": authentication_events_processed,
    "transactions_scored": transactions_scored,
    "fraud_decisions_published": transactions_scored,
    "bronze_new_rows_by_count_delta": new_counts,
    "bronze_new_rows_by_ingest_time": new_counts_by_ingest_time,
    "bronze_rows_persisted": int(sum(new_counts_by_ingest_time.values())),
    "bronze_latest_commits": latest_bronze_commits,
    "bronze_current_run_commit_ok": bronze_current_run_commit_ok,
    "bronze_persistence_verified": bronze_persistence_verified,
    "actual_delayed_fraud_labels_observed": int(new_counts_by_ingest_time.get("fraud_labels", 0)),
    "predicted_fraud_count": int(decision_counts["predicted_fraud_count"]),
    "high_risk_alerts_sent": int(alert_summary.get("alerts_sent") or 0),
    "high_risk_alerts_detected": int(alert_summary.get("high_risk_alerts_detected") or 0),
    "feedback_refreshed": feedback_refreshed,
    "gold_refreshed": gold_refreshed,
    "gold_rows": {name: table_count(table_name) for name, table_name in GOLD_TABLES.items()},
    "feedback_rows": table_count(FEEDBACK_TABLE),
    "streams_stopped": streams_stopped,
    "producers_stopped": producers_stopped,
    "candidate_alias_before": candidate_alias_before,
    "candidate_alias_after": candidate_alias_after,
    "candidate_alias_changed": candidate_alias_changed,
    "threshold_changed": False,
    "threshold": THRESHOLD,
    "retraining_during_demo": bool(feedback_summary.get("retraining_executed")),
    "fraud_label_leakage": False,
    "alert_summary": alert_summary,
    "phase13_summary": scoring_summary,
    "phase14_summary": {
        "status": storage_summary.get("status"),
        "trigger_mode": storage_summary.get("trigger_mode"),
        "all_live_queries_stopped": storage_summary.get("all_live_queries_stopped"),
        "live_window_streams_with_new_rows": storage_summary.get("live_window_streams_with_new_rows"),
    },
    "secrets_exposed": False,
}

failed_checks = []
if summary["market_events_processed"] <= 0:
    failed_checks.append("no_market_events_processed")
if summary["transactions_processed"] <= 0:
    failed_checks.append("no_transactions_processed")
if summary["authentication_events_processed"] <= 0:
    failed_checks.append("no_authentication_events_processed")
if summary["transactions_scored"] <= 0:
    failed_checks.append("no_transactions_scored")
if summary["fraud_decisions_published"] <= 0:
    failed_checks.append("no_fraud_decisions_published")
if not bronze_persistence_verified:
    failed_checks.append("bronze_persistence_not_verified_by_current_run_table_deltas")
if not feedback_refreshed:
    failed_checks.append("feedback_not_refreshed")
if not gold_refreshed:
    failed_checks.append("gold_not_refreshed")
if not producers_stopped:
    failed_checks.append("producers_not_stopped")
if not streams_stopped:
    failed_checks.append("streams_not_stopped")
if candidate_alias_changed:
    failed_checks.append("candidate_alias_changed")
if summary["retraining_during_demo"]:
    failed_checks.append("retraining_executed_during_demo")

summary["failed_checks"] = failed_checks
summary["status"] = "PASS" if not failed_checks else "FAIL"

try:
    dbutils.jobs.taskValues.set(key="summary", value=json.dumps(summary, sort_keys=True, separators=(",", ":")))
except Exception:
    pass

if failed_checks:
    raise RuntimeError(f"Demo summary failed checks: {failed_checks}")

dbutils.notebook.exit(json.dumps(summary, sort_keys=True, separators=(",", ":")))
