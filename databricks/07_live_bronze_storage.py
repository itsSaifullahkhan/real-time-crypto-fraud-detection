# Databricks notebook source
# MAGIC %md
# MAGIC # 07 Live Bronze Storage
# MAGIC
# MAGIC Persist live Event Hubs records into raw Unity Catalog Bronze Delta tables.

# COMMAND ----------

from __future__ import annotations

import json
import re
import time
from typing import Any

from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from pyspark.sql import types as T

spark.conf.set("spark.sql.session.timeZone", "UTC")
spark.conf.set("spark.sql.shuffle.partitions", "1")

CATALOG = "crypto_fraud"
BRONZE_SCHEMA = "bronze"
SECRET_SCOPE = "crypto-fraud-secrets"
SECRET_KEY = "eventhubs-databricks-connection"
CONSUMER_GROUP = "bronze-storage"
CHECKPOINT_BASE_PATH = (
    "/Volumes/crypto_fraud/monitoring/ingestion_checkpoints/phase14_live_bronze_storage"
)
DEMO_LIVE_CHECKPOINT_PATH = f"{CHECKPOINT_BASE_PATH}/live_demo_combined_v4/checkpoint"

dbutils.widgets.text("starting_offsets", "earliest")
dbutils.widgets.text("run_resume_check", "true")
dbutils.widgets.text("trigger_mode", "availableNow")
dbutils.widgets.text("timeout_seconds", "300")
dbutils.widgets.text("processing_time_trigger", "10 seconds")

STARTING_OFFSETS = dbutils.widgets.get("starting_offsets") or "earliest"
RUN_RESUME_CHECK = (dbutils.widgets.get("run_resume_check") or "true").lower() == "true"
TRIGGER_MODE = (dbutils.widgets.get("trigger_mode") or "availableNow").strip()
TIMEOUT_SECONDS = int(dbutils.widgets.get("timeout_seconds") or "300")
PROCESSING_TIME_TRIGGER = dbutils.widgets.get("processing_time_trigger") or "10 seconds"


def field(name: str, data_type: T.DataType, nullable: bool = True) -> T.StructField:
    return T.StructField(name, data_type, nullable)


common_fields = [
    field("event_id", T.StringType()),
    field("event_type", T.StringType()),
    field("schema_version", T.StringType()),
    field("source", T.StringType()),
    field("event_timestamp", T.StringType()),
    field("source_timestamp", T.StringType()),
    field("ingestion_timestamp", T.StringType()),
]

market_schema = T.StructType(
    common_fields
    + [
        field("product_id", T.StringType()),
        field("trade_id", T.StringType()),
        field("price_usd", T.DoubleType()),
        field("size", T.DoubleType()),
        field("side", T.StringType()),
        field("trade_timestamp", T.StringType()),
        field("message_timestamp", T.StringType()),
        field("sequence_number", T.LongType()),
    ]
)

transaction_schema = T.StructType(
    common_fields
    + [
        field("transaction_id", T.StringType()),
        field("account_id", T.StringType()),
        field("asset", T.StringType()),
        field("crypto_quantity", T.DoubleType()),
        field("transaction_type", T.StringType()),
        field("source_wallet_id", T.StringType()),
        field("destination_wallet_id", T.StringType()),
        field("device_id", T.StringType()),
        field("country", T.StringType()),
        field("market_price_usd", T.DoubleType()),
        field("transaction_amount_usd", T.DoubleType()),
        field("transaction_status", T.StringType()),
    ]
)

authentication_schema = T.StructType(
    common_fields
    + [
        field("login_id", T.StringType()),
        field("account_id", T.StringType()),
        field("device_id", T.StringType()),
        field("country", T.StringType()),
        field("ip_address", T.StringType()),
        field("login_success", T.BooleanType()),
        field("mfa_success", T.BooleanType()),
        field("password_reset_flag", T.BooleanType()),
        field("failure_reason", T.StringType()),
    ]
)

fraud_label_schema = T.StructType(
    common_fields
    + [
        field("transaction_id", T.StringType()),
        field("is_fraud", T.BooleanType()),
        field("fraud_type", T.StringType()),
        field("label_timestamp", T.StringType()),
        field("label_source", T.StringType()),
        field("investigation_status", T.StringType()),
    ]
)

fraud_decision_schema = T.StructType(
    common_fields
    + [
        field("transaction_id", T.StringType()),
        field("risk_score", T.DoubleType()),
        field("decision", T.StringType()),
        field("reason_codes", T.ArrayType(T.StringType())),
        field("model_name", T.StringType()),
        field("model_version", T.StringType()),
        field("prediction_timestamp", T.StringType()),
        field("processing_latency_ms", T.DoubleType()),
        field("threshold_policy_version", T.StringType()),
    ]
)

bronze_event_schema = T.StructType(
    common_fields
    + [
        field("transaction_id", T.StringType()),
        field("account_id", T.StringType()),
        field("asset", T.StringType()),
        field("crypto_quantity", T.DoubleType()),
        field("transaction_type", T.StringType()),
        field("source_wallet_id", T.StringType()),
        field("destination_wallet_id", T.StringType()),
        field("device_id", T.StringType()),
        field("country", T.StringType()),
        field("market_price_usd", T.DoubleType()),
        field("transaction_amount_usd", T.DoubleType()),
        field("transaction_status", T.StringType()),
        field("login_id", T.StringType()),
        field("ip_address", T.StringType()),
        field("login_success", T.BooleanType()),
        field("mfa_success", T.BooleanType()),
        field("password_reset_flag", T.BooleanType()),
        field("failure_reason", T.StringType()),
        field("product_id", T.StringType()),
        field("trade_id", T.StringType()),
        field("price_usd", T.DoubleType()),
        field("size", T.DoubleType()),
        field("side", T.StringType()),
        field("trade_timestamp", T.StringType()),
        field("message_timestamp", T.StringType()),
        field("sequence_number", T.LongType()),
        field("is_fraud", T.BooleanType()),
        field("fraud_type", T.StringType()),
        field("label_timestamp", T.StringType()),
        field("label_source", T.StringType()),
        field("investigation_status", T.StringType()),
        field("risk_score", T.DoubleType()),
        field("decision", T.StringType()),
        field("reason_codes", T.ArrayType(T.StringType())),
        field("model_name", T.StringType()),
        field("model_version", T.StringType()),
        field("prediction_timestamp", T.StringType()),
        field("processing_latency_ms", T.DoubleType()),
        field("threshold_policy_version", T.StringType()),
    ]
)
STREAMS: dict[str, dict[str, Any]] = {
    "market": {
        "topic": "market-events",
        "table": "live_market_events",
        "schema": bronze_event_schema,
    },
    "transactions": {
        "topic": "transaction-events",
        "table": "live_customer_transactions",
        "schema": bronze_event_schema,
    },
    "authentication": {
        "topic": "authentication-events",
        "table": "live_authentication_events",
        "schema": bronze_event_schema,
    },
    "fraud_labels": {
        "topic": "fraud-labels",
        "table": "live_fraud_labels",
        "schema": bronze_event_schema,
    },
    "fraud_decisions": {
        "topic": "fraud-decisions",
        "table": "live_fraud_decisions",
        "schema": bronze_event_schema,
    },
}


def quoted_name(name: str) -> str:
    return ".".join(f"`{part}`" for part in name.split("."))


def table_name(short_name: str) -> str:
    return f"{CATALOG}.{BRONZE_SCHEMA}.{short_name}"


def checkpoint_path(short_name: str) -> str:
    return f"{CHECKPOINT_BASE_PATH}/{short_name}/checkpoint"


def secret_connection_string() -> str:
    return dbutils.secrets.get(scope=SECRET_SCOPE, key=SECRET_KEY)


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
        "kafkashaded.org.apache.kafka.common.security.plain.PlainLoginModule required "
        f'username="$ConnectionString" password="{kafka_connection_string}";'
    )
    return {
        "kafka.bootstrap.servers": event_hubs_bootstrap(connection_string),
        "kafka.security.protocol": "SASL_SSL",
        "kafka.sasl.mechanism": "PLAIN",
        "kafka.sasl.jaas.config": jaas,
        "subscribe": topic,
        "startingOffsets": STARTING_OFFSETS,
        "failOnDataLoss": "false",
        "kafka.group.id": CONSUMER_GROUP,
        "maxOffsetsPerTrigger": "10000",
    }


def table_exists(full_name: str) -> bool:
    return spark.catalog.tableExists(full_name)


def count_rows(full_name: str) -> int:
    if not table_exists(full_name):
        return 0
    return int(spark.table(full_name).count())


def path_exists(path: str) -> bool:
    try:
        dbutils.fs.ls(path)
        return True
    except Exception:
        return False


def live_bronze_df(topic: str, schema: T.StructType, connection_string: str) -> DataFrame:
    payload_text = F.col("value").cast("string")
    payload = F.from_json(payload_text, schema)
    raw = spark.readStream.format("kafka").options(**kafka_options(topic, connection_string)).load()
    return parsed_bronze_events(raw, schema)


def parsed_bronze_events(raw: DataFrame, schema: T.StructType) -> DataFrame:
    payload_text = F.col("value").cast("string")
    payload = F.from_json(payload_text, schema)
    parsed = raw.select(
        F.col("topic"),
        F.col("partition").cast("int").alias("kafka_partition"),
        F.col("offset").cast("long").alias("kafka_offset"),
        F.col("timestamp").alias("kafka_timestamp"),
        F.col("timestampType").cast("int").alias("kafka_timestamp_type"),
        F.col("key").cast("string").alias("raw_key"),
        payload_text.alias("raw_payload"),
        payload.alias("payload"),
    )

    return parsed.select(
        F.col("topic"),
        F.lit(CONSUMER_GROUP).alias("consumer_group"),
        F.col("kafka_partition"),
        F.col("kafka_offset"),
        F.col("kafka_timestamp"),
        F.col("kafka_timestamp_type"),
        F.current_timestamp().alias("bronze_ingested_at"),
        F.col("raw_key"),
        F.col("raw_payload"),
        F.sha2(F.col("raw_payload"), 256).alias("raw_payload_sha256"),
        F.col("payload.event_id").isNotNull().alias("parsed_json_valid"),
        F.col("payload.event_id").alias("event_id"),
        F.col("payload.event_type").alias("event_type"),
        F.col("payload.schema_version").alias("schema_version"),
        F.col("payload.source").alias("source"),
        F.to_timestamp("payload.event_timestamp").alias("event_timestamp"),
        F.to_timestamp("payload.source_timestamp").alias("source_timestamp"),
        F.to_timestamp("payload.ingestion_timestamp").alias("source_ingestion_timestamp"),
        F.col("payload.transaction_id").alias("transaction_id"),
        F.col("payload.account_id").alias("account_id"),
        F.col("payload.device_id").alias("device_id"),
        F.col("payload.login_id").alias("login_id"),
        F.col("payload.product_id").alias("product_id"),
        F.col("payload.trade_id").alias("trade_id"),
        F.col("payload.label_source").alias("label_source"),
        F.col("payload.investigation_status").alias("investigation_status"),
        F.col("payload.is_fraud").alias("is_fraud"),
        F.col("payload.decision").alias("decision"),
        F.col("payload.risk_score").alias("risk_score"),
        F.col("payload.model_name").alias("model_name"),
        F.col("payload.model_version").alias("model_version"),
        F.col("payload.threshold_policy_version").alias("threshold_policy_version"),
        F.to_timestamp("payload.prediction_timestamp").alias("prediction_timestamp"),
    )


def run_storage_pass(pass_name: str, connection_string: str) -> list[dict[str, Any]]:
    results = []
    for stream_name, config in STREAMS.items():
        full_name = table_name(config["table"])
        pre_rows = count_rows(full_name)

        query = (
            live_bronze_df(config["topic"], config["schema"], connection_string)
            .writeStream.format("delta")
            .outputMode("append")
            .option("checkpointLocation", checkpoint_path(config["table"]))
            .option("mergeSchema", "true")
            .trigger(availableNow=True)
            .toTable(full_name)
        )
        query.awaitTermination()

        stream_input_rows = sum(int(progress.get("numInputRows") or 0) for progress in query.recentProgress)
        post_rows = count_rows(full_name)
        results.append(
            {
                "pass": pass_name,
                "stream": stream_name,
                "topic": config["topic"],
                "table": full_name,
                "checkpoint_path": checkpoint_path(config["table"]),
                "pre_rows": pre_rows,
                "post_rows": post_rows,
                "new_rows": post_rows - pre_rows,
                "stream_input_rows": stream_input_rows,
                "query_id": str(query.id),
                "run_id": str(query.runId),
            }
        )
    return results


def run_storage_live_window(pass_name: str, connection_string: str) -> list[dict[str, Any]]:
    pre_rows = {}
    for stream_name, config in STREAMS.items():
        full_name = table_name(config["table"])
        pre_rows[stream_name] = count_rows(full_name)

    topics = ",".join(config["topic"] for config in STREAMS.values())
    combined_raw = spark.readStream.format("kafka").options(**kafka_options(topics, connection_string)).load()
    combined_events = parsed_bronze_events(combined_raw, bronze_event_schema)
    topic_progress: dict[str, dict[str, Any]] = {
        config["topic"]: {
            "stream_input_rows": 0,
            "min_offset": None,
            "max_offset": None,
            "sample_transaction_id": None,
            "sample_partition": None,
            "sample_offset": None,
            "first_batch_id": None,
            "last_batch_id": None,
        }
        for config in STREAMS.values()
    }

    def write_batch(batch_df: DataFrame, batch_id: int) -> None:
        for config in STREAMS.values():
            topic_df = batch_df.filter(F.col("topic") == F.lit(config["topic"])).dropDuplicates(
                ["topic", "kafka_partition", "kafka_offset"]
            )
            topic_counts = topic_df.agg(
                F.count("*").alias("row_count"),
                F.min("kafka_offset").alias("min_offset"),
                F.max("kafka_offset").alias("max_offset"),
            ).collect()[0].asDict()
            row_count = int(topic_counts.get("row_count") or 0)
            progress = topic_progress[config["topic"]]
            if row_count > 0:
                sample = (
                    topic_df.select("transaction_id", "kafka_partition", "kafka_offset")
                    .orderBy(F.col("kafka_timestamp").desc(), F.col("kafka_offset").desc())
                    .limit(1)
                    .collect()[0]
                    .asDict()
                )
                progress["stream_input_rows"] += row_count
                progress["min_offset"] = (
                    int(topic_counts["min_offset"])
                    if progress["min_offset"] is None
                    else min(int(progress["min_offset"]), int(topic_counts["min_offset"]))
                )
                progress["max_offset"] = (
                    int(topic_counts["max_offset"])
                    if progress["max_offset"] is None
                    else max(int(progress["max_offset"]), int(topic_counts["max_offset"]))
                )
                progress["sample_transaction_id"] = sample.get("transaction_id")
                progress["sample_partition"] = sample.get("kafka_partition")
                progress["sample_offset"] = sample.get("kafka_offset")
                progress["first_batch_id"] = (
                    int(batch_id) if progress["first_batch_id"] is None else progress["first_batch_id"]
                )
                progress["last_batch_id"] = int(batch_id)
                target_offsets = (
                    spark.table(table_name(config["table"]))
                    .filter(F.col("topic") == F.lit(config["topic"]))
                    .select("topic", "kafka_partition", "kafka_offset")
                    .dropDuplicates()
                )
                rows_to_insert = topic_df.join(
                    target_offsets,
                    on=["topic", "kafka_partition", "kafka_offset"],
                    how="left_anti",
                )
                (
                    rows_to_insert.write.format("delta")
                    .mode("append")
                    .option("mergeSchema", "true")
                    .saveAsTable(table_name(config["table"]))
                )

    query = (
        combined_events.writeStream.foreachBatch(write_batch)
        .option("checkpointLocation", DEMO_LIVE_CHECKPOINT_PATH)
        .trigger(processingTime=PROCESSING_TIME_TRIGGER)
        .start()
    )

    deadline = time.time() + TIMEOUT_SECONDS
    try:
        while time.time() < deadline:
            exception = query.exception()
            if exception is not None:
                raise RuntimeError(f"combined storage stream failed: {exception}")
            time.sleep(5)
    finally:
        if query.isActive:
            query.stop()
        try:
            query.awaitTermination(30)
        except Exception:
            pass

    results = []
    for stream_name, config in STREAMS.items():
        full_name = table_name(config["table"])
        post_rows = count_rows(full_name)
        results.append(
            {
                "pass": pass_name,
                "stream": stream_name,
                "topic": config["topic"],
                "table": full_name,
                "checkpoint_path": DEMO_LIVE_CHECKPOINT_PATH,
                "pre_rows": pre_rows[stream_name],
                "post_rows": post_rows,
                "new_rows": post_rows - pre_rows[stream_name],
                "stream_input_rows": topic_progress[config["topic"]]["stream_input_rows"],
                "min_offset": topic_progress[config["topic"]]["min_offset"],
                "max_offset": topic_progress[config["topic"]]["max_offset"],
                "sample_transaction_id": topic_progress[config["topic"]]["sample_transaction_id"],
                "sample_partition": topic_progress[config["topic"]]["sample_partition"],
                "sample_offset": topic_progress[config["topic"]]["sample_offset"],
                "first_batch_id": topic_progress[config["topic"]]["first_batch_id"],
                "last_batch_id": topic_progress[config["topic"]]["last_batch_id"],
                "query_id": str(query.id),
                "run_id": str(query.runId),
                "stopped": not query.isActive,
            }
        )
    return results


def table_detail(full_name: str) -> dict[str, Any]:
    row = spark.sql(f"DESCRIBE DETAIL {quoted_name(full_name)}").collect()[0].asDict()
    return {
        "format": row.get("format"),
        "location": row.get("location"),
    }


def verify_table(stream_name: str, config: dict[str, Any], first_pass_result: dict[str, Any]) -> dict[str, Any]:
    full_name = table_name(config["table"])
    exists = table_exists(full_name)
    row_count = count_rows(full_name)
    stream_checkpoint_path = first_pass_result.get("checkpoint_path") or checkpoint_path(config["table"])
    result: dict[str, Any] = {
        "stream": stream_name,
        "topic": config["topic"],
        "table": full_name,
        "checkpoint_path": stream_checkpoint_path,
        "table_exists": exists,
        "row_count": row_count,
        "first_pass_new_rows": first_pass_result["new_rows"],
        "checkpoint_exists": path_exists(stream_checkpoint_path),
        "delta_format": False,
        "adls_backed": False,
        "metadata_columns_retained": False,
        "queryable": False,
        "malformed_rows": None,
        "duplicate_kafka_offsets": None,
    }
    if not exists:
        return result

    df = spark.table(full_name)
    metadata_columns = {"topic", "kafka_partition", "kafka_offset", "kafka_timestamp", "raw_payload"}
    result["metadata_columns_retained"] = metadata_columns.issubset(set(df.columns))
    result["queryable"] = True
    result["malformed_rows"] = int(df.filter(~F.col("parsed_json_valid")).count())
    result["duplicate_kafka_offsets"] = int(
        df.groupBy("topic", "kafka_partition", "kafka_offset").count().filter(F.col("count") > 1).count()
    )
    detail = table_detail(full_name)
    location = str(detail.get("location") or "")
    result["delta_format"] = detail.get("format") == "delta"
    result["adls_backed"] = location.startswith("abfss://") or location.startswith("dbfs:/mnt/")
    return result


spark.sql(f"USE CATALOG {quoted_name(CATALOG)}")
schema_rows = spark.sql(f"SHOW SCHEMAS IN {quoted_name(CATALOG)} LIKE '{BRONZE_SCHEMA}'").collect()
if not schema_rows:
    raise RuntimeError(f"Required schema {CATALOG}.{BRONZE_SCHEMA} does not exist.")

connection_string = secret_connection_string()

if TRIGGER_MODE == "availableNow":
    first_pass = run_storage_pass("initial", connection_string)
    resume_pass = run_storage_pass("resume", connection_string) if RUN_RESUME_CHECK else []
else:
    first_pass = run_storage_live_window("bounded_live", connection_string)
    resume_pass = []

first_by_stream = {row["stream"]: row for row in first_pass}
verification = [
    verify_table(stream_name, config, first_by_stream[stream_name])
    for stream_name, config in STREAMS.items()
]

resume_new_rows = sum(row["new_rows"] for row in resume_pass)
resume_input_rows = sum(row["stream_input_rows"] for row in resume_pass)
live_window_new_rows = sum(row["new_rows"] for row in first_pass)
live_window_streams_with_new_rows = all(row["new_rows"] > 0 for row in first_pass)
all_live_queries_stopped = all(row.get("stopped", True) for row in first_pass)
metadata_retained = all(row["metadata_columns_retained"] for row in verification)
tables_queryable = all(row["queryable"] for row in verification)
delta_tables = all(row["delta_format"] for row in verification)
adls_backed = all(row["adls_backed"] for row in verification)
checkpoints_created = all(row["checkpoint_exists"] for row in verification)
no_duplicate_offsets = all((row["duplicate_kafka_offsets"] or 0) == 0 for row in verification)
all_streams_have_rows = all(row["row_count"] > 0 for row in verification)
resume_ok = (
    True
    if not RUN_RESUME_CHECK
    else resume_new_rows == 0 and resume_input_rows == 0 and no_duplicate_offsets
)
live_window_ok = (
    live_window_streams_with_new_rows
    and (TRIGGER_MODE == "availableNow" or all_live_queries_stopped)
)

summary = {
    "phase": "14",
    "status": "PASS"
    if all(
        [
            all_streams_have_rows,
            metadata_retained,
            tables_queryable,
            delta_tables,
            adls_backed,
            checkpoints_created,
            resume_ok,
            live_window_ok,
        ]
    )
    else "FAIL",
    "notebook_path": "/Users/akanaskhan1506@gmail.com/07_live_bronze_storage",
    "consumer_group": CONSUMER_GROUP,
    "checkpoint_base_path": CHECKPOINT_BASE_PATH,
    "trigger_mode": TRIGGER_MODE,
    "timeout_seconds": TIMEOUT_SECONDS,
    "processing_time_trigger": PROCESSING_TIME_TRIGGER,
    "streams": verification,
    "first_pass": first_pass,
    "resume_pass": resume_pass,
    "live_window_new_rows": live_window_new_rows,
    "live_window_streams_with_new_rows": live_window_streams_with_new_rows,
    "all_live_queries_stopped": all_live_queries_stopped,
    "resume_idempotency_status": (
        "SKIPPED" if not RUN_RESUME_CHECK else "PASS" if resume_ok else "FAIL"
    ),
    "kafka_partition_offset_retained": metadata_retained,
    "delta_tables_queryable": tables_queryable,
    "adls_backed_unity_catalog_storage": adls_backed,
    "checkpoints_created": checkpoints_created,
    "secrets_exposed": False,
}

print(json.dumps(summary, indent=2, sort_keys=True))
try:
    dbutils.jobs.taskValues.set(key="summary", value=json.dumps(summary, sort_keys=True, separators=(",", ":")))
except Exception:
    pass

dbutils.notebook.exit(json.dumps(summary, sort_keys=True, separators=(",", ":")))
