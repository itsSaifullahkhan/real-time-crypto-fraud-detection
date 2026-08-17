# Databricks notebook source
# MAGIC %md
# MAGIC # 01 Bronze Ingestion

# COMMAND ----------

import json
import re
import uuid
from typing import Any

from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from pyspark.sql import types as T

# Historical Parquet files were produced with nanosecond timestamps. Spark 18
# needs this compatibility flag to read those fields without changing values.
spark.conf.set("spark.sql.legacy.parquet.nanosAsLong", "true")

CATALOG = "crypto_fraud"
BRONZE_SCHEMA = "bronze"
SOURCE_BASE_PATH = "/Volumes/crypto_fraud/landing/raw_files/historical"
CHECKPOINT_BASE_PATH = "/Volumes/crypto_fraud/monitoring/ingestion_checkpoints/bronze"
INGESTION_RUN_ID = str(uuid.uuid4())

METADATA_COLUMNS = [
    "_ingested_at",
    "_source_file",
    "_source_modification_time",
    "_ingestion_run_id",
    "_rescued_data",
]


def field(name: str, data_type: T.DataType, nullable: bool = True) -> T.StructField:
    return T.StructField(name, data_type, nullable)


DATASETS: dict[str, dict[str, Any]] = {
    "accounts": {
        "expected_rows": 100,
        "primary_key": ["account_id"],
        "schema": T.StructType(
            [
                field("account_id", T.StringType()),
                field("schema_version", T.StringType()),
                field("created_at", T.LongType()),
                field("updated_at", T.LongType()),
                field("home_country", T.StringType()),
                field("kyc_level", T.StringType()),
                field("customer_risk_tier", T.StringType()),
                field("normal_transaction_amount_usd", T.DecimalType(12, 8)),
                field("normal_transaction_frequency_per_day", T.DoubleType()),
                field("preferred_assets", T.ArrayType(T.StringType())),
                field("account_status", T.StringType()),
            ]
        ),
    },
    "devices": {
        "expected_rows": 141,
        "primary_key": ["device_id"],
        "schema": T.StructType(
            [
                field("device_id", T.StringType()),
                field("schema_version", T.StringType()),
                field("first_seen_at", T.LongType()),
                field("last_seen_at", T.LongType()),
                field("device_type", T.StringType()),
                field("operating_system", T.StringType()),
                field("is_trusted", T.BooleanType()),
                field("device_country", T.StringType()),
                field("primary_account_id", T.StringType()),
            ]
        ),
    },
    "wallets": {
        "expected_rows": 342,
        "primary_key": ["wallet_id"],
        "schema": T.StructType(
            [
                field("wallet_id", T.StringType()),
                field("schema_version", T.StringType()),
                field("owner_account_id", T.StringType()),
                field("wallet_type", T.StringType()),
                field("first_seen_at", T.LongType()),
                field("risk_level", T.StringType()),
                field("is_known_destination", T.BooleanType()),
                field("supported_assets", T.ArrayType(T.StringType())),
            ]
        ),
    },
    "authentication_events": {
        "expected_rows": 2500,
        "primary_key": ["event_id"],
        "schema": T.StructType(
            [
                field("event_id", T.StringType()),
                field("event_type", T.StringType()),
                field("schema_version", T.StringType()),
                field("source", T.StringType()),
                field("event_timestamp", T.LongType()),
                field("source_timestamp", T.LongType()),
                field("ingestion_timestamp", T.LongType()),
                field("login_id", T.StringType()),
                field("account_id", T.StringType()),
                field("device_id", T.StringType()),
                field("country", T.StringType()),
                field("ip_address", T.StringType()),
                field("login_success", T.BooleanType()),
                field("mfa_success", T.BooleanType()),
                field("password_reset_flag", T.BooleanType()),
                field("failure_reason", T.StringType()),
                field("event_date", T.DateType()),
            ]
        ),
    },
    "customer_transactions": {
        "expected_rows": 5000,
        "primary_key": ["transaction_id"],
        "schema": T.StructType(
            [
                field("event_id", T.StringType()),
                field("event_type", T.StringType()),
                field("schema_version", T.StringType()),
                field("source", T.StringType()),
                field("event_timestamp", T.LongType()),
                field("source_timestamp", T.LongType()),
                field("ingestion_timestamp", T.LongType()),
                field("transaction_id", T.StringType()),
                field("account_id", T.StringType()),
                field("asset", T.StringType()),
                field("crypto_quantity", T.DecimalType(9, 8)),
                field("transaction_type", T.StringType()),
                field("source_wallet_id", T.StringType()),
                field("destination_wallet_id", T.StringType()),
                field("device_id", T.StringType()),
                field("country", T.StringType()),
                field("market_price_usd", T.DecimalType(13, 8)),
                field("transaction_amount_usd", T.DecimalType(12, 8)),
                field("transaction_status", T.StringType()),
                field("event_date", T.DateType()),
            ]
        ),
    },
    "market_candles": {
        "expected_rows": 20158,
        "primary_key": ["product_id", "candle_start_timestamp"],
        "schema": T.StructType(
            [
                field("schema_version", T.StringType()),
                field("source", T.StringType()),
                field("product_id", T.StringType()),
                field("candle_start_timestamp", T.LongType()),
                field("candle_end_timestamp", T.LongType()),
                field("granularity_seconds", T.LongType()),
                field("open_price_usd", T.DecimalType(13, 8)),
                field("high_price_usd", T.DecimalType(13, 8)),
                field("low_price_usd", T.DecimalType(13, 8)),
                field("close_price_usd", T.DecimalType(13, 8)),
                field("volume", T.DecimalType(12, 8)),
                field("retrieved_at", T.LongType()),
                field("event_date", T.DateType()),
            ]
        ),
    },
    "fraud_labels": {
        "expected_rows": 4479,
        "primary_key": ["event_id"],
        "schema": T.StructType(
            [
                field("event_id", T.StringType()),
                field("event_type", T.StringType()),
                field("schema_version", T.StringType()),
                field("source", T.StringType()),
                field("event_timestamp", T.LongType()),
                field("source_timestamp", T.LongType()),
                field("ingestion_timestamp", T.LongType()),
                field("transaction_id", T.StringType()),
                field("is_fraud", T.BooleanType()),
                field("fraud_type", T.StringType()),
                field("label_timestamp", T.LongType()),
                field("label_source", T.StringType()),
                field("investigation_status", T.StringType()),
                field("label_date", T.DateType()),
            ]
        ),
    },
}


def quoted_name(name: str) -> str:
    return ".".join(f"`{part}`" for part in name.split("."))


def normalize_dbfs_path(path: str) -> str:
    return path[5:] if path.startswith("dbfs:") else path


def source_path(dataset: str) -> str:
    return f"{SOURCE_BASE_PATH}/{dataset}/"


def table_name(dataset: str) -> str:
    return f"{CATALOG}.{BRONZE_SCHEMA}.{dataset}"


def checkpoint_path(dataset: str) -> str:
    return f"{CHECKPOINT_BASE_PATH}/{dataset}/checkpoint/"


def schema_path(dataset: str) -> str:
    return f"{CHECKPOINT_BASE_PATH}/{dataset}/schema/"


def table_exists(full_name: str) -> bool:
    return spark.catalog.tableExists(full_name)


def recursive_list(path: str, max_depth: int = 8) -> list[str]:
    output: list[str] = []

    def visit(current_path: str, depth: int) -> None:
        for entry in dbutils.fs.ls(current_path):
            output.append(entry.path)
            if entry.path.endswith("/") and depth < max_depth:
                visit(entry.path, depth + 1)

    visit(path, 0)
    return output


def inspect_source_layout(dataset: str) -> dict[str, Any]:
    path = source_path(dataset)
    entries = recursive_list(path)
    parquet_files = [entry for entry in entries if entry.endswith(".parquet")]
    partition_columns: set[str] = set()
    normalized_base = normalize_dbfs_path(path).rstrip("/") + "/"

    for entry in entries:
        normalized_entry = normalize_dbfs_path(entry)
        if not normalized_entry.startswith(normalized_base):
            continue
        relative_path = normalized_entry[len(normalized_base):]
        for segment in relative_path.split("/"):
            if re.match(r"^[A-Za-z_][A-Za-z0-9_]*=", segment):
                partition_columns.add(segment.split("=", 1)[0])

    return {
        "source_path": path,
        "parquet_file_count": len(parquet_files),
        "partition_columns": sorted(partition_columns),
        "sample_parquet_files": parquet_files[:5],
    }


def read_source_snapshot(dataset: str, schema: T.StructType) -> DataFrame:
    return spark.read.option("basePath", source_path(dataset)).schema(schema).parquet(source_path(dataset))


def count_rows(df: DataFrame) -> int:
    return int(df.count())


def count_null_primary_keys(df: DataFrame, primary_key: list[str]) -> int:
    condition = None
    for column in primary_key:
        current = F.col(column).isNull()
        condition = current if condition is None else condition | current
    return count_rows(df.filter(condition))


def count_duplicate_primary_keys(df: DataFrame, primary_key: list[str]) -> int:
    return count_rows(df.groupBy(*primary_key).count().filter(F.col("count") > 1))


def rescued_condition(df: DataFrame):
    if "_rescued_data" not in df.columns:
        return F.lit(False)
    rescued = F.col("_rescued_data").cast("string")
    return rescued.isNotNull() & (F.length(F.trim(rescued)) > 0) & (rescued != "{}")


def ingest_dataset(dataset: str, config: dict[str, Any], layout: dict[str, Any]) -> dict[str, Any]:
    path = source_path(dataset)
    partition_columns = layout["partition_columns"]

    reader = (
        spark.readStream.format("cloudFiles")
        .option("cloudFiles.format", "parquet")
        .option("cloudFiles.schemaLocation", schema_path(dataset))
        .option("cloudFiles.schemaEvolutionMode", "rescue")
        .option("rescuedDataColumn", "_rescued_data")
        .option("cloudFiles.includeExistingFiles", "true")
        .schema(config["schema"])
    )
    if partition_columns:
        reader = reader.option("cloudFiles.partitionColumns", ",".join(partition_columns))

    stream_df = reader.load(path)
    if "_rescued_data" not in stream_df.columns:
        stream_df = stream_df.withColumn("_rescued_data", F.lit(None).cast("string"))

    bronze_df = (
        stream_df.withColumn("_ingested_at", F.current_timestamp())
        .withColumn("_source_file", F.col("_metadata.file_path"))
        .withColumn("_source_modification_time", F.col("_metadata.file_modification_time"))
        .withColumn("_ingestion_run_id", F.lit(INGESTION_RUN_ID))
    )

    source_columns = config["schema"].fieldNames()
    bronze_df = bronze_df.select(*source_columns, *METADATA_COLUMNS)

    query = (
        bronze_df.writeStream.format("delta")
        .outputMode("append")
        .option("checkpointLocation", checkpoint_path(dataset))
        .trigger(availableNow=True)
        .toTable(table_name(dataset))
    )
    query.awaitTermination()

    progress_rows = 0
    for progress in query.recentProgress:
        progress_rows += int(progress.get("numInputRows") or 0)

    return {
        "dataset": dataset,
        "stream_input_rows": progress_rows,
        "query_id": str(query.id),
        "run_id": str(query.runId),
    }


def verify_dataset(
    dataset: str,
    config: dict[str, Any],
    layout: dict[str, Any],
    pre_count: int,
    ingest_result: dict[str, Any] | None,
) -> dict[str, Any]:
    full_table_name = table_name(dataset)
    exists = table_exists(full_table_name)
    result: dict[str, Any] = {
        "dataset": dataset,
        "source_path": source_path(dataset),
        "bronze_table": full_table_name,
        "checkpoint_path": checkpoint_path(dataset),
        "schema_path": schema_path(dataset),
        "expected_rows": config["expected_rows"],
        "pre_run_rows": pre_count,
        "table_exists": exists,
        "parquet_file_count": layout["parquet_file_count"],
        "partition_columns": layout["partition_columns"],
        "stream_input_rows": ingest_result["stream_input_rows"] if ingest_result else 0,
    }

    if not exists:
        result.update(
            {
                "actual_rows": 0,
                "new_rows": 0,
                "null_primary_ids": None,
                "duplicate_primary_ids": None,
                "rescued_records": None,
                "metadata_null_records": None,
                "missing_source_columns": config["schema"].fieldNames(),
                "unexpected_business_columns": [],
                "type_mismatches": [],
                "status": "FAIL",
            }
        )
        return result

    bronze_df = spark.table(full_table_name)
    source_df = read_source_snapshot(dataset, config["schema"])
    source_columns = source_df.columns
    bronze_columns = bronze_df.columns
    source_types = dict(source_df.dtypes)
    bronze_types = dict(bronze_df.dtypes)

    rescued_rows = count_rows(bronze_df.filter(rescued_condition(bronze_df)))
    metadata_null_rows = count_rows(
        bronze_df.filter(
            F.col("_ingested_at").isNull()
            | F.col("_source_file").isNull()
            | (F.length(F.trim(F.col("_source_file"))) == 0)
            | F.col("_source_modification_time").isNull()
            | F.col("_ingestion_run_id").isNull()
            | (F.length(F.trim(F.col("_ingestion_run_id"))) == 0)
        )
    )
    missing_source_columns = [column for column in source_columns if column not in bronze_columns]
    unexpected_business_columns = [
        column for column in bronze_columns if column not in source_columns and column not in METADATA_COLUMNS
    ]
    type_mismatches = [
        {"column": column, "source_type": source_types[column], "bronze_type": bronze_types.get(column)}
        for column in source_columns
        if column in bronze_types and source_types[column] != bronze_types[column]
    ]
    missing_partition_columns = [column for column in layout["partition_columns"] if column not in bronze_columns]
    actual_rows = count_rows(bronze_df)
    null_primary_ids = count_null_primary_keys(bronze_df, config["primary_key"])
    duplicate_primary_ids = count_duplicate_primary_keys(bronze_df, config["primary_key"])

    checks_failed = []
    if layout["parquet_file_count"] == 0:
        checks_failed.append("source_parquet_files_missing")
    if actual_rows != config["expected_rows"]:
        checks_failed.append("row_count_mismatch")
    if null_primary_ids:
        checks_failed.append("null_primary_ids")
    if duplicate_primary_ids:
        checks_failed.append("duplicate_primary_ids")
    if rescued_rows:
        checks_failed.append("rescued_records")
    if metadata_null_rows:
        checks_failed.append("metadata_null_records")
    if missing_source_columns:
        checks_failed.append("missing_source_columns")
    if unexpected_business_columns:
        checks_failed.append("unexpected_business_columns")
    if type_mismatches:
        checks_failed.append("type_mismatches")
    if missing_partition_columns:
        checks_failed.append("missing_partition_columns")

    rescued_source_files = []
    if rescued_rows:
        rescued_source_files = [
            row["_source_file"]
            for row in bronze_df.filter(rescued_condition(bronze_df))
            .select("_source_file")
            .distinct()
            .limit(25)
            .collect()
        ]

    result.update(
        {
            "actual_rows": actual_rows,
            "new_rows": actual_rows - pre_count,
            "null_primary_ids": null_primary_ids,
            "duplicate_primary_ids": duplicate_primary_ids,
            "rescued_records": rescued_rows,
            "rescued_source_files": rescued_source_files,
            "metadata_null_records": metadata_null_rows,
            "missing_source_columns": missing_source_columns,
            "unexpected_business_columns": unexpected_business_columns,
            "type_mismatches": type_mismatches,
            "missing_partition_columns": missing_partition_columns,
            "partition_columns_preserved": len(missing_partition_columns) == 0,
            "checks_failed": checks_failed,
            "status": "PASS" if not checks_failed else "FAIL",
        }
    )
    return result


spark.sql(f"USE CATALOG {quoted_name(CATALOG)}")

schema_rows = spark.sql(f"SHOW SCHEMAS IN {quoted_name(CATALOG)} LIKE '{BRONZE_SCHEMA}'").collect()
if not schema_rows:
    raise RuntimeError(f"Required schema {CATALOG}.{BRONZE_SCHEMA} does not exist.")

source_layouts = {dataset: inspect_source_layout(dataset) for dataset in DATASETS}
source_layout_failures = [
    {"dataset": dataset, "source_path": layout["source_path"], "reason": "no_parquet_files"}
    for dataset, layout in source_layouts.items()
    if layout["parquet_file_count"] == 0
]
if source_layout_failures:
    raise RuntimeError(json.dumps({"source_layout_failures": source_layout_failures}, sort_keys=True))

pre_counts = {
    dataset: count_rows(spark.table(table_name(dataset))) if table_exists(table_name(dataset)) else 0
    for dataset in DATASETS
}

ingest_results = {}
for dataset, config in DATASETS.items():
    ingest_results[dataset] = ingest_dataset(dataset, config, source_layouts[dataset])

validation_summary = [
    verify_dataset(dataset, config, source_layouts[dataset], pre_counts[dataset], ingest_results[dataset])
    for dataset, config in DATASETS.items()
]

failed_checks = [
    {
        "dataset": row["dataset"],
        "checks_failed": row.get("checks_failed", []),
        "rescued_records": row.get("rescued_records"),
        "rescued_source_files": row.get("rescued_source_files", []),
    }
    for row in validation_summary
    if row["status"] != "PASS"
]

output = {
    "notebook": "01_bronze_ingestion",
    "ingestion_run_id": INGESTION_RUN_ID,
    "source_base_path": SOURCE_BASE_PATH,
    "checkpoint_base_path": CHECKPOINT_BASE_PATH,
    "summary": validation_summary,
    "failed_checks": failed_checks,
    "overall_status": "PASS" if not failed_checks else "FAIL",
}

print(json.dumps(output, indent=2, sort_keys=True))
dbutils.notebook.exit(json.dumps(output, sort_keys=True))
