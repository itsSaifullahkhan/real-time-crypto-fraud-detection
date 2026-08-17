# Databricks notebook source
# MAGIC %md
# MAGIC # 00 Validate Landing Data
# MAGIC
# MAGIC Read-only validation for the Crypto Fraud Detection Platform historical landing data.
# MAGIC This notebook does not create, update, or delete raw files, Unity Catalog objects, or Bronze/Silver tables.

# COMMAND ----------

from __future__ import annotations

import json
from functools import reduce
from typing import Any

from pyspark.sql import DataFrame
from pyspark.sql import functions as F


CATALOG = "crypto_fraud"
LANDING_SCHEMA = "landing"
RAW_VOLUME = "raw_files"
BASE_PATH = f"/Volumes/{CATALOG}/{LANDING_SCHEMA}/{RAW_VOLUME}/historical"

EXPECTED_COUNTS = {
    "accounts": 100,
    "devices": 141,
    "wallets": 342,
    "authentication_events": 2500,
    "customer_transactions": 5000,
    "market_candles": 20158,
    "fraud_labels": 4479,
}

APPROVED_TRANSACTION_TYPES = ["DEPOSIT", "WITHDRAWAL", "TRANSFER"]
TRANSACTION_LEAKAGE_COLUMNS = {
    "is_fraud",
    "fraud_type",
    "scenario_id",
    "investigation_status",
    "label_status",
}

# Historical Parquet files were produced by PyArrow with nanosecond timestamps.
# Spark 18 rejects TIMESTAMP(NANOS,true) unless this compatibility flag is set.
spark.conf.set("spark.sql.legacy.parquet.nanosAsLong", "true")

DATASETS = {
    "accounts": {
        "path": f"{BASE_PATH}/accounts",
        "primary_key": ["account_id"],
        "required_columns": [
            "account_id",
            "schema_version",
            "created_at",
            "updated_at",
            "home_country",
            "kyc_level",
            "customer_risk_tier",
            "normal_transaction_amount_usd",
            "normal_transaction_frequency_per_day",
            "preferred_assets",
            "account_status",
        ],
        "timestamp_columns": ["created_at", "updated_at"],
    },
    "devices": {
        "path": f"{BASE_PATH}/devices",
        "primary_key": ["device_id"],
        "required_columns": [
            "device_id",
            "schema_version",
            "first_seen_at",
            "last_seen_at",
            "device_type",
            "operating_system",
            "is_trusted",
            "device_country",
            "primary_account_id",
        ],
        "timestamp_columns": ["first_seen_at", "last_seen_at"],
    },
    "wallets": {
        "path": f"{BASE_PATH}/wallets",
        "primary_key": ["wallet_id"],
        "required_columns": [
            "wallet_id",
            "schema_version",
            "owner_account_id",
            "wallet_type",
            "first_seen_at",
            "risk_level",
            "is_known_destination",
            "supported_assets",
        ],
        "timestamp_columns": ["first_seen_at"],
    },
    "authentication_events": {
        "path": f"{BASE_PATH}/authentication_events",
        "primary_key": ["event_id"],
        "required_columns": [
            "event_id",
            "event_type",
            "schema_version",
            "source",
            "event_timestamp",
            "source_timestamp",
            "ingestion_timestamp",
            "login_id",
            "account_id",
            "device_id",
            "country",
            "ip_address",
            "login_success",
            "mfa_success",
            "password_reset_flag",
            "failure_reason",
        ],
        "timestamp_columns": [
            "event_timestamp",
            "source_timestamp",
            "ingestion_timestamp",
        ],
    },
    "customer_transactions": {
        "path": f"{BASE_PATH}/customer_transactions",
        "primary_key": ["transaction_id"],
        "required_columns": [
            "event_id",
            "event_type",
            "schema_version",
            "source",
            "event_timestamp",
            "source_timestamp",
            "ingestion_timestamp",
            "transaction_id",
            "account_id",
            "asset",
            "crypto_quantity",
            "transaction_type",
            "source_wallet_id",
            "destination_wallet_id",
            "device_id",
            "country",
            "market_price_usd",
            "transaction_amount_usd",
            "transaction_status",
        ],
        "timestamp_columns": [
            "event_timestamp",
            "source_timestamp",
            "ingestion_timestamp",
        ],
    },
    "market_candles": {
        "path": f"{BASE_PATH}/market_candles",
        "primary_key": ["product_id", "candle_start_timestamp"],
        "required_columns": [
            "schema_version",
            "source",
            "product_id",
            "candle_start_timestamp",
            "candle_end_timestamp",
            "granularity_seconds",
            "open_price_usd",
            "high_price_usd",
            "low_price_usd",
            "close_price_usd",
            "volume",
            "retrieved_at",
        ],
        "timestamp_columns": [
            "candle_start_timestamp",
            "candle_end_timestamp",
            "retrieved_at",
        ],
    },
    "fraud_labels": {
        "path": f"{BASE_PATH}/fraud_labels",
        "primary_key": ["event_id"],
        "required_columns": [
            "event_id",
            "event_type",
            "schema_version",
            "source",
            "event_timestamp",
            "source_timestamp",
            "ingestion_timestamp",
            "transaction_id",
            "is_fraud",
            "fraud_type",
            "label_timestamp",
            "label_source",
            "investigation_status",
        ],
        "timestamp_columns": [
            "event_timestamp",
            "source_timestamp",
            "ingestion_timestamp",
            "label_timestamp",
        ],
    },
}


# COMMAND ----------

def safe_ls(path: str) -> list[Any]:
    return dbutils.fs.ls(path)


def list_files_recursive(path: str) -> list[str]:
    files: list[str] = []
    for entry in safe_ls(path):
        if entry.path.endswith("/"):
            files.extend(list_files_recursive(entry.path))
        else:
            files.append(entry.path)
    return files


def source_path_profile(path: str) -> dict[str, Any]:
    try:
        files = list_files_recursive(path)
    except Exception as exc:
        return {
            "path_exists": False,
            "parquet_file_count": 0,
            "path_error": str(exc),
        }

    parquet_files = [file for file in files if file.lower().endswith(".parquet")]
    return {
        "path_exists": True,
        "parquet_file_count": len(parquet_files),
        "path_error": "",
    }


def read_parquet_dataset(path: str) -> DataFrame:
    return spark.read.option("basePath", path).parquet(path)


def missing_columns(df: DataFrame, required_columns: list[str]) -> list[str]:
    present = set(df.columns)
    return [column for column in required_columns if column not in present]


def primary_key_null_condition(primary_key: list[str]):
    return reduce(lambda left, right: left | right, [F.col(column).isNull() for column in primary_key])


def null_primary_id_count(df: DataFrame, primary_key: list[str]) -> int:
    if any(column not in df.columns for column in primary_key):
        return df.count()
    return int(df.filter(primary_key_null_condition(primary_key)).count())


def duplicate_primary_id_count(df: DataFrame, primary_key: list[str]) -> int:
    if any(column not in df.columns for column in primary_key):
        return 0

    non_null_df = df.filter(~primary_key_null_condition(primary_key))
    duplicate_rows = (
        non_null_df.groupBy(*primary_key)
        .count()
        .filter(F.col("count") > 1)
        .select(F.coalesce(F.sum(F.col("count") - F.lit(1)), F.lit(0)).alias("duplicate_count"))
        .collect()[0]["duplicate_count"]
    )
    return int(duplicate_rows or 0)


def quote_identifier(column: str) -> str:
    return f"`{column.replace('`', '``')}`"


def invalid_timestamp_count(df: DataFrame, column: str) -> int:
    if column not in df.columns:
        return 0

    dtype = dict(df.dtypes).get(column, "")
    column_ref = F.col(column)
    if dtype in {"bigint", "long"}:
        return int(df.filter(column_ref.isNull() | (column_ref <= 0)).count())
    if dtype.startswith("timestamp") or dtype == "date":
        return int(df.filter(column_ref.isNull()).count())

    parsed = F.expr(f"try_to_timestamp({quote_identifier(column)})")
    return int(df.filter(column_ref.isNull() | parsed.isNull()).count())


def anti_join_count(left: DataFrame, left_column: str, right: DataFrame, right_column: str) -> int:
    if left_column not in left.columns or right_column not in right.columns:
        return left.count()

    return int(
        left.select(F.col(left_column).alias("_left_key"))
        .filter(F.col("_left_key").isNotNull())
        .join(
            right.select(F.col(right_column).alias("_right_key")).dropDuplicates(),
            F.col("_left_key") == F.col("_right_key"),
            "left_anti",
        )
        .count()
    )


def count_where(df: DataFrame, condition) -> int:
    return int(df.filter(condition).count())


def timestamp_sort_value(df: DataFrame, column: str):
    dtype = dict(df.dtypes).get(column, "")
    quoted = quote_identifier(column)
    if dtype in {"bigint", "long"}:
        return F.col(column).cast("decimal(38,0)")
    if dtype.startswith("timestamp") or dtype == "date":
        return F.expr(f"unix_micros({quoted})").cast("decimal(38,0)") * F.lit(1000).cast("decimal(38,0)")
    return F.expr(f"unix_micros(try_to_timestamp({quoted}))").cast("decimal(38,0)") * F.lit(1000).cast("decimal(38,0)")


def add_detail(details: list[dict[str, Any]], dataset: str, category: str, check: str, failures: int, note: str = "") -> None:
    if failures:
        details.append(
            {
                "dataset": dataset,
                "category": category,
                "check": check,
                "failures": int(failures),
                "note": note,
            }
        )


def read_json_file(path: str) -> dict[str, Any]:
    raw = dbutils.fs.head(path, 1024 * 1024)
    return json.loads(raw)


def print_metadata_summary(name: str, payload: dict[str, Any]) -> None:
    print(f"{name} keys: {sorted(payload.keys())}")
    if "status" in payload:
        print(f"{name} status: {payload['status']}")
    if "record_counts" in payload:
        print(f"{name} record_counts: {json.dumps(payload['record_counts'], sort_keys=True)}")
    if "fraud_distribution" in payload:
        print(
            f"{name} fraud_distribution: "
            f"{json.dumps(payload['fraud_distribution'], sort_keys=True)}"
        )


# COMMAND ----------

print(f"Validating landing data under: {BASE_PATH}")
print("This notebook is read-only and does not write to raw, Bronze, or Silver storage.")

source_profiles: dict[str, dict[str, Any]] = {}
datasets: dict[str, DataFrame] = {}

for dataset, config in DATASETS.items():
    path = config["path"]
    print(f"\n=== Dataset: {dataset} ===")
    print(f"Source path: {path}")

    profile = source_path_profile(path)
    source_profiles[dataset] = profile
    print(f"Path exists: {profile['path_exists']}")
    print(f"Parquet files found: {profile['parquet_file_count']}")
    if profile["path_error"]:
        print(f"Path error: {profile['path_error']}")

    if not profile["path_exists"] or profile["parquet_file_count"] == 0:
        continue

    df = read_parquet_dataset(path)
    datasets[dataset] = df.cache()
    print(f"Column count: {len(df.columns)}")
    print("Schema:")
    df.printSchema()
    print("Five sample rows:")
    display(df.limit(5))


# COMMAND ----------

metadata_paths = {
    "generation_manifest": f"{BASE_PATH}/_metadata/generation_manifest.json",
    "quality_report": f"{BASE_PATH}/_metadata/quality_report.json",
}

metadata_payloads: dict[str, dict[str, Any]] = {}
for name, path in metadata_paths.items():
    print(f"\n=== Metadata: {name} ===")
    print(f"Path: {path}")
    try:
        payload = read_json_file(path)
        metadata_payloads[name] = payload
        print_metadata_summary(name, payload)
    except Exception as exc:
        print(f"Metadata read failed for {name}: {exc}")


# COMMAND ----------

summary_rows: list[dict[str, Any]] = []
failure_details: list[dict[str, Any]] = []

for dataset, config in DATASETS.items():
    profile = source_profiles.get(
        dataset,
        {"path_exists": False, "parquet_file_count": 0, "path_error": "not checked"},
    )
    df = datasets.get(dataset)
    expected_rows = EXPECTED_COUNTS[dataset]

    actual_rows = 0
    column_count = 0
    missing_required = config["required_columns"]
    null_primary_ids = 0
    duplicate_primary_ids = 0
    business_rule_failures = 0

    source_failures = 0
    if not profile["path_exists"]:
        source_failures += 1
        add_detail(failure_details, dataset, "source", "source_path_exists", 1, profile["path_error"])
    if profile["parquet_file_count"] == 0:
        source_failures += 1
        add_detail(failure_details, dataset, "source", "source_path_contains_parquet", 1)

    if df is not None:
        actual_rows = int(df.count())
        column_count = len(df.columns)
        missing_required = missing_columns(df, config["required_columns"])
        null_primary_ids = null_primary_id_count(df, config["primary_key"])
        duplicate_primary_ids = duplicate_primary_id_count(df, config["primary_key"])

        row_count_failures = 0 if actual_rows == expected_rows else abs(actual_rows - expected_rows)
        add_detail(
            failure_details,
            dataset,
            "row_count",
            "actual_rows_match_expected_rows",
            row_count_failures,
            f"expected={expected_rows}; actual={actual_rows}",
        )
        add_detail(
            failure_details,
            dataset,
            "required_columns",
            "required_columns_present",
            len(missing_required),
            ",".join(missing_required),
        )
        add_detail(
            failure_details,
            dataset,
            "primary_key",
            "primary_key_not_null",
            null_primary_ids,
            ",".join(config["primary_key"]),
        )
        add_detail(
            failure_details,
            dataset,
            "primary_key",
            "primary_key_unique",
            duplicate_primary_ids,
            ",".join(config["primary_key"]),
        )

        timestamp_failures = sum(
            invalid_timestamp_count(df, column) for column in config["timestamp_columns"]
        )
        business_rule_failures += timestamp_failures
        add_detail(
            failure_details,
            dataset,
            "business_rule",
            "timestamp_columns_valid",
            timestamp_failures,
            ",".join(config["timestamp_columns"]),
        )

    summary_rows.append(
        {
            "dataset": dataset,
            "expected_rows": expected_rows,
            "actual_rows": actual_rows,
            "column_count": column_count,
            "parquet_file_count": int(profile["parquet_file_count"]),
            "source_path_failures": source_failures,
            "missing_required_columns": ",".join(missing_required),
            "null_primary_ids": null_primary_ids,
            "duplicate_primary_ids": duplicate_primary_ids,
            "foreign_key_failures": 0,
            "business_rule_failures": business_rule_failures,
            "status": "PASS",
        }
    )


summary_by_dataset = {row["dataset"]: row for row in summary_rows}


# COMMAND ----------

accounts = datasets.get("accounts")
devices = datasets.get("devices")
wallets = datasets.get("wallets")
authentication_events = datasets.get("authentication_events")
customer_transactions = datasets.get("customer_transactions")
market_candles = datasets.get("market_candles")
fraud_labels = datasets.get("fraud_labels")

if customer_transactions is not None and accounts is not None:
    failures = anti_join_count(customer_transactions, "account_id", accounts, "account_id")
    summary_by_dataset["customer_transactions"]["foreign_key_failures"] += failures
    add_detail(
        failure_details,
        "customer_transactions",
        "foreign_key",
        "transactions.account_id_exists_in_accounts",
        failures,
    )

if customer_transactions is not None and devices is not None:
    failures = anti_join_count(customer_transactions, "device_id", devices, "device_id")
    summary_by_dataset["customer_transactions"]["foreign_key_failures"] += failures
    add_detail(
        failure_details,
        "customer_transactions",
        "foreign_key",
        "transactions.device_id_exists_in_devices",
        failures,
    )

if authentication_events is not None and accounts is not None:
    failures = anti_join_count(authentication_events, "account_id", accounts, "account_id")
    summary_by_dataset["authentication_events"]["foreign_key_failures"] += failures
    add_detail(
        failure_details,
        "authentication_events",
        "foreign_key",
        "authentication_events.account_id_exists_in_accounts",
        failures,
    )

if authentication_events is not None and devices is not None:
    failures = anti_join_count(authentication_events, "device_id", devices, "device_id")
    summary_by_dataset["authentication_events"]["foreign_key_failures"] += failures
    add_detail(
        failure_details,
        "authentication_events",
        "foreign_key",
        "authentication_events.device_id_exists_in_devices",
        failures,
    )

if fraud_labels is not None and customer_transactions is not None:
    failures = anti_join_count(fraud_labels, "transaction_id", customer_transactions, "transaction_id")
    summary_by_dataset["fraud_labels"]["foreign_key_failures"] += failures
    add_detail(
        failure_details,
        "fraud_labels",
        "foreign_key",
        "fraud_labels.transaction_id_exists_in_customer_transactions",
        failures,
    )

if customer_transactions is not None and wallets is not None:
    source_failures = anti_join_count(
        customer_transactions.filter(F.col("source_wallet_id").isNotNull()),
        "source_wallet_id",
        wallets,
        "wallet_id",
    )
    destination_failures = anti_join_count(
        customer_transactions.filter(F.col("destination_wallet_id").isNotNull()),
        "destination_wallet_id",
        wallets,
        "wallet_id",
    )
    wallet_failures = source_failures + destination_failures
    summary_by_dataset["customer_transactions"]["foreign_key_failures"] += wallet_failures
    add_detail(
        failure_details,
        "customer_transactions",
        "foreign_key",
        "transactions.non_null_source_wallet_id_exists_in_wallets",
        source_failures,
    )
    add_detail(
        failure_details,
        "customer_transactions",
        "foreign_key",
        "transactions.non_null_destination_wallet_id_exists_in_wallets",
        destination_failures,
    )


# COMMAND ----------

if customer_transactions is not None:
    checks = {
        "crypto_quantity_positive": count_where(customer_transactions, F.col("crypto_quantity") <= 0),
        "transaction_amount_usd_non_negative": count_where(
            customer_transactions,
            F.col("transaction_amount_usd") < 0,
        ),
        "market_price_usd_positive": count_where(customer_transactions, F.col("market_price_usd") <= 0),
        "transaction_type_approved": count_where(
            customer_transactions,
            ~F.col("transaction_type").isin(APPROVED_TRANSACTION_TYPES),
        ),
        "transaction_records_have_no_label_leakage_columns": len(
            TRANSACTION_LEAKAGE_COLUMNS.intersection(customer_transactions.columns)
        ),
    }
    for check_name, failures in checks.items():
        summary_by_dataset["customer_transactions"]["business_rule_failures"] += failures
        add_detail(failure_details, "customer_transactions", "business_rule", check_name, failures)

if market_candles is not None:
    market_price_condition = (
        (F.col("open_price_usd") <= 0)
        | (F.col("high_price_usd") <= 0)
        | (F.col("low_price_usd") <= 0)
        | (F.col("close_price_usd") <= 0)
    )
    checks = {
        "coinbase_candle_prices_positive": count_where(market_candles, market_price_condition),
        "coinbase_candle_volume_non_negative": count_where(market_candles, F.col("volume") < 0),
        "coinbase_candle_granularity_is_60_seconds": count_where(
            market_candles,
            F.col("granularity_seconds") != 60,
        ),
    }
    for check_name, failures in checks.items():
        summary_by_dataset["market_candles"]["business_rule_failures"] += failures
        add_detail(failure_details, "market_candles", "business_rule", check_name, failures)

if fraud_labels is not None and customer_transactions is not None:
    joined_labels = fraud_labels.select(
        "transaction_id",
        timestamp_sort_value(fraud_labels, "event_timestamp").alias("label_event_timestamp"),
        timestamp_sort_value(fraud_labels, "label_timestamp").alias("label_timestamp"),
    ).join(
        customer_transactions.select(
            "transaction_id",
            timestamp_sort_value(customer_transactions, "event_timestamp").alias("transaction_event_timestamp"),
        ),
        "transaction_id",
        "left",
    )
    label_timing_failures = count_where(
        joined_labels,
        F.col("transaction_event_timestamp").isNull()
        | (F.col("label_event_timestamp") < F.col("transaction_event_timestamp"))
        | (F.col("label_timestamp") < F.col("transaction_event_timestamp")),
    )
    summary_by_dataset["fraud_labels"]["business_rule_failures"] += label_timing_failures
    add_detail(
        failure_details,
        "fraud_labels",
        "business_rule",
        "fraud_labels_occur_at_or_after_transaction_timestamps",
        label_timing_failures,
    )


audit_match_path = f"{BASE_PATH}/_audit/market_enrichment_matches.parquet"
print(f"\n=== Audit: market enrichment matches ===")
print(f"Path: {audit_match_path}")
try:
    market_matches = spark.read.parquet(audit_match_path)
    print(f"Rows: {market_matches.count()}")
    market_matches.printSchema()
    display(market_matches.limit(5))
    future_candle_failures = count_where(
        market_matches,
        timestamp_sort_value(market_matches, "matched_market_candle_end_timestamp") > timestamp_sort_value(market_matches, "transaction_event_timestamp"),
    )
except Exception as exc:
    market_matches = None
    future_candle_failures = 1
    print(f"Market enrichment audit read failed: {exc}")

summary_by_dataset["customer_transactions"]["business_rule_failures"] += future_candle_failures
add_detail(
    failure_details,
    "customer_transactions",
    "business_rule",
    "no_future_coinbase_candle_used_for_transaction_enrichment",
    future_candle_failures,
)


# COMMAND ----------

for row in summary_rows:
    total_failures = (
        row["source_path_failures"]
        + (0 if row["actual_rows"] == row["expected_rows"] else 1)
        + (0 if row["missing_required_columns"] == "" else 1)
        + row["null_primary_ids"]
        + row["duplicate_primary_ids"]
        + row["foreign_key_failures"]
        + row["business_rule_failures"]
    )
    row["status"] = "PASS" if total_failures == 0 else "FAIL"

summary_df = spark.createDataFrame(summary_rows).select(
    "dataset",
    "expected_rows",
    "actual_rows",
    "column_count",
    "parquet_file_count",
    "source_path_failures",
    "missing_required_columns",
    "null_primary_ids",
    "duplicate_primary_ids",
    "foreign_key_failures",
    "business_rule_failures",
    "status",
)

failure_details_df = spark.createDataFrame(failure_details) if failure_details else None

print("\n=== Final validation summary ===")
display(summary_df.orderBy("dataset"))
print(json.dumps(summary_rows, indent=2, sort_keys=True))

if failure_details_df is not None:
    print("\n=== Failed checks ===")
    display(failure_details_df.orderBy("dataset", "category", "check"))
    print(json.dumps(failure_details, indent=2, sort_keys=True))
else:
    print("\n=== Failed checks ===")
    print("None")

overall_status = "PASS" if all(row["status"] == "PASS" for row in summary_rows) else "FAIL"
print(f"\nOVERALL_STATUS {overall_status}")

metadata_summary = {
    name: {
        "status": payload.get("status"),
        "record_counts": payload.get("record_counts"),
        "keys": sorted(payload.keys()),
    }
    for name, payload in metadata_payloads.items()
}

validation_result = {
    "overall_status": overall_status,
    "summary": summary_rows,
    "failed_checks": failure_details,
    "metadata": metadata_summary,
}

print("\nVALIDATION_RESULT_JSON")
print(json.dumps(validation_result, indent=2, sort_keys=True))

for df in datasets.values():
    df.unpersist()

# Return structured output to Databricks Jobs without writing to raw, Bronze, or Silver storage.
dbutils.notebook.exit(json.dumps(validation_result, sort_keys=True))


