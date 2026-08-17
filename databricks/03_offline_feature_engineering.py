# Databricks notebook source
# MAGIC %md
# MAGIC # 03 Offline Feature Engineering
# MAGIC
# MAGIC Point-in-time-correct offline feature engineering for Silver customer transactions.
# MAGIC This notebook writes feature and labeled-training Delta tables only. It does not train,
# MAGIC evaluate, register, or serve any machine-learning model.

# COMMAND ----------

from __future__ import annotations

import json
import uuid
from datetime import timedelta
from typing import Any

from delta.tables import DeltaTable
from pyspark.sql import DataFrame, Window
from pyspark.sql import functions as F

spark.conf.set("spark.sql.session.timeZone", "UTC")
spark.conf.set("spark.sql.legacy.parquet.nanosAsLong", "true")

CATALOG = "crypto_fraud"
SOURCE_SCHEMA = "silver"
FEATURE_SCHEMA = "features"
FEATURE_TABLE_NAME = "transaction_features_offline"
TRAINING_TABLE_NAME = "transaction_training_dataset"
FEATURE_VERSION = "v1"
CLUSTER_ID = "0803-061312-78fw66xn"
RUN_ID = str(uuid.uuid4())

EXPECTED_SOURCE_COUNTS = {
    "accounts": 100,
    "devices": 141,
    "wallets": 342,
    "authentication_events": 2500,
    "customer_transactions": 5000,
    "market_candles": 20158,
    "fraud_labels": 4479,
}

ASSET_PRODUCT_MAPPING = {"BTC": "BTC-USD", "ETH": "ETH-USD"}
NIGHT_START_HOUR_UTC = 0
NIGHT_END_HOUR_UTC = 5

FORBIDDEN_FEATURE_TABLE_COLUMNS = {
    "is_fraud", "target_is_fraud", "fraud_type", "fraud_label", "confirmed_fraud",
    "label_timestamp", "label_status", "label_source", "investigation_status",
    "investigation_result", "scenario_id", "scenario_execution_id", "fraud_scenario",
}

IDENTITY_AND_TIMING_COLUMNS = [
    "transaction_id", "account_id", "device_id", "source_wallet_id", "destination_wallet_id",
    "feature_timestamp", "event_date", "asset", "transaction_type", "country",
]
CURRENT_TRANSACTION_FEATURES = [
    "transaction_amount_usd", "crypto_quantity", "market_price_usd", "transaction_hour_utc",
    "transaction_day_of_week_utc", "is_weekend_utc", "is_night_transaction_utc",
]
ACCOUNT_FEATURES = [
    "account_age_days", "account_profile_available", "customer_risk_tier",
    "normal_transaction_amount_usd", "normal_transaction_amount_available",
    "amount_to_normal_ratio", "amount_above_normal_usd", "transaction_country_mismatch_home_country",
]
TRANSACTION_HISTORY_FEATURES = [
    "prior_tx_count_5m", "prior_tx_count_1h", "prior_tx_count_24h",
    "prior_tx_amount_sum_1h", "prior_tx_amount_sum_24h", "prior_tx_amount_avg_24h",
    "prior_tx_amount_max_24h", "has_prior_tx_24h", "seconds_since_previous_tx",
    "has_previous_transaction",
]
AUTHENTICATION_FEATURES = [
    "failed_auth_count_10m", "failed_auth_count_1h", "successful_auth_count_1h",
    "recent_auth_failure_flag_10m", "seconds_since_last_successful_auth",
    "has_previous_successful_auth",
]
DEVICE_AUTH_FEATURES = [
    "device_failed_auth_count_10m", "device_successful_auth_count_1h",
    "device_distinct_account_count_1h",
]
DEVICE_FEATURES = ["device_profile_available", "device_age_hours", "device_age_days", "is_new_device_24h"]
WALLET_FEATURES = [
    "source_wallet_applicable", "destination_wallet_applicable",
    "destination_wallet_first_seen_available", "destination_wallet_age_hours",
    "is_new_destination_wallet", "prior_destination_wallet_tx_count",
    "prior_source_wallet_tx_count",
]
MARKET_FEATURES = [
    "market_product_id", "latest_market_candle_end_timestamp", "market_data_freshness_seconds",
    "latest_market_close_usd", "market_return_5m", "market_return_1h",
    "market_volatility_1h", "market_volume_sum_1h", "market_data_available",
]

FEATURE_VALUE_COLUMNS = (
    IDENTITY_AND_TIMING_COLUMNS + CURRENT_TRANSACTION_FEATURES + ACCOUNT_FEATURES
    + TRANSACTION_HISTORY_FEATURES + AUTHENTICATION_FEATURES + DEVICE_AUTH_FEATURES
    + DEVICE_FEATURES + WALLET_FEATURES + MARKET_FEATURES
)
FEATURE_METADATA_COLUMNS = [
    "_feature_generated_at", "_feature_version", "_feature_hash", "_source_silver_transaction_hash",
]
FEATURE_TABLE_COLUMNS = FEATURE_VALUE_COLUMNS + FEATURE_METADATA_COLUMNS

MODEL_INPUT_CANDIDATE_COLUMNS = (
    ["asset", "transaction_type", "country"]
    + CURRENT_TRANSACTION_FEATURES + ACCOUNT_FEATURES + TRANSACTION_HISTORY_FEATURES
    + AUTHENTICATION_FEATURES + DEVICE_AUTH_FEATURES + DEVICE_FEATURES + WALLET_FEATURES
    + [
        "market_data_freshness_seconds", "latest_market_close_usd", "market_return_5m",
        "market_return_1h", "market_volatility_1h", "market_volume_sum_1h",
        "market_data_available",
    ]
)
NUMERIC_FEATURE_COLUMNS = [
    "transaction_amount_usd", "crypto_quantity", "market_price_usd", "transaction_hour_utc",
    "transaction_day_of_week_utc", "is_weekend_utc", "is_night_transaction_utc",
    "account_age_days", "account_profile_available", "normal_transaction_amount_usd",
    "normal_transaction_amount_available", "amount_to_normal_ratio", "amount_above_normal_usd",
    "transaction_country_mismatch_home_country", "prior_tx_count_5m", "prior_tx_count_1h",
    "prior_tx_count_24h", "prior_tx_amount_sum_1h", "prior_tx_amount_sum_24h",
    "prior_tx_amount_avg_24h", "prior_tx_amount_max_24h", "has_prior_tx_24h",
    "seconds_since_previous_tx", "has_previous_transaction", "failed_auth_count_10m",
    "failed_auth_count_1h", "successful_auth_count_1h", "recent_auth_failure_flag_10m",
    "seconds_since_last_successful_auth", "has_previous_successful_auth",
    "device_failed_auth_count_10m", "device_successful_auth_count_1h",
    "device_distinct_account_count_1h", "device_profile_available", "device_age_hours",
    "device_age_days", "is_new_device_24h", "source_wallet_applicable",
    "destination_wallet_applicable", "destination_wallet_first_seen_available",
    "destination_wallet_age_hours", "is_new_destination_wallet",
    "prior_destination_wallet_tx_count", "prior_source_wallet_tx_count",
    "market_data_freshness_seconds", "latest_market_close_usd", "market_return_5m",
    "market_return_1h", "market_volatility_1h", "market_volume_sum_1h", "market_data_available",
]
COUNT_FEATURE_COLUMNS = [
    "prior_tx_count_5m", "prior_tx_count_1h", "prior_tx_count_24h", "failed_auth_count_10m",
    "failed_auth_count_1h", "successful_auth_count_1h", "device_failed_auth_count_10m",
    "device_successful_auth_count_1h", "device_distinct_account_count_1h",
    "prior_destination_wallet_tx_count", "prior_source_wallet_tx_count",
]
AGE_FEATURE_COLUMNS = ["account_age_days", "device_age_hours", "device_age_days", "destination_wallet_age_hours"]
CATEGORICAL_FEATURE_COLUMNS = ["asset", "transaction_type", "country", "customer_risk_tier", "market_product_id"]
TRAINING_LABEL_COLUMNS = [
    "target_is_fraud", "label_timestamp", "investigation_status", "fraud_type", "label_source",
    "_source_silver_label_hash", "_source_silver_label_event_id",
]
TRAINING_METADATA_COLUMNS = ["_training_generated_at", "_training_row_hash"]

def quote_identifier(name: str) -> str:
    return ".".join(f"`{part}`" for part in name.split("."))


def full_table(schema_name: str, table_name: str) -> str:
    return f"{CATALOG}.{schema_name}.{table_name}"


FEATURE_TABLE = full_table(FEATURE_SCHEMA, FEATURE_TABLE_NAME)
TRAINING_TABLE = full_table(FEATURE_SCHEMA, TRAINING_TABLE_NAME)


def table_exists(table_name: str) -> bool:
    return spark.catalog.tableExists(table_name)


def count_rows(df: DataFrame) -> int:
    return int(df.count())


def create_delta_table_if_missing(table_name: str, df: DataFrame) -> bool:
    if table_exists(table_name):
        return False
    df.limit(0).write.format("delta").saveAsTable(table_name)
    return True


def merge_to_delta(table_name: str, source_df: DataFrame, key_columns: list[str], hash_column: str) -> dict[str, Any]:
    table_created = create_delta_table_if_missing(table_name, source_df)
    target_df = spark.table(table_name)
    pre_count = count_rows(target_df)
    existing_keys = target_df.select(*key_columns, hash_column)
    insert_count = count_rows(source_df.join(existing_keys.select(*key_columns), key_columns, "left_anti"))
    joined = source_df.alias("s").join(existing_keys.alias("t"), key_columns, "inner")
    update_count = count_rows(joined.filter(F.col(f"s.{hash_column}") != F.col(f"t.{hash_column}")))
    merge_condition = " AND ".join([f"t.`{column_name}` <=> s.`{column_name}`" for column_name in key_columns])
    update_condition = f"t.`{hash_column}` <> s.`{hash_column}`"
    (
        DeltaTable.forName(spark, table_name)
        .alias("t")
        .merge(source_df.alias("s"), merge_condition)
        .whenMatchedUpdateAll(condition=update_condition)
        .whenNotMatchedInsertAll()
        .execute()
    )
    final_count = count_rows(spark.table(table_name))
    detail = spark.sql(f"DESCRIBE DETAIL {quote_identifier(table_name)}").collect()[0].asDict()
    return {
        "table": table_name,
        "table_created": table_created,
        "pre_count": pre_count,
        "inserted_count": insert_count,
        "updated_count": update_count,
        "final_count": final_count,
        "partitionColumns": detail.get("partitionColumns") or [],
    }


def unix_seconds_between(later: str, earlier: str):
    return F.unix_timestamp(F.col(later)) - F.unix_timestamp(F.col(earlier))


def int_flag(condition: Any):
    return F.when(condition, F.lit(1)).otherwise(F.lit(0)).cast("int")


def source_table_snapshot(datasets: list[str]) -> dict[str, Any]:
    snapshot: dict[str, Any] = {}
    for dataset in datasets:
        table_name = full_table(SOURCE_SCHEMA, dataset)
        exists = table_exists(table_name)
        item: dict[str, Any] = {"table": table_name, "exists": exists}
        if exists:
            df = spark.table(table_name)
            detail = spark.sql(f"DESCRIBE DETAIL {quote_identifier(table_name)}").collect()[0].asDict()
            item.update({
                "count": count_rows(df),
                "schema": df.dtypes,
                "columns": df.columns,
                "partitionColumns": detail.get("partitionColumns") or [],
            })
        snapshot[dataset] = item
    return snapshot


def numeric_stats(df: DataFrame, columns: list[str]) -> dict[str, dict[str, Any]]:
    stats: dict[str, dict[str, Any]] = {}
    for column_name in columns:
        row = df.agg(
            F.min(F.col(column_name)).alias("min"),
            F.max(F.col(column_name)).alias("max"),
            F.avg(F.col(column_name).cast("double")).alias("avg"),
            F.sum(F.when(F.col(column_name).isNull(), F.lit(1)).otherwise(F.lit(0))).alias("null_count"),
            F.sum(F.when(F.col(column_name) == F.lit(0), F.lit(1)).otherwise(F.lit(0))).alias("zero_count"),
        ).collect()[0].asDict()
        stats[column_name] = {
            "min": row["min"],
            "max": row["max"],
            "avg": row["avg"],
            "null_count": int(row["null_count"] or 0),
            "zero_count": int(row["zero_count"] or 0),
        }
    return stats


def categorical_counts(df: DataFrame, columns: list[str]) -> dict[str, dict[str, int]]:
    output: dict[str, dict[str, int]] = {}
    for column_name in columns:
        rows = df.groupBy(column_name).count().orderBy(column_name).collect()
        output[column_name] = {"null" if row[column_name] is None else str(row[column_name]): int(row["count"]) for row in rows}
    return output


def non_finite_counts(df: DataFrame, columns: list[str]) -> dict[str, Any]:
    by_column: dict[str, dict[str, int]] = {}
    total_nan = 0
    total_infinite = 0
    positive_inf = float("inf")
    negative_inf = float("-inf")
    for column_name in columns:
        numeric_col = F.col(column_name).cast("double")
        nan_count = count_rows(df.filter(F.isnan(numeric_col)))
        inf_count = count_rows(df.filter((numeric_col == F.lit(positive_inf)) | (numeric_col == F.lit(negative_inf))))
        by_column[column_name] = {"nan": nan_count, "infinite": inf_count}
        total_nan += nan_count
        total_infinite += inf_count
    return {"nan_total": total_nan, "infinite_total": total_infinite, "by_column": by_column}


def negative_value_count(df: DataFrame, columns: list[str]) -> dict[str, int]:
    return {column_name: count_rows(df.filter(F.col(column_name) < F.lit(0))) for column_name in columns}


def duplicate_key_count(df: DataFrame, key_columns: list[str]) -> int:
    return count_rows(df.groupBy(*key_columns).count().filter(F.col("count") > 1))


def null_key_count(df: DataFrame, key_columns: list[str]) -> int:
    condition = None
    for column_name in key_columns:
        current = F.col(column_name).isNull()
        condition = current if condition is None else condition | current
    return count_rows(df.filter(condition))


def add_current_transaction_features(tx_df: DataFrame) -> DataFrame:
    day_of_week = (F.pmod(F.dayofweek(F.col("feature_timestamp")) + F.lit(5), F.lit(7)) + F.lit(1)).cast("int")
    hour = F.hour(F.col("feature_timestamp")).cast("int")
    return (
        tx_df.withColumn("transaction_hour_utc", hour)
        .withColumn("transaction_day_of_week_utc", day_of_week)
        .withColumn("is_weekend_utc", int_flag(F.col("transaction_day_of_week_utc").isin(6, 7)))
        .withColumn(
            "is_night_transaction_utc",
            int_flag((F.col("transaction_hour_utc") >= F.lit(NIGHT_START_HOUR_UTC)) & (F.col("transaction_hour_utc") <= F.lit(NIGHT_END_HOUR_UTC))),
        )
    )

def add_account_features(feature_df: DataFrame) -> DataFrame:
    accounts = spark.table(full_table(SOURCE_SCHEMA, "accounts")).select(
        "account_id",
        F.col("created_at").alias("_account_created_at"),
        F.col("updated_at").alias("_account_updated_at"),
        F.col("home_country").alias("_account_home_country"),
        F.col("customer_risk_tier").alias("_customer_risk_tier"),
        F.col("normal_transaction_amount_usd").cast("double").alias("_normal_transaction_amount_usd"),
    )
    joined = feature_df.join(F.broadcast(accounts), "account_id", "left")
    account_exists_at_tx = F.col("_account_created_at").isNotNull() & (F.col("_account_created_at") <= F.col("feature_timestamp"))
    profile_available = account_exists_at_tx & (F.col("_account_updated_at").isNull() | (F.col("_account_updated_at") <= F.col("feature_timestamp")))
    normal_available = profile_available & F.col("_normal_transaction_amount_usd").isNotNull() & (F.col("_normal_transaction_amount_usd") > F.lit(0.0))
    return (
        joined.withColumn("account_age_days", F.when(account_exists_at_tx, F.floor(unix_seconds_between("feature_timestamp", "_account_created_at") / F.lit(86400.0))).cast("long"))
        .withColumn("account_profile_available", int_flag(profile_available))
        .withColumn("customer_risk_tier", F.when(profile_available, F.col("_customer_risk_tier")).otherwise(F.lit(None).cast("string")))
        .withColumn("normal_transaction_amount_usd", F.when(profile_available, F.col("_normal_transaction_amount_usd")).otherwise(F.lit(None).cast("double")))
        .withColumn("normal_transaction_amount_available", int_flag(normal_available))
        .withColumn("amount_to_normal_ratio", F.when(normal_available, F.col("transaction_amount_usd") / F.col("_normal_transaction_amount_usd")).otherwise(F.lit(None).cast("double")))
        .withColumn("amount_above_normal_usd", F.when(normal_available, F.col("transaction_amount_usd") - F.col("_normal_transaction_amount_usd")).otherwise(F.lit(None).cast("double")))
        .withColumn(
            "transaction_country_mismatch_home_country",
            F.when(profile_available & F.col("country").isNotNull() & F.col("_account_home_country").isNotNull(), int_flag(F.col("country") != F.col("_account_home_country"))).otherwise(F.lit(None).cast("int")),
        )
        .drop("_account_created_at", "_account_updated_at", "_account_home_country", "_customer_risk_tier", "_normal_transaction_amount_usd")
    )



def build_transaction_history_features(tx_df: DataFrame) -> DataFrame:
    prepared = tx_df.select(
        "transaction_id", "account_id", "feature_timestamp",
        F.col("transaction_amount_usd").cast("double").alias("transaction_amount_usd"),
    ).withColumn("_feature_epoch", F.unix_timestamp("feature_timestamp"))
    ordered = Window.partitionBy("account_id").orderBy("_feature_epoch")
    w_5m = ordered.rangeBetween(-300, -1)
    w_1h = ordered.rangeBetween(-3600, -1)
    w_24h = ordered.rangeBetween(-86400, -1)
    prior_all = ordered.rangeBetween(Window.unboundedPreceding, -1)
    return (
        prepared
        .withColumn("prior_tx_count_5m", F.count("transaction_id").over(w_5m).cast("long"))
        .withColumn("prior_tx_count_1h", F.count("transaction_id").over(w_1h).cast("long"))
        .withColumn("prior_tx_count_24h", F.count("transaction_id").over(w_24h).cast("long"))
        .withColumn("prior_tx_amount_sum_1h", F.coalesce(F.sum("transaction_amount_usd").over(w_1h), F.lit(0.0)))
        .withColumn("prior_tx_amount_sum_24h", F.coalesce(F.sum("transaction_amount_usd").over(w_24h), F.lit(0.0)))
        .withColumn("prior_tx_amount_avg_24h", F.avg("transaction_amount_usd").over(w_24h))
        .withColumn("prior_tx_amount_max_24h", F.max("transaction_amount_usd").over(w_24h))
        .withColumn("_previous_tx_timestamp", F.max("feature_timestamp").over(prior_all))
        .withColumn("has_prior_tx_24h", int_flag(F.col("prior_tx_count_24h") > F.lit(0)))
        .withColumn(
            "seconds_since_previous_tx",
            F.when(F.col("_previous_tx_timestamp").isNotNull(), F.col("_feature_epoch") - F.unix_timestamp("_previous_tx_timestamp")).cast("long"),
        )
        .withColumn("has_previous_transaction", int_flag(F.col("_previous_tx_timestamp").isNotNull()))
        .select(
            "transaction_id", "prior_tx_count_5m", "prior_tx_count_1h", "prior_tx_count_24h",
            "prior_tx_amount_sum_1h", "prior_tx_amount_sum_24h", "prior_tx_amount_avg_24h",
            "prior_tx_amount_max_24h", "has_prior_tx_24h", "seconds_since_previous_tx",
            "has_previous_transaction",
        )
    )


def build_authentication_features(tx_df: DataFrame) -> DataFrame:
    tx_events = tx_df.select(
        "account_id", "transaction_id",
        F.col("feature_timestamp").alias("_event_timestamp"),
        F.lit(None).cast("boolean").alias("_auth_login_success"),
        F.lit("tx").alias("_event_kind"),
    )
    auth_events = spark.table(full_table(SOURCE_SCHEMA, "authentication_events")).select(
        "account_id",
        F.lit(None).cast("string").alias("transaction_id"),
        F.col("event_timestamp").alias("_event_timestamp"),
        F.col("login_success").alias("_auth_login_success"),
        F.lit("auth").alias("_event_kind"),
    )
    timeline = tx_events.unionByName(auth_events).withColumn("_event_epoch", F.unix_timestamp("_event_timestamp"))
    ordered = Window.partitionBy("account_id").orderBy("_event_epoch")
    w_10m = ordered.rangeBetween(-600, -1)
    w_1h = ordered.rangeBetween(-3600, -1)
    prior_all = ordered.rangeBetween(Window.unboundedPreceding, -1)
    failed = (F.col("_event_kind") == F.lit("auth")) & (F.col("_auth_login_success") == F.lit(False))
    successful = (F.col("_event_kind") == F.lit("auth")) & (F.col("_auth_login_success") == F.lit(True))
    return (
        timeline
        .withColumn("failed_auth_count_10m", F.sum(F.when(failed, F.lit(1)).otherwise(F.lit(0))).over(w_10m).cast("long"))
        .withColumn("failed_auth_count_1h", F.sum(F.when(failed, F.lit(1)).otherwise(F.lit(0))).over(w_1h).cast("long"))
        .withColumn("successful_auth_count_1h", F.sum(F.when(successful, F.lit(1)).otherwise(F.lit(0))).over(w_1h).cast("long"))
        .withColumn("_last_successful_auth_timestamp", F.max(F.when(successful, F.col("_event_timestamp"))).over(prior_all))
        .filter(F.col("_event_kind") == F.lit("tx"))
        .withColumn("recent_auth_failure_flag_10m", int_flag(F.col("failed_auth_count_10m") > F.lit(0)))
        .withColumn(
            "seconds_since_last_successful_auth",
            F.when(F.col("_last_successful_auth_timestamp").isNotNull(), F.col("_event_epoch") - F.unix_timestamp("_last_successful_auth_timestamp")).cast("long"),
        )
        .withColumn("has_previous_successful_auth", int_flag(F.col("_last_successful_auth_timestamp").isNotNull()))
        .select(
            "transaction_id", "failed_auth_count_10m", "failed_auth_count_1h",
            "successful_auth_count_1h", "recent_auth_failure_flag_10m",
            "seconds_since_last_successful_auth", "has_previous_successful_auth",
        )
    )


def build_device_authentication_features(tx_df: DataFrame) -> DataFrame:
    tx_events = tx_df.select(
        "device_id", "transaction_id",
        F.lit(None).cast("string").alias("_auth_account_id"),
        F.col("feature_timestamp").alias("_event_timestamp"),
        F.lit(None).cast("boolean").alias("_auth_login_success"),
        F.lit("tx").alias("_event_kind"),
    )
    auth_events = spark.table(full_table(SOURCE_SCHEMA, "authentication_events")).select(
        "device_id",
        F.lit(None).cast("string").alias("transaction_id"),
        F.col("account_id").alias("_auth_account_id"),
        F.col("event_timestamp").alias("_event_timestamp"),
        F.col("login_success").alias("_auth_login_success"),
        F.lit("auth").alias("_event_kind"),
    )
    timeline = tx_events.unionByName(auth_events).withColumn("_event_epoch", F.unix_timestamp("_event_timestamp"))
    ordered = Window.partitionBy("device_id").orderBy("_event_epoch")
    w_10m = ordered.rangeBetween(-600, -1)
    w_1h = ordered.rangeBetween(-3600, -1)
    failed = (F.col("_event_kind") == F.lit("auth")) & (F.col("_auth_login_success") == F.lit(False))
    successful = (F.col("_event_kind") == F.lit("auth")) & (F.col("_auth_login_success") == F.lit(True))
    auth_account_in_window = F.when(F.col("_event_kind") == F.lit("auth"), F.col("_auth_account_id"))
    return (
        timeline
        .withColumn("device_failed_auth_count_10m", F.sum(F.when(failed, F.lit(1)).otherwise(F.lit(0))).over(w_10m).cast("long"))
        .withColumn("device_successful_auth_count_1h", F.sum(F.when(successful, F.lit(1)).otherwise(F.lit(0))).over(w_1h).cast("long"))
        .withColumn("device_distinct_account_count_1h", F.size(F.collect_set(auth_account_in_window).over(w_1h)).cast("long"))
        .filter(F.col("_event_kind") == F.lit("tx"))
        .select("transaction_id", "device_failed_auth_count_10m", "device_successful_auth_count_1h", "device_distinct_account_count_1h")
    )

def add_device_features(feature_df: DataFrame) -> DataFrame:
    devices = spark.table(full_table(SOURCE_SCHEMA, "devices")).select("device_id", F.col("first_seen_at").alias("_device_first_seen_at"))
    joined = feature_df.join(F.broadcast(devices), "device_id", "left")
    available = F.col("_device_first_seen_at").isNotNull() & (F.col("_device_first_seen_at") <= F.col("feature_timestamp"))
    age_hours = unix_seconds_between("feature_timestamp", "_device_first_seen_at") / F.lit(3600.0)
    return (
        joined.withColumn("device_profile_available", int_flag(available))
        .withColumn("device_age_hours", F.when(available, age_hours).otherwise(F.lit(None).cast("double")))
        .withColumn("device_age_days", F.when(available, age_hours / F.lit(24.0)).otherwise(F.lit(None).cast("double")))
        .withColumn("is_new_device_24h", F.when(available, int_flag(age_hours <= F.lit(24.0))).otherwise(F.lit(None).cast("int")))
        .drop("_device_first_seen_at")
    )



def build_wallet_history_features(tx_df: DataFrame) -> DataFrame:
    prepared = tx_df.select(
        "transaction_id", "source_wallet_id", "destination_wallet_id", "feature_timestamp"
    ).withColumn("_feature_epoch", F.unix_timestamp("feature_timestamp"))
    source_applicable = F.col("source_wallet_id").isNotNull()
    destination_applicable = F.col("destination_wallet_id").isNotNull()
    source_window = Window.partitionBy("source_wallet_id").orderBy("_feature_epoch").rangeBetween(Window.unboundedPreceding, -1)
    destination_window = Window.partitionBy("destination_wallet_id").orderBy("_feature_epoch").rangeBetween(Window.unboundedPreceding, -1)
    return (
        prepared
        .withColumn("prior_destination_wallet_tx_count", F.when(destination_applicable, F.count("transaction_id").over(destination_window)).cast("long"))
        .withColumn("prior_source_wallet_tx_count", F.when(source_applicable, F.count("transaction_id").over(source_window)).cast("long"))
        .withColumn("source_wallet_applicable", int_flag(F.col("source_wallet_id").isNotNull()))
        .withColumn("destination_wallet_applicable", int_flag(F.col("destination_wallet_id").isNotNull()))
        .withColumn("is_new_destination_wallet", F.when(F.col("destination_wallet_id").isNotNull(), int_flag(F.col("prior_destination_wallet_tx_count") == F.lit(0))).otherwise(F.lit(None).cast("int")))
        .select(
            "transaction_id", "prior_destination_wallet_tx_count", "prior_source_wallet_tx_count",
            "source_wallet_applicable", "destination_wallet_applicable", "is_new_destination_wallet",
        )
    )

def add_destination_wallet_age(feature_df: DataFrame) -> DataFrame:
    wallets = spark.table(full_table(SOURCE_SCHEMA, "wallets")).select(
        F.col("wallet_id").alias("_destination_wallet_id"),
        F.col("first_seen_at").alias("_destination_wallet_first_seen_at"),
    )
    joined = feature_df.join(F.broadcast(wallets), F.col("destination_wallet_id") == F.col("_destination_wallet_id"), "left")
    available = F.col("destination_wallet_id").isNotNull() & F.col("_destination_wallet_first_seen_at").isNotNull() & (F.col("_destination_wallet_first_seen_at") <= F.col("feature_timestamp"))
    return (
        joined.withColumn("destination_wallet_first_seen_available", int_flag(available))
        .withColumn("destination_wallet_age_hours", F.when(available, unix_seconds_between("feature_timestamp", "_destination_wallet_first_seen_at") / F.lit(3600.0)).otherwise(F.lit(None).cast("double")))
        .drop("_destination_wallet_id", "_destination_wallet_first_seen_at")
    )



def build_market_features(tx_df: DataFrame) -> DataFrame:
    candles = (
        spark.table(full_table(SOURCE_SCHEMA, "market_candles"))
        .select(
            "product_id", "candle_start_timestamp", "candle_end_timestamp",
            F.col("close_price_usd").cast("double").alias("close_price_usd"),
            F.col("volume").cast("double").alias("volume"),
        )
        .withColumn("_candle_end_epoch", F.unix_timestamp("candle_end_timestamp"))
    )
    ordered = Window.partitionBy("product_id").orderBy("_candle_end_epoch")
    candles = candles.withColumn("_previous_close_price_usd", F.lag("close_price_usd").over(ordered))
    candles = candles.withColumn(
        "_log_return",
        F.when(F.col("_previous_close_price_usd").isNotNull() & (F.col("_previous_close_price_usd") > F.lit(0.0)), F.log(F.col("close_price_usd") / F.col("_previous_close_price_usd"))).otherwise(F.lit(None).cast("double")),
    )
    baseline_5m = F.last("close_price_usd", ignorenulls=True).over(ordered.rangeBetween(Window.unboundedPreceding, -300))
    baseline_1h = F.last("close_price_usd", ignorenulls=True).over(ordered.rangeBetween(Window.unboundedPreceding, -3600))
    one_hour = ordered.rangeBetween(-3600, 0)
    candle_features = (
        candles.withColumn("_baseline_close_5m", baseline_5m)
        .withColumn("_baseline_close_1h", baseline_1h)
        .withColumn("market_return_5m", F.when(F.col("_baseline_close_5m") > F.lit(0.0), F.col("close_price_usd") / F.col("_baseline_close_5m") - F.lit(1.0)).otherwise(F.lit(None).cast("double")))
        .withColumn("market_return_1h", F.when(F.col("_baseline_close_1h") > F.lit(0.0), F.col("close_price_usd") / F.col("_baseline_close_1h") - F.lit(1.0)).otherwise(F.lit(None).cast("double")))
        .withColumn("market_volatility_1h", F.stddev_samp("_log_return").over(one_hour))
        .withColumn("market_volume_sum_1h", F.sum("volume").over(one_hour))
        .select("product_id", "candle_end_timestamp", F.col("close_price_usd").alias("latest_market_close_usd"), "market_return_5m", "market_return_1h", "market_volatility_1h", "market_volume_sum_1h")
    )
    mapping_expr = F.create_map([F.lit(value) for pair in ASSET_PRODUCT_MAPPING.items() for value in pair])
    tx_events = (
        tx_df.select("transaction_id", "asset", "feature_timestamp")
        .withColumn("product_id", mapping_expr[F.col("asset")])
        .withColumn("_event_timestamp", F.col("feature_timestamp"))
        .withColumn("_event_epoch", F.unix_timestamp("feature_timestamp"))
        .withColumn("_event_order", F.lit(1))
        .withColumn("candle_end_timestamp", F.lit(None).cast("timestamp"))
        .withColumn("latest_market_close_usd", F.lit(None).cast("double"))
        .withColumn("market_return_5m", F.lit(None).cast("double"))
        .withColumn("market_return_1h", F.lit(None).cast("double"))
        .withColumn("market_volatility_1h", F.lit(None).cast("double"))
        .withColumn("market_volume_sum_1h", F.lit(None).cast("double"))
    )
    candle_events = (
        candle_features
        .withColumn("transaction_id", F.lit(None).cast("string"))
        .withColumn("asset", F.lit(None).cast("string"))
        .withColumn("feature_timestamp", F.lit(None).cast("timestamp"))
        .withColumn("_event_timestamp", F.col("candle_end_timestamp"))
        .withColumn("_event_epoch", F.unix_timestamp("candle_end_timestamp"))
        .withColumn("_event_order", F.lit(0))
        .select(tx_events.columns)
    )
    timeline = tx_events.unionByName(candle_events)
    market_window = Window.partitionBy("product_id").orderBy("_event_epoch", "_event_order").rowsBetween(Window.unboundedPreceding, Window.currentRow)
    ranked = (
        timeline
        .withColumn("_latest_candle_end_timestamp", F.last(F.when(F.col("_event_order") == F.lit(0), F.col("candle_end_timestamp")), ignorenulls=True).over(market_window))
        .withColumn("_latest_market_close_usd", F.last(F.when(F.col("_event_order") == F.lit(0), F.col("latest_market_close_usd")), ignorenulls=True).over(market_window))
        .withColumn("_market_return_5m", F.last(F.when(F.col("_event_order") == F.lit(0), F.col("market_return_5m")), ignorenulls=True).over(market_window))
        .withColumn("_market_return_1h", F.last(F.when(F.col("_event_order") == F.lit(0), F.col("market_return_1h")), ignorenulls=True).over(market_window))
        .withColumn("_market_volatility_1h", F.last(F.when(F.col("_event_order") == F.lit(0), F.col("market_volatility_1h")), ignorenulls=True).over(market_window))
        .withColumn("_market_volume_sum_1h", F.last(F.when(F.col("_event_order") == F.lit(0), F.col("market_volume_sum_1h")), ignorenulls=True).over(market_window))
        .filter(F.col("_event_order") == F.lit(1))
    )
    return (
        ranked
        .select(
            "transaction_id",
            F.col("product_id").alias("market_product_id"),
            F.col("_latest_candle_end_timestamp").alias("latest_market_candle_end_timestamp"),
            F.when(F.col("_latest_candle_end_timestamp").isNotNull(), F.unix_timestamp("feature_timestamp") - F.unix_timestamp("_latest_candle_end_timestamp")).cast("long").alias("market_data_freshness_seconds"),
            F.col("_latest_market_close_usd").alias("latest_market_close_usd"),
            F.col("_market_return_5m").alias("market_return_5m"),
            F.col("_market_return_1h").alias("market_return_1h"),
            F.col("_market_volatility_1h").alias("market_volatility_1h"),
            F.col("_market_volume_sum_1h").alias("market_volume_sum_1h"),
            int_flag(F.col("_latest_candle_end_timestamp").isNotNull()).alias("market_data_available"),
        )
    )

def build_feature_rows() -> DataFrame:
    transactions = (
        spark.table(full_table(SOURCE_SCHEMA, "customer_transactions"))
        .select(
            "transaction_id", "account_id", "device_id", "source_wallet_id", "destination_wallet_id",
            F.col("event_timestamp").alias("feature_timestamp"), "event_date", "asset", "transaction_type", "country",
            F.col("transaction_amount_usd").cast("double").alias("transaction_amount_usd"),
            F.col("crypto_quantity").cast("double").alias("crypto_quantity"),
            F.col("market_price_usd").cast("double").alias("market_price_usd"),
            F.col("_record_hash").alias("_source_silver_transaction_hash"),
        )
        .cache()
    )
    count_rows(transactions)
    feature_df = add_current_transaction_features(transactions)
    feature_df = add_account_features(feature_df)
    feature_df = feature_df.join(build_transaction_history_features(transactions), "transaction_id", "left")
    feature_df = feature_df.join(build_authentication_features(transactions), "transaction_id", "left")
    feature_df = feature_df.join(build_device_authentication_features(transactions), "transaction_id", "left")
    feature_df = add_device_features(feature_df)
    feature_df = feature_df.join(build_wallet_history_features(transactions), "transaction_id", "left")
    feature_df = add_destination_wallet_age(feature_df)
    feature_df = feature_df.join(build_market_features(transactions), "transaction_id", "left")
    for column_name in COUNT_FEATURE_COLUMNS:
        feature_df = feature_df.withColumn(column_name, F.coalesce(F.col(column_name), F.lit(0)).cast("long"))
    for column_name in [
        "is_weekend_utc", "is_night_transaction_utc", "account_profile_available",
        "normal_transaction_amount_available", "has_prior_tx_24h", "has_previous_transaction",
        "recent_auth_failure_flag_10m", "has_previous_successful_auth", "device_profile_available",
        "source_wallet_applicable", "destination_wallet_applicable", "destination_wallet_first_seen_available",
        "market_data_available",
    ]:
        feature_df = feature_df.withColumn(column_name, F.coalesce(F.col(column_name), F.lit(0)).cast("int"))
    feature_df = feature_df.withColumn("_feature_generated_at", F.current_timestamp()).withColumn("_feature_version", F.lit(FEATURE_VERSION))
    hash_columns = [column_name for column_name in FEATURE_TABLE_COLUMNS if column_name not in {"_feature_generated_at", "_feature_hash"}]
    feature_df = feature_df.withColumn("_feature_hash", F.sha2(F.to_json(F.struct(*[F.col(column_name).alias(column_name) for column_name in hash_columns])), 256))
    return feature_df.select(*FEATURE_TABLE_COLUMNS)


def validate_feature_table(feature_df: DataFrame) -> dict[str, Any]:
    non_finite = non_finite_counts(feature_df, NUMERIC_FEATURE_COLUMNS)
    negative_counts = negative_value_count(feature_df, COUNT_FEATURE_COLUMNS)
    negative_ages = negative_value_count(feature_df, AGE_FEATURE_COLUMNS)
    leakage_columns = sorted(FORBIDDEN_FEATURE_TABLE_COLUMNS.intersection(feature_df.columns))
    return {
        "expected_rows": EXPECTED_SOURCE_COUNTS["customer_transactions"],
        "row_count": count_rows(feature_df),
        "unique_transaction_ids": count_rows(feature_df.select("transaction_id").dropDuplicates()),
        "null_transaction_ids": null_key_count(feature_df, ["transaction_id"]),
        "duplicate_transaction_ids": duplicate_key_count(feature_df, ["transaction_id"]),
        "fraud_label_columns_present": len(leakage_columns),
        "fraud_label_columns": leakage_columns,
        "nan_numeric_values": non_finite["nan_total"],
        "infinite_numeric_values": non_finite["infinite_total"],
        "non_finite_numeric_values_by_column": non_finite["by_column"],
        "negative_count_features": sum(negative_counts.values()),
        "negative_count_features_by_column": negative_counts,
        "invalid_negative_age_features": sum(negative_ages.values()),
        "invalid_negative_age_features_by_column": negative_ages,
        "market_data_unavailable_count": count_rows(feature_df.filter(F.col("market_data_available") == F.lit(0))),
    }

def leakage_checks() -> dict[str, int]:
    tx = spark.table(full_table(SOURCE_SCHEMA, "customer_transactions")).select(
        "transaction_id", "account_id", "device_id", "source_wallet_id", "destination_wallet_id", F.col("event_timestamp").alias("feature_timestamp")
    )
    tx_current = tx.alias("c")
    tx_history = tx.select(
        F.col("transaction_id").alias("hist_transaction_id"), F.col("account_id").alias("hist_account_id"),
        F.col("source_wallet_id").alias("hist_source_wallet_id"), F.col("destination_wallet_id").alias("hist_destination_wallet_id"),
        F.col("feature_timestamp").alias("hist_event_timestamp"),
    ).alias("h")
    tx_pairs = tx_current.join(tx_history, (F.col("c.account_id") == F.col("h.hist_account_id")) & (F.col("h.hist_event_timestamp") < F.col("c.feature_timestamp")), "inner")
    auth = spark.table(full_table(SOURCE_SCHEMA, "authentication_events")).select(
        F.col("account_id").alias("auth_account_id"), F.col("device_id").alias("auth_device_id"), F.col("event_timestamp").alias("auth_event_timestamp")
    )
    auth_pairs = tx_current.join(auth.alias("a"), (F.col("c.account_id") == F.col("a.auth_account_id")) & (F.col("a.auth_event_timestamp") < F.col("c.feature_timestamp")), "inner")
    device_auth_pairs = tx_current.join(auth.alias("a"), (F.col("c.device_id") == F.col("a.auth_device_id")) & (F.col("a.auth_event_timestamp") < F.col("c.feature_timestamp")), "inner")
    feature_table = spark.table(FEATURE_TABLE)
    accounts = spark.table(full_table(SOURCE_SCHEMA, "accounts")).select("account_id", "created_at", "updated_at")
    devices = spark.table(full_table(SOURCE_SCHEMA, "devices")).select("device_id", "first_seen_at")
    wallets = spark.table(full_table(SOURCE_SCHEMA, "wallets")).select(F.col("wallet_id").alias("_wallet_id"), F.col("first_seen_at").alias("_wallet_first_seen_at"))
    account_joined = feature_table.join(F.broadcast(accounts), "account_id", "left")
    device_joined = feature_table.join(F.broadcast(devices), "device_id", "left")
    wallet_joined = feature_table.join(F.broadcast(wallets), F.col("destination_wallet_id") == F.col("_wallet_id"), "left")
    return {
        "historical_transaction_timestamp_gte_feature_timestamp": count_rows(tx_pairs.filter(F.col("h.hist_event_timestamp") >= F.col("c.feature_timestamp"))),
        "authentication_timestamp_gte_feature_timestamp": count_rows(auth_pairs.filter(F.col("a.auth_event_timestamp") >= F.col("c.feature_timestamp"))),
        "device_authentication_timestamp_gte_feature_timestamp": count_rows(device_auth_pairs.filter(F.col("a.auth_event_timestamp") >= F.col("c.feature_timestamp"))),
        "market_candle_end_after_feature_timestamp": count_rows(feature_table.filter(F.col("latest_market_candle_end_timestamp").isNotNull() & (F.col("latest_market_candle_end_timestamp") > F.col("feature_timestamp")))),
        "account_created_after_feature_timestamp": count_rows(account_joined.filter(F.col("created_at").isNotNull() & (F.col("created_at") > F.col("feature_timestamp")))),
        "account_profile_updated_after_feature_timestamp": count_rows(account_joined.filter(F.col("updated_at").isNotNull() & (F.col("updated_at") > F.col("feature_timestamp")))),
        "device_first_seen_after_feature_timestamp": count_rows(device_joined.filter(F.col("first_seen_at").isNotNull() & (F.col("first_seen_at") > F.col("feature_timestamp")))),
        "destination_wallet_first_seen_after_feature_timestamp": count_rows(wallet_joined.filter(F.col("_wallet_first_seen_at").isNotNull() & (F.col("_wallet_first_seen_at") > F.col("feature_timestamp")))),
    }


def build_final_labels() -> tuple[DataFrame, dict[str, int]]:
    labels = spark.table(full_table(SOURCE_SCHEMA, "fraud_labels"))
    duplicate_labels = labels.groupBy("transaction_id").count().filter(F.col("count") > 1)
    multiplicity = {
        "transactions_with_multiple_labels": count_rows(duplicate_labels),
        "max_labels_per_transaction": int(duplicate_labels.agg(F.max("count")).collect()[0][0] or 1),
    }
    supervised = labels.filter(F.col("is_fraud").isNotNull() & F.col("investigation_status").isin("CONFIRMED_FRAUD", "CLEARED"))
    label_order = Window.partitionBy("transaction_id").orderBy(
        F.col("label_timestamp").desc_nulls_last(), F.col("ingestion_timestamp").desc_nulls_last(),
        F.col("source_timestamp").desc_nulls_last(), F.col("event_id").desc_nulls_last(),
    )
    final_labels = (
        supervised.withColumn("_label_rank", F.row_number().over(label_order))
        .filter(F.col("_label_rank") == 1)
        .select(
            "transaction_id", F.col("is_fraud").alias("target_is_fraud"), "label_timestamp",
            "investigation_status", "fraud_type", "label_source",
            F.col("_record_hash").alias("_source_silver_label_hash"),
            F.col("event_id").alias("_source_silver_label_event_id"),
        )
    )
    return final_labels, multiplicity


def build_training_rows(feature_df: DataFrame, final_labels: DataFrame) -> DataFrame:
    training = feature_df.join(final_labels, "transaction_id", "inner")
    training = training.withColumn("_training_generated_at", F.current_timestamp())
    hash_columns = ["_feature_hash", "target_is_fraud", "label_timestamp", "investigation_status", "fraud_type", "label_source", "_source_silver_label_hash"]
    training = training.withColumn("_training_row_hash", F.sha2(F.to_json(F.struct(*[F.col(column_name).alias(column_name) for column_name in hash_columns])), 256))
    return training.select(*(FEATURE_TABLE_COLUMNS + TRAINING_LABEL_COLUMNS + TRAINING_METADATA_COLUMNS))


def validate_training_table(training_df: DataFrame, feature_df: DataFrame) -> dict[str, Any]:
    target_distribution = categorical_counts(training_df, ["target_is_fraud"])["target_is_fraud"]
    training_row_count = count_rows(training_df)
    feature_row_count = count_rows(feature_df)
    return {
        "expected_labeled_rows_approx": EXPECTED_SOURCE_COUNTS["fraud_labels"],
        "row_count": training_row_count,
        "unlabeled_feature_rows": feature_row_count - training_row_count,
        "unique_transaction_ids": count_rows(training_df.select("transaction_id").dropDuplicates()),
        "duplicate_transaction_ids": duplicate_key_count(training_df, ["transaction_id"]),
        "target_null_count": count_rows(training_df.filter(F.col("target_is_fraud").isNull())),
        "target_class_distribution": target_distribution,
        "confirmed_fraud_count": count_rows(training_df.filter(F.col("target_is_fraud") == F.lit(True))),
        "normal_count": count_rows(training_df.filter(F.col("target_is_fraud") == F.lit(False))),
        "label_timestamp_before_transaction_violations": count_rows(training_df.filter(F.col("label_timestamp") < F.col("feature_timestamp"))),
        "label_timestamp_not_after_transaction_violations": count_rows(training_df.filter(F.col("label_timestamp") <= F.col("feature_timestamp"))),
        "training_rows_missing_in_feature_table": count_rows(training_df.select("transaction_id").join(feature_df.select("transaction_id"), "transaction_id", "left_anti")),
    }


def collect_rows_as_dict(df: DataFrame, limit: int = 20) -> list[dict[str, Any]]:
    return [row.asDict(recursive=True) for row in df.limit(limit).collect()]


def point_in_time_audit_sample(feature_df: DataFrame, training_df: DataFrame) -> dict[str, Any]:
    candidate_tags: dict[str, set[str]] = {}

    def add_candidates(tag: str, df: DataFrame, limit: int = 2) -> None:
        for row in df.select("transaction_id").orderBy("transaction_id").limit(limit).collect():
            candidate_tags.setdefault(row["transaction_id"], set()).add(tag)

    add_candidates("fraud_transaction", training_df.filter(F.col("target_is_fraud") == F.lit(True)), 2)
    add_candidates("normal_transaction", training_df.filter(F.col("target_is_fraud") == F.lit(False)), 2)
    add_candidates("no_prior_history", feature_df.filter(F.col("has_previous_transaction") == F.lit(0)), 2)
    add_candidates("recent_auth_failure", feature_df.filter(F.col("recent_auth_failure_flag_10m") == F.lit(1)), 2)
    add_candidates("new_device_or_wallet", feature_df.filter((F.col("is_new_device_24h") == F.lit(1)) | (F.col("is_new_destination_wallet") == F.lit(1))), 2)
    add_candidates("BTC", feature_df.filter(F.col("asset") == F.lit("BTC")), 1)
    add_candidates("ETH", feature_df.filter(F.col("asset") == F.lit("ETH")), 1)

    tx = spark.table(full_table(SOURCE_SCHEMA, "customer_transactions")).select(
        "transaction_id", "account_id", "source_wallet_id", "destination_wallet_id", F.col("event_timestamp").alias("event_timestamp")
    )
    auth = spark.table(full_table(SOURCE_SCHEMA, "authentication_events")).select(
        "event_id", "account_id", "device_id", F.col("event_timestamp").alias("event_timestamp"), "login_success"
    )
    samples = []
    for transaction_id in sorted(candidate_tags):
        feature_row = feature_df.filter(F.col("transaction_id") == F.lit(transaction_id)).collect()[0].asDict(recursive=True)
        feature_timestamp = feature_row["feature_timestamp"]
        account_id = feature_row["account_id"]
        device_id = feature_row["device_id"]
        source_wallet_id = feature_row["source_wallet_id"]
        destination_wallet_id = feature_row["destination_wallet_id"]
        prior_tx_all = tx.filter((F.col("account_id") == F.lit(account_id)) & (F.col("event_timestamp") < F.lit(feature_timestamp)))
        prior_tx_24h = prior_tx_all.filter(F.col("event_timestamp") >= F.lit(feature_timestamp - timedelta(hours=24))).orderBy(F.col("event_timestamp").desc(), F.col("transaction_id"))
        prior_auth_1h = auth.filter((F.col("account_id") == F.lit(account_id)) & (F.col("event_timestamp") < F.lit(feature_timestamp)) & (F.col("event_timestamp") >= F.lit(feature_timestamp - timedelta(hours=1)))).orderBy(F.col("event_timestamp").desc(), F.col("event_id"))
        prior_device_auth_1h = auth.filter((F.col("device_id") == F.lit(device_id)) & (F.col("event_timestamp") < F.lit(feature_timestamp)) & (F.col("event_timestamp") >= F.lit(feature_timestamp - timedelta(hours=1)))).orderBy(F.col("event_timestamp").desc(), F.col("event_id"))
        prior_destination = []
        if destination_wallet_id is not None:
            prior_destination = collect_rows_as_dict(prior_tx_all.filter(F.col("destination_wallet_id") == F.lit(destination_wallet_id)).orderBy(F.col("event_timestamp").desc(), F.col("transaction_id")), 20)
        prior_source = []
        if source_wallet_id is not None:
            prior_source = collect_rows_as_dict(prior_tx_all.filter(F.col("source_wallet_id") == F.lit(source_wallet_id)).orderBy(F.col("event_timestamp").desc(), F.col("transaction_id")), 20)
        prior_tx_rows = collect_rows_as_dict(prior_tx_24h, 20)
        prior_auth_rows = collect_rows_as_dict(prior_auth_1h, 20)
        prior_device_auth_rows = collect_rows_as_dict(prior_device_auth_1h, 20)
        latest_market_candle_timestamp = feature_row.get("latest_market_candle_end_timestamp")
        samples.append({
            "transaction_id": transaction_id,
            "category_tags": sorted(candidate_tags[transaction_id]),
            "feature_timestamp": feature_timestamp,
            "window_boundaries": {
                "transaction_5m_start_exclusive_current": feature_timestamp - timedelta(minutes=5),
                "authentication_10m_start_exclusive_current": feature_timestamp - timedelta(minutes=10),
                "one_hour_start_exclusive_current": feature_timestamp - timedelta(hours=1),
                "twenty_four_hour_start_exclusive_current": feature_timestamp - timedelta(hours=24),
            },
            "prior_transaction_events_24h": prior_tx_rows,
            "prior_authentication_events_1h": prior_auth_rows,
            "prior_device_authentication_events_1h": prior_device_auth_rows,
            "prior_destination_wallet_transactions": prior_destination,
            "prior_source_wallet_transactions": prior_source,
            "latest_market_candle_end_timestamp": latest_market_candle_timestamp,
            "calculated_feature_values": {column_name: feature_row.get(column_name) for column_name in FEATURE_VALUE_COLUMNS},
            "no_future_event_used": {
                "transactions": all(row["event_timestamp"] < feature_timestamp for row in prior_tx_rows),
                "authentication_events": all(row["event_timestamp"] < feature_timestamp for row in prior_auth_rows),
                "device_authentication_events": all(row["event_timestamp"] < feature_timestamp for row in prior_device_auth_rows),
                "destination_wallet_history": all(row["event_timestamp"] < feature_timestamp for row in prior_destination),
                "source_wallet_history": all(row["event_timestamp"] < feature_timestamp for row in prior_source),
                "market_candle": latest_market_candle_timestamp is None or latest_market_candle_timestamp <= feature_timestamp,
            },
        })
    return {
        "sample_policy": "Deterministic lexicographic transaction_id samples by required category; duplicate transaction ids are merged with category tags.",
        "strict_timestamp_policy": "Historical events are included only when event_timestamp < feature_timestamp. Identical timestamps are excluded.",
        "samples": samples,
    }

def failed_feature_checks(feature_quality: dict[str, Any], leakage: dict[str, int]) -> list[str]:
    failures = []
    if feature_quality["row_count"] != feature_quality["expected_rows"]:
        failures.append("feature_row_count")
    if feature_quality["unique_transaction_ids"] != feature_quality["expected_rows"]:
        failures.append("feature_unique_transaction_ids")
    if feature_quality["null_transaction_ids"] != 0:
        failures.append("feature_null_transaction_ids")
    if feature_quality["duplicate_transaction_ids"] != 0:
        failures.append("feature_duplicate_transaction_ids")
    if feature_quality["fraud_label_columns_present"] != 0:
        failures.append("feature_label_columns_present")
    if feature_quality["nan_numeric_values"] != 0:
        failures.append("feature_nan_numeric_values")
    if feature_quality["infinite_numeric_values"] != 0:
        failures.append("feature_infinite_numeric_values")
    if feature_quality["negative_count_features"] != 0:
        failures.append("feature_negative_count_features")
    if feature_quality["invalid_negative_age_features"] != 0:
        failures.append("feature_negative_age_features")
    if sum(leakage.values()) != 0:
        failures.append("feature_point_in_time_leakage")
    return failures


def failed_training_checks(training_quality: dict[str, Any]) -> list[str]:
    failures = []
    if training_quality["duplicate_transaction_ids"] != 0:
        failures.append("training_duplicate_transaction_ids")
    if training_quality["target_null_count"] != 0:
        failures.append("training_target_null_count")
    if training_quality["label_timestamp_before_transaction_violations"] != 0:
        failures.append("training_label_timestamp_before_transaction")
    if training_quality["label_timestamp_not_after_transaction_violations"] != 0:
        failures.append("training_label_timestamp_not_after_transaction")
    if training_quality["training_rows_missing_in_feature_table"] != 0:
        failures.append("training_rows_missing_in_feature_table")
    if training_quality["unique_transaction_ids"] != training_quality["row_count"]:
        failures.append("training_unique_transaction_ids")
    return failures


spark.sql(f"USE CATALOG {quote_identifier(CATALOG)}")
spark.sql(f"CREATE SCHEMA IF NOT EXISTS {quote_identifier(CATALOG)}.{quote_identifier(FEATURE_SCHEMA)}")

feature_source_datasets = ["accounts", "devices", "wallets", "authentication_events", "customer_transactions", "market_candles"]
source_snapshot = source_table_snapshot(feature_source_datasets)
source_count_failures = [
    dataset for dataset in feature_source_datasets
    if not source_snapshot[dataset].get("exists") or source_snapshot[dataset].get("count") != EXPECTED_SOURCE_COUNTS[dataset]
]

feature_rows = build_feature_rows().cache()
count_rows(feature_rows)
feature_prewrite_quality = validate_feature_table(feature_rows)
prewrite_failures = failed_feature_checks(feature_prewrite_quality, {"prewrite_no_persistent_leakage_check": 0})
if prewrite_failures:
    output = {
        "notebook": "03_offline_feature_engineering",
        "phase": "8",
        "run_id": RUN_ID,
        "cluster_id": CLUSTER_ID,
        "feature_table": FEATURE_TABLE,
        "source_silver_snapshot": source_snapshot,
        "feature_prewrite_quality": feature_prewrite_quality,
        "failed_checks": prewrite_failures,
        "overall_status": "FAIL",
    }
    print("PHASE8_RESULT_JSON=" + json.dumps(output, indent=2, sort_keys=True, default=str))
    raise RuntimeError(f"Feature prewrite quality checks failed: {prewrite_failures}")

feature_merge_result = merge_to_delta(FEATURE_TABLE, feature_rows, ["transaction_id"], "_feature_hash")
feature_table_df = spark.table(FEATURE_TABLE).cache()
count_rows(feature_table_df)
feature_quality = validate_feature_table(feature_table_df)
leakage = leakage_checks()
feature_failures = failed_feature_checks(feature_quality, leakage)

if feature_failures:
    output = {
        "notebook": "03_offline_feature_engineering",
        "phase": "8",
        "run_id": RUN_ID,
        "cluster_id": CLUSTER_ID,
        "feature_table": FEATURE_TABLE,
        "feature_merge": feature_merge_result,
        "source_silver_snapshot": source_snapshot,
        "feature_quality": feature_quality,
        "leakage_checks": leakage,
        "failed_checks": feature_failures,
        "overall_status": "FAIL",
    }
    print("PHASE8_RESULT_JSON=" + json.dumps(output, indent=2, sort_keys=True, default=str))
    raise RuntimeError(f"Feature validation failed: {feature_failures}")

# Fraud labels are read only after the feature-only table has been written and validated.
label_snapshot = source_table_snapshot(["fraud_labels"])
source_snapshot.update(label_snapshot)
if label_snapshot["fraud_labels"].get("count") != EXPECTED_SOURCE_COUNTS["fraud_labels"]:
    source_count_failures.append("fraud_labels")

final_labels, label_multiplicity = build_final_labels()
training_rows = build_training_rows(feature_table_df, final_labels).cache()
count_rows(training_rows)
training_merge_result = merge_to_delta(TRAINING_TABLE, training_rows, ["transaction_id"], "_training_row_hash")
training_table_df = spark.table(TRAINING_TABLE).cache()
count_rows(training_table_df)
training_quality = validate_training_table(training_table_df, feature_table_df)
training_failures = failed_training_checks(training_quality)

feature_statistics = numeric_stats(feature_table_df, NUMERIC_FEATURE_COLUMNS)
categorical_statistics = categorical_counts(feature_table_df, CATEGORICAL_FEATURE_COLUMNS)
training_categorical_statistics = categorical_counts(training_table_df, ["target_is_fraud", "investigation_status", "fraud_type"])
parity_baseline = point_in_time_audit_sample(feature_table_df, training_table_df)

omitted_candidate_features = [
    {"feature_name": "mfa_failures_last_10_minutes", "reason": "The Phase 8 requested authentication outputs define success/failure by approved login_success semantics. MFA-specific semantics exist but no approved output name was requested."},
    {"feature_name": "password_reset_before_transaction", "reason": "password_reset_flag exists, but no requested window length or default policy was approved for this phase."},
    {"feature_name": "transaction_velocity_vs_normal", "reason": "normal_transaction_frequency_per_day exists, but no approved formula was requested for comparing rolling counts to profile frequency."},
    {"feature_name": "destination_wallet_risk_score", "reason": "Silver wallets contain risk_level as a categorical value, not an approved numeric risk score. No fake score was generated."},
    {"feature_name": "wallet_recent_fraud_count", "reason": "Would require prior confirmed labels. Labels are deliberately excluded from transaction_features_offline in this phase."},
    {"feature_name": "transaction_amount_adjusted_for_market_context", "reason": "No approved formula exists in the contracts for market-adjusted amount. Current amount and point-in-time market context were retained separately."},
]

point_in_time_rules = [
    "feature_timestamp equals customer_transactions.event_timestamp.",
    "Historical transaction, wallet, and authentication windows use source_event_timestamp < feature_timestamp.",
    "Events with identical timestamps are not treated as prior events.",
    "Rolling windows are immediately before feature_timestamp: 5 minutes, 10 minutes, 1 hour, or 24 hours as named.",
    "Market features use the latest completed candle where candle_end_timestamp <= feature_timestamp.",
    "Fraud labels are not read for target attachment until after transaction_features_offline is written and validated.",
]

default_null_policies = {
    "missing_historical_counts": "0",
    "missing_historical_sums": "0",
    "missing_avg_or_max": "null with has_prior_tx_24h flag",
    "missing_time_since_previous_transaction": "null with has_previous_transaction = 0",
    "missing_time_since_successful_authentication": "null with has_previous_successful_auth = 0",
    "missing_or_future_account_profile": "profile-derived values null with account_profile_available = 0",
    "zero_or_null_normal_transaction_amount": "ratio and amount-above-normal null with normal_transaction_amount_available = 0",
    "missing_or_future_device_first_seen": "device ages null with device_profile_available = 0",
    "non_applicable_source_wallet": "source wallet count defaults to 0 with source_wallet_applicable = 0",
    "non_applicable_destination_wallet": "destination wallet age and is_new flag null/defaulted with destination_wallet_applicable = 0",
    "missing_market_data": "market numeric values null with market_data_available = 0",
    "missing_boolean_flags": "0 unless the feature is explicitly not applicable, where null is retained",
}

failed_checks = []
failed_checks.extend([f"source_count_{dataset}" for dataset in source_count_failures])
failed_checks.extend(feature_failures)
failed_checks.extend(training_failures)

output = {
    "notebook": "03_offline_feature_engineering",
    "phase": "8",
    "run_id": RUN_ID,
    "cluster_id": CLUSTER_ID,
    "catalog": CATALOG,
    "source_schema": SOURCE_SCHEMA,
    "target_schema": FEATURE_SCHEMA,
    "feature_table": FEATURE_TABLE,
    "training_table": TRAINING_TABLE,
    "feature_version": FEATURE_VERSION,
    "night_transaction_utc_hour_range": {"inclusive_start_hour": NIGHT_START_HOUR_UTC, "inclusive_end_hour": NIGHT_END_HOUR_UTC},
    "source_silver_snapshot": source_snapshot,
    "feature_columns": {
        "identity_and_timing": IDENTITY_AND_TIMING_COLUMNS,
        "current_transaction": CURRENT_TRANSACTION_FEATURES,
        "account": ACCOUNT_FEATURES,
        "historical_transaction_velocity": TRANSACTION_HISTORY_FEATURES,
        "authentication": AUTHENTICATION_FEATURES,
        "device_authentication": DEVICE_AUTH_FEATURES,
        "device": DEVICE_FEATURES,
        "wallet": WALLET_FEATURES,
        "market": MARKET_FEATURES,
        "metadata": FEATURE_METADATA_COLUMNS,
        "model_input_candidates": MODEL_INPUT_CANDIDATE_COLUMNS,
        "non_model_columns": [column_name for column_name in FEATURE_TABLE_COLUMNS if column_name not in MODEL_INPUT_CANDIDATE_COLUMNS],
    },
    "feature_column_count": len(FEATURE_TABLE_COLUMNS),
    "model_input_candidate_count": len(MODEL_INPUT_CANDIDATE_COLUMNS),
    "feature_merge": feature_merge_result,
    "feature_quality": feature_quality,
    "feature_statistics": feature_statistics,
    "categorical_statistics": categorical_statistics,
    "leakage_checks": leakage,
    "market_data_availability": {
        "available_count": count_rows(feature_table_df.filter(F.col("market_data_available") == F.lit(1))),
        "unavailable_count": count_rows(feature_table_df.filter(F.col("market_data_available") == F.lit(0))),
    },
    "label_multiplicity": label_multiplicity,
    "training_merge": training_merge_result,
    "training_quality": training_quality,
    "training_categorical_statistics": training_categorical_statistics,
    "labeled_count": training_quality["row_count"],
    "unlabeled_count": training_quality["unlabeled_feature_rows"],
    "omitted_candidate_features": omitted_candidate_features,
    "point_in_time_rules": point_in_time_rules,
    "default_null_policies": default_null_policies,
    "parity_baseline": parity_baseline,
    "failed_checks": failed_checks,
    "overall_status": "PASS" if not failed_checks else "FAIL",
}

result_json = json.dumps(output, indent=2, sort_keys=True, default=str)
print("PHASE8_RESULT_JSON=" + result_json)
dbutils.notebook.exit(result_json)
if failed_checks:
    raise RuntimeError(f"Phase 8 checks failed: {failed_checks}")