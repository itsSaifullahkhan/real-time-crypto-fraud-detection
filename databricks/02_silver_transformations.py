# Databricks notebook source
# MAGIC %md
# MAGIC # 02 Silver Transformations

# COMMAND ----------

import json
import uuid
from collections import OrderedDict
from decimal import Decimal
from typing import Any

from delta.tables import DeltaTable
from pyspark.sql import DataFrame, Window
from pyspark.sql import functions as F

spark.conf.set("spark.sql.session.timeZone", "UTC")
spark.conf.set("spark.sql.legacy.parquet.nanosAsLong", "true")

CATALOG = "crypto_fraud"
BRONZE_SCHEMA = "bronze"
SILVER_SCHEMA = "silver"
QUARANTINE_BASE_PATH = "/Volumes/crypto_fraud/quarantine/rejected_files/silver"
SILVER_RUN_ID = str(uuid.uuid4())

BRONZE_METADATA_COLUMNS = [
    "_ingested_at",
    "_source_file",
    "_source_modification_time",
    "_ingestion_run_id",
    "_rescued_data",
]
SILVER_METADATA_COLUMNS = [
    "_silver_processed_at",
    "_record_hash",
    "_quality_status",
]
QUARANTINE_METADATA_COLUMNS = [
    "_quarantined_at",
    "_quarantine_reason",
    "_source_bronze_table",
    "_silver_run_id",
]

UUID_REGEX = r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[1-5][0-9a-fA-F]{3}-[89abAB][0-9a-fA-F]{3}-[0-9a-fA-F]{12}$"
COUNTRY_REGEX = r"^[A-Z]{2}$"
IP_ADDRESS_REGEX = r"^(([0-9]{1,3}\.){3}[0-9]{1,3}|[0-9A-Fa-f:]+)$"

ALLOWED_ASSETS = ["BTC", "ETH"]
ALLOWED_PRODUCTS = ["BTC-USD", "ETH-USD"]
ALLOWED_ACCOUNT_STATUS = ["ACTIVE", "SUSPENDED", "CLOSED"]
ALLOWED_KYC_LEVELS = ["BASIC", "STANDARD", "ENHANCED"]
ALLOWED_RISK_TIERS = ["LOW", "MEDIUM", "HIGH"]
ALLOWED_WALLET_RISK = ["LOW", "MEDIUM", "HIGH", "UNKNOWN"]
ALLOWED_WALLET_TYPES = ["CUSTOMER", "EXTERNAL"]
ALLOWED_DEVICE_TYPES = ["MOBILE", "DESKTOP", "TABLET"]
ALLOWED_OPERATING_SYSTEMS = ["ANDROID", "IOS", "WINDOWS", "MACOS", "LINUX", "OTHER"]
ALLOWED_FAILURE_REASONS = ["INVALID_PASSWORD", "MFA_FAILED", "ACCOUNT_LOCKED", "DEVICE_BLOCKED", "OTHER"]
ALLOWED_TRANSACTION_TYPES = ["DEPOSIT", "WITHDRAWAL", "TRANSFER"]
ALLOWED_TRANSACTION_STATUS = ["PENDING", "COMPLETED", "FAILED"]
ALLOWED_FRAUD_TYPES = [
    "ACCOUNT_TAKEOVER",
    "HIGH_TRANSACTION_VELOCITY",
    "UNUSUAL_TRANSACTION_AMOUNT",
    "STRUCTURING",
    "MULE_ACCOUNT_ACTIVITY",
    "SHARED_SUSPICIOUS_DEVICE",
    "HIGH_VOLATILITY_UNUSUAL_WITHDRAWAL",
]
ALLOWED_INVESTIGATION_STATUS = ["CONFIRMED_FRAUD", "CLEARED", "INCONCLUSIVE"]


DATASETS: OrderedDict[str, dict[str, Any]] = OrderedDict(
    [
        (
            "accounts",
            {
                "expected_bronze_rows": 100,
                "business_key": ["account_id"],
                "business_columns": [
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
                "required_non_null": [
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
                "string_columns": [
                    "account_id",
                    "schema_version",
                    "home_country",
                    "kyc_level",
                    "customer_risk_tier",
                    "account_status",
                ],
                "upper_columns": ["home_country", "kyc_level", "customer_risk_tier", "account_status"],
                "timestamp_columns": ["created_at", "updated_at"],
                "decimal_columns": {"normal_transaction_amount_usd": "decimal(20,8)"},
                "double_columns": ["normal_transaction_frequency_per_day"],
                "array_upper_columns": ["preferred_assets"],
                "uuid_columns": ["account_id"],
                "country_columns": ["home_country"],
                "enums": {
                    "schema_version": ["1.0"],
                    "kyc_level": ALLOWED_KYC_LEVELS,
                    "customer_risk_tier": ALLOWED_RISK_TIERS,
                    "account_status": ALLOWED_ACCOUNT_STATUS,
                },
                "array_enums": {"preferred_assets": ALLOWED_ASSETS},
                "array_min_max": {"preferred_assets": (1, 2)},
            },
        ),
        (
            "devices",
            {
                "expected_bronze_rows": 141,
                "business_key": ["device_id"],
                "business_columns": [
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
                "required_non_null": [
                    "device_id",
                    "schema_version",
                    "first_seen_at",
                    "last_seen_at",
                    "device_type",
                    "operating_system",
                    "is_trusted",
                    "device_country",
                ],
                "nullable_empty_to_null": ["primary_account_id"],
                "string_columns": [
                    "device_id",
                    "schema_version",
                    "device_type",
                    "operating_system",
                    "device_country",
                    "primary_account_id",
                ],
                "upper_columns": ["device_type", "operating_system", "device_country"],
                "timestamp_columns": ["first_seen_at", "last_seen_at"],
                "uuid_columns": ["device_id"],
                "nullable_uuid_columns": ["primary_account_id"],
                "country_columns": ["device_country"],
                "enums": {
                    "schema_version": ["1.0"],
                    "device_type": ALLOWED_DEVICE_TYPES,
                    "operating_system": ALLOWED_OPERATING_SYSTEMS,
                },
            },
        ),
        (
            "wallets",
            {
                "expected_bronze_rows": 342,
                "business_key": ["wallet_id"],
                "business_columns": [
                    "wallet_id",
                    "schema_version",
                    "owner_account_id",
                    "wallet_type",
                    "first_seen_at",
                    "risk_level",
                    "is_known_destination",
                    "supported_assets",
                ],
                "required_non_null": [
                    "wallet_id",
                    "schema_version",
                    "wallet_type",
                    "first_seen_at",
                    "risk_level",
                    "is_known_destination",
                    "supported_assets",
                ],
                "nullable_empty_to_null": ["owner_account_id"],
                "string_columns": [
                    "wallet_id",
                    "schema_version",
                    "owner_account_id",
                    "wallet_type",
                    "risk_level",
                ],
                "upper_columns": ["wallet_type", "risk_level"],
                "timestamp_columns": ["first_seen_at"],
                "array_upper_columns": ["supported_assets"],
                "uuid_columns": ["wallet_id"],
                "nullable_uuid_columns": ["owner_account_id"],
                "enums": {
                    "schema_version": ["1.0"],
                    "wallet_type": ALLOWED_WALLET_TYPES,
                    "risk_level": ALLOWED_WALLET_RISK,
                },
                "array_enums": {"supported_assets": ALLOWED_ASSETS},
                "array_min_max": {"supported_assets": (1, 2)},
            },
        ),
        (
            "authentication_events",
            {
                "expected_bronze_rows": 2500,
                "business_key": ["event_id"],
                "business_columns": [
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
                    "event_date",
                ],
                "required_non_null": [
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
                    "event_date",
                ],
                "nullable_empty_to_null": ["failure_reason"],
                "string_columns": [
                    "event_id",
                    "event_type",
                    "schema_version",
                    "source",
                    "login_id",
                    "account_id",
                    "device_id",
                    "country",
                    "ip_address",
                    "failure_reason",
                ],
                "upper_columns": ["country", "failure_reason"],
                "lower_columns": ["event_type", "source"],
                "timestamp_columns": ["event_timestamp", "source_timestamp", "ingestion_timestamp"],
                "date_columns": ["event_date"],
                "uuid_columns": ["event_id", "login_id", "account_id", "device_id"],
                "country_columns": ["country"],
                "ip_columns": ["ip_address"],
                "enums": {
                    "event_type": ["authentication_event"],
                    "schema_version": ["1.0"],
                    "source": ["authentication_generator"],
                    "failure_reason": ALLOWED_FAILURE_REASONS,
                },
                "nullable_enum_columns": ["failure_reason"],
                "date_derivations": {"event_date": "event_timestamp"},
            },
        ),
        (
            "customer_transactions",
            {
                "expected_bronze_rows": 5000,
                "business_key": ["transaction_id"],
                "business_columns": [
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
                    "event_date",
                ],
                "required_non_null": [
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
                    "device_id",
                    "country",
                    "market_price_usd",
                    "transaction_amount_usd",
                    "transaction_status",
                    "event_date",
                ],
                "nullable_empty_to_null": ["source_wallet_id", "destination_wallet_id"],
                "string_columns": [
                    "event_id",
                    "event_type",
                    "schema_version",
                    "source",
                    "transaction_id",
                    "account_id",
                    "asset",
                    "transaction_type",
                    "source_wallet_id",
                    "destination_wallet_id",
                    "device_id",
                    "country",
                    "transaction_status",
                ],
                "upper_columns": ["asset", "transaction_type", "country", "transaction_status"],
                "lower_columns": ["event_type", "source"],
                "timestamp_columns": ["event_timestamp", "source_timestamp", "ingestion_timestamp"],
                "date_columns": ["event_date"],
                "decimal_columns": {
                    "crypto_quantity": "decimal(20,8)",
                    "market_price_usd": "decimal(20,8)",
                    "transaction_amount_usd": "decimal(20,8)",
                },
                "uuid_columns": ["event_id", "transaction_id", "account_id", "device_id"],
                "nullable_uuid_columns": ["source_wallet_id", "destination_wallet_id"],
                "country_columns": ["country"],
                "enums": {
                    "event_type": ["customer_transaction"],
                    "schema_version": ["1.0"],
                    "source": ["historical_customer_generator", "realtime_customer_generator"],
                    "asset": ALLOWED_ASSETS,
                    "transaction_type": ALLOWED_TRANSACTION_TYPES,
                    "transaction_status": ALLOWED_TRANSACTION_STATUS,
                },
                "date_derivations": {"event_date": "event_timestamp"},
            },
        ),
        (
            "market_candles",
            {
                "expected_bronze_rows": 20158,
                "business_key": ["product_id", "candle_start_timestamp"],
                "business_columns": [
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
                    "event_date",
                ],
                "required_non_null": [
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
                    "event_date",
                ],
                "string_columns": ["schema_version", "source", "product_id"],
                "upper_columns": ["product_id"],
                "lower_columns": ["source"],
                "timestamp_columns": ["candle_start_timestamp", "candle_end_timestamp", "retrieved_at"],
                "date_columns": ["event_date"],
                "decimal_columns": {
                    "open_price_usd": "decimal(20,8)",
                    "high_price_usd": "decimal(20,8)",
                    "low_price_usd": "decimal(20,8)",
                    "close_price_usd": "decimal(20,8)",
                    "volume": "decimal(20,8)",
                },
                "long_columns": ["granularity_seconds"],
                "enums": {
                    "schema_version": ["1.0"],
                    "source": ["coinbase_exchange_rest_api"],
                    "product_id": ALLOWED_PRODUCTS,
                },
                "date_derivations": {"event_date": "candle_start_timestamp"},
            },
        ),
        (
            "fraud_labels",
            {
                "expected_bronze_rows": 4479,
                "business_key": ["event_id"],
                "business_columns": [
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
                    "label_date",
                ],
                "required_non_null": [
                    "event_id",
                    "event_type",
                    "schema_version",
                    "source",
                    "event_timestamp",
                    "source_timestamp",
                    "ingestion_timestamp",
                    "transaction_id",
                    "is_fraud",
                    "label_timestamp",
                    "label_source",
                    "investigation_status",
                    "label_date",
                ],
                "nullable_empty_to_null": ["fraud_type"],
                "string_columns": [
                    "event_id",
                    "event_type",
                    "schema_version",
                    "source",
                    "transaction_id",
                    "fraud_type",
                    "label_source",
                    "investigation_status",
                ],
                "upper_columns": ["fraud_type", "label_source", "investigation_status"],
                "lower_columns": ["event_type", "source"],
                "timestamp_columns": ["event_timestamp", "source_timestamp", "ingestion_timestamp", "label_timestamp"],
                "date_columns": ["label_date"],
                "uuid_columns": ["event_id", "transaction_id"],
                "enums": {
                    "event_type": ["fraud_label"],
                    "schema_version": ["1.0"],
                    "source": ["simulated_investigation"],
                    "fraud_type": ALLOWED_FRAUD_TYPES,
                    "label_source": ["SIMULATED_INVESTIGATION"],
                    "investigation_status": ALLOWED_INVESTIGATION_STATUS,
                },
                "nullable_enum_columns": ["fraud_type"],
                "date_derivations": {"label_date": "label_timestamp"},
            },
        ),
    ]
)


def full_table(schema_name: str, dataset: str) -> str:
    return f"{CATALOG}.{schema_name}.{dataset}"


def quote_identifier(name: str) -> str:
    return ".".join(f"`{part}`" for part in name.split("."))


def quarantine_path(dataset: str) -> str:
    return f"{QUARANTINE_BASE_PATH}/{dataset}/"


def table_exists(table_name: str) -> bool:
    return spark.catalog.tableExists(table_name)


def decimal_type(type_text: str) -> str:
    return type_text.lower()


def cast_decimal(column_name: str, type_text: str):
    return F.col(column_name).cast(decimal_type(type_text))


def epoch_to_utc_timestamp(raw_column_name: str):
    quoted = f"`{raw_column_name}`"
    return F.expr(
        f"""
        CASE
          WHEN {quoted} IS NULL THEN CAST(NULL AS TIMESTAMP)
          WHEN ABS(CAST({quoted} AS DECIMAL(38,0))) >= CAST(100000000000000000 AS DECIMAL(38,0))
            THEN timestamp_micros(CAST(FLOOR(CAST({quoted} AS DECIMAL(38,0)) / 1000) AS BIGINT))
          WHEN ABS(CAST({quoted} AS DECIMAL(38,0))) >= CAST(100000000000000 AS DECIMAL(38,0))
            THEN timestamp_micros(CAST({quoted} AS BIGINT))
          WHEN ABS(CAST({quoted} AS DECIMAL(38,0))) >= CAST(100000000000 AS DECIMAL(38,0))
            THEN timestamp_millis(CAST({quoted} AS BIGINT))
          ELSE timestamp_seconds(CAST({quoted} AS BIGINT))
        END
        """
    )


def observed_epoch_unit(df: DataFrame, raw_column_name: str) -> str:
    row = df.select(F.max(F.abs(F.col(raw_column_name).cast("decimal(38,0)"))).alias("max_abs")).collect()[0]
    max_abs = row["max_abs"]
    if max_abs is None:
        return "none"
    max_abs = Decimal(max_abs)
    if max_abs >= Decimal("100000000000000000"):
        return "nanoseconds"
    if max_abs >= Decimal("100000000000000"):
        return "microseconds"
    if max_abs >= Decimal("100000000000"):
        return "milliseconds"
    return "seconds"


def normalize_string_column(df: DataFrame, column_name: str, config: dict[str, Any]) -> DataFrame:
    value = F.trim(F.col(column_name))
    if column_name in config.get("upper_columns", []):
        value = F.upper(value)
    if column_name in config.get("lower_columns", []):
        value = F.lower(value)
    if column_name in config.get("nullable_empty_to_null", []):
        value = F.when(F.length(value) == 0, F.lit(None)).otherwise(value)
    return df.withColumn(column_name, value)


def normalize_array_column(df: DataFrame, column_name: str) -> DataFrame:
    return df.withColumn(column_name, F.expr(f"transform(`{column_name}`, x -> upper(trim(x)))"))


def add_initial_quality_reasons(df: DataFrame, reason_conditions: list[tuple[str, Any]]) -> DataFrame:
    reason_columns = [
        F.when(condition, F.lit(reason)).otherwise(F.lit(None).cast("string"))
        for reason, condition in reason_conditions
    ]
    if not reason_columns:
        return df.withColumn("_quality_reasons", F.array())
    return df.withColumn("_quality_reasons", F.array(*reason_columns)).withColumn(
        "_quality_reasons",
        F.expr("filter(_quality_reasons, reason -> reason is not null)"),
    )


def add_quality_reason(df: DataFrame, condition: Any, reason: str) -> DataFrame:
    return df.withColumn(
        "_quality_reasons",
        F.when(condition, F.array_union(F.col("_quality_reasons"), F.array(F.lit(reason)))).otherwise(
            F.col("_quality_reasons")
        ),
    )


def rescued_data_present_condition():
    rescued = F.col("_rescued_data").cast("string")
    return rescued.isNotNull() & (F.length(F.trim(rescued)) > 0) & (rescued != "{}")


def required_condition(column_name: str, config: dict[str, Any]):
    if column_name in config.get("array_upper_columns", []):
        return F.col(column_name).isNull() | (F.size(F.col(column_name)) == 0)
    if column_name in config.get("string_columns", []):
        return F.col(column_name).isNull() | (F.length(F.trim(F.col(column_name))) == 0)
    return F.col(column_name).isNull()


def normalize_dataset(dataset: str, config: dict[str, Any]) -> tuple[DataFrame, dict[str, str]]:
    bronze_table = full_table(BRONZE_SCHEMA, dataset)
    df = spark.table(bronze_table).select(*config["business_columns"], *BRONZE_METADATA_COLUMNS)

    timestamp_units = {}
    for column_name in config.get("timestamp_columns", []):
        raw_column_name = f"_raw_{column_name}"
        df = df.withColumn(raw_column_name, F.col(column_name))
        timestamp_units[column_name] = observed_epoch_unit(df, raw_column_name)
        df = df.withColumn(column_name, epoch_to_utc_timestamp(raw_column_name))

    for column_name in config.get("date_columns", []):
        raw_column_name = f"_raw_{column_name}"
        df = df.withColumn(raw_column_name, F.col(column_name))
        df = df.withColumn(column_name, F.col(raw_column_name).cast("date"))

    for column_name in config.get("string_columns", []):
        df = normalize_string_column(df, column_name, config)

    for column_name in config.get("array_upper_columns", []):
        df = normalize_array_column(df, column_name)

    for column_name, type_text in config.get("decimal_columns", {}).items():
        raw_column_name = f"_raw_{column_name}"
        df = df.withColumn(raw_column_name, F.col(column_name))
        df = df.withColumn(column_name, cast_decimal(column_name, type_text))

    for column_name in config.get("double_columns", []):
        raw_column_name = f"_raw_{column_name}"
        df = df.withColumn(raw_column_name, F.col(column_name))
        df = df.withColumn(column_name, F.col(column_name).cast("double"))

    for column_name in config.get("long_columns", []):
        raw_column_name = f"_raw_{column_name}"
        df = df.withColumn(raw_column_name, F.col(column_name))
        df = df.withColumn(column_name, F.col(column_name).cast("bigint"))

    reason_conditions: list[tuple[str, Any]] = []
    for column_name in config.get("required_non_null", []):
        reason_conditions.append((f"{column_name}_required", required_condition(column_name, config)))

    for column_name in config.get("uuid_columns", []):
        reason_conditions.append(
            (f"{column_name}_uuid_format", F.col(column_name).isNotNull() & (~F.col(column_name).rlike(UUID_REGEX)))
        )
    for column_name in config.get("nullable_uuid_columns", []):
        reason_conditions.append(
            (f"{column_name}_uuid_format", F.col(column_name).isNotNull() & (~F.col(column_name).rlike(UUID_REGEX)))
        )
    for column_name in config.get("country_columns", []):
        reason_conditions.append(
            (f"{column_name}_country_code_format", F.col(column_name).isNotNull() & (~F.col(column_name).rlike(COUNTRY_REGEX)))
        )
    for column_name in config.get("ip_columns", []):
        reason_conditions.append(
            (f"{column_name}_ip_address_format", F.col(column_name).isNotNull() & (~F.col(column_name).rlike(IP_ADDRESS_REGEX)))
        )

    nullable_enums = set(config.get("nullable_enum_columns", []))
    for column_name, allowed_values in config.get("enums", {}).items():
        condition = ~F.col(column_name).isin(allowed_values)
        if column_name in nullable_enums:
            condition = F.col(column_name).isNotNull() & condition
        else:
            condition = F.col(column_name).isNotNull() & condition
        reason_conditions.append((f"{column_name}_approved_value", condition))

    for column_name, allowed_values in config.get("array_enums", {}).items():
        allowed_sql = ", ".join(f"'{value}'" for value in allowed_values)
        reason_conditions.append(
            (
                f"{column_name}_approved_values",
                F.col(column_name).isNotNull()
                & F.expr(f"exists(`{column_name}`, value -> value is null OR NOT (value IN ({allowed_sql})))"),
            )
        )
    for column_name, (min_items, max_items) in config.get("array_min_max", {}).items():
        reason_conditions.append(
            (
                f"{column_name}_array_size",
                F.col(column_name).isNotNull()
                & ((F.size(F.col(column_name)) < min_items) | (F.size(F.col(column_name)) > max_items)),
            )
        )
        reason_conditions.append(
            (
                f"{column_name}_unique_items",
                F.col(column_name).isNotNull()
                & (F.size(F.array_distinct(F.col(column_name))) != F.size(F.col(column_name))),
            )
        )

    for column_name in config.get("timestamp_columns", []):
        reason_conditions.append(
            (
                f"{column_name}_timestamp_conversion",
                F.col(f"_raw_{column_name}").isNotNull() & F.col(column_name).isNull(),
            )
        )
    for column_name in config.get("date_columns", []):
        reason_conditions.append(
            (
                f"{column_name}_date_conversion",
                F.col(f"_raw_{column_name}").isNotNull() & F.col(column_name).isNull(),
            )
        )
    for column_name in config.get("decimal_columns", {}):
        reason_conditions.append(
            (
                f"{column_name}_decimal_conversion",
                F.col(f"_raw_{column_name}").isNotNull() & F.col(column_name).isNull(),
            )
        )
    for column_name in config.get("double_columns", []):
        reason_conditions.append(
            (
                f"{column_name}_double_conversion",
                F.col(f"_raw_{column_name}").isNotNull() & F.col(column_name).isNull(),
            )
        )
    for column_name in config.get("long_columns", []):
        reason_conditions.append(
            (
                f"{column_name}_long_conversion",
                F.col(f"_raw_{column_name}").isNotNull() & F.col(column_name).isNull(),
            )
        )

    reason_conditions.append(("bronze_rescued_data_present", rescued_data_present_condition()))
    df = add_initial_quality_reasons(df, reason_conditions)
    df = add_dataset_specific_quality_rules(dataset, df)

    hash_struct = F.struct(*[F.col(column_name).alias(column_name) for column_name in config["business_columns"]])
    df = df.withColumn("_record_hash", F.sha2(F.to_json(hash_struct), 256))
    return df, timestamp_units


def add_timestamp_order_rule(df: DataFrame, left: str, right: str, reason: str) -> DataFrame:
    return add_quality_reason(
        df,
        F.col(left).isNotNull() & F.col(right).isNotNull() & (F.col(left) > F.col(right)),
        reason,
    )


def add_date_derivation_rules(df: DataFrame, config: dict[str, Any]) -> DataFrame:
    for date_column, timestamp_column in config.get("date_derivations", {}).items():
        df = add_quality_reason(
            df,
            F.col(date_column).isNotNull()
            & F.col(timestamp_column).isNotNull()
            & (F.col(date_column) != F.to_date(F.col(timestamp_column))),
            f"{date_column}_mismatch_from_{timestamp_column}",
        )
    return df


def add_dataset_specific_quality_rules(dataset: str, df: DataFrame) -> DataFrame:
    if dataset == "accounts":
        df = add_quality_reason(
            df,
            F.col("updated_at").isNotNull()
            & F.col("created_at").isNotNull()
            & (F.col("updated_at") < F.col("created_at")),
            "updated_at_before_created_at",
        )
        df = add_quality_reason(
            df,
            F.col("normal_transaction_amount_usd").isNotNull()
            & (F.col("normal_transaction_amount_usd") < F.lit(0).cast("decimal(20,8)")),
            "normal_transaction_amount_usd_negative",
        )
        df = add_quality_reason(
            df,
            F.col("normal_transaction_frequency_per_day").isNotNull()
            & (F.col("normal_transaction_frequency_per_day") < 0),
            "normal_transaction_frequency_per_day_negative",
        )

    if dataset == "devices":
        df = add_quality_reason(
            df,
            F.col("last_seen_at").isNotNull()
            & F.col("first_seen_at").isNotNull()
            & (F.col("last_seen_at") < F.col("first_seen_at")),
            "last_seen_at_before_first_seen_at",
        )

    if dataset == "customer_transactions":
        df = add_timestamp_order_rule(df, "event_timestamp", "source_timestamp", "event_timestamp_after_source_timestamp")
        df = add_timestamp_order_rule(df, "source_timestamp", "ingestion_timestamp", "source_timestamp_after_ingestion_timestamp")
        df = add_quality_reason(
            df,
            F.col("crypto_quantity").isNotNull()
            & (F.col("crypto_quantity") <= F.lit(0).cast("decimal(20,8)")),
            "crypto_quantity_not_positive",
        )
        df = add_quality_reason(
            df,
            F.col("market_price_usd").isNotNull()
            & (F.col("market_price_usd") <= F.lit(0).cast("decimal(20,8)")),
            "market_price_usd_not_positive",
        )
        df = add_quality_reason(
            df,
            F.col("transaction_amount_usd").isNotNull()
            & (F.col("transaction_amount_usd") < F.lit(0).cast("decimal(20,8)")),
            "transaction_amount_usd_negative",
        )
        df = add_quality_reason(
            df,
            F.col("source_wallet_id").isNotNull()
            & F.col("destination_wallet_id").isNotNull()
            & (F.col("source_wallet_id") == F.col("destination_wallet_id")),
            "source_and_destination_wallet_identical",
        )
        df = add_quality_reason(
            df,
            (F.col("transaction_type") == "DEPOSIT") & F.col("destination_wallet_id").isNull(),
            "deposit_missing_destination_wallet_id",
        )
        df = add_quality_reason(
            df,
            (F.col("transaction_type").isin(["WITHDRAWAL", "TRANSFER"])) & F.col("source_wallet_id").isNull(),
            "outgoing_transaction_missing_source_wallet_id",
        )
        df = add_quality_reason(
            df,
            (F.col("transaction_type").isin(["WITHDRAWAL", "TRANSFER"])) & F.col("destination_wallet_id").isNull(),
            "outgoing_transaction_missing_destination_wallet_id",
        )
        amount_delta = F.abs(
            F.col("transaction_amount_usd")
            - (F.col("crypto_quantity") * F.col("market_price_usd")).cast("decimal(20,8)")
        )
        df = add_quality_reason(
            df,
            F.col("transaction_amount_usd").isNotNull()
            & F.col("crypto_quantity").isNotNull()
            & F.col("market_price_usd").isNotNull()
            & (amount_delta > F.lit(Decimal("0.01")).cast("decimal(20,8)")),
            "transaction_amount_mismatch_quantity_times_price",
        )

    if dataset == "authentication_events":
        df = add_timestamp_order_rule(df, "event_timestamp", "source_timestamp", "event_timestamp_after_source_timestamp")
        df = add_timestamp_order_rule(df, "source_timestamp", "ingestion_timestamp", "source_timestamp_after_ingestion_timestamp")

    if dataset == "market_candles":
        zero = F.lit(0).cast("decimal(20,8)")
        for column_name in ["open_price_usd", "high_price_usd", "low_price_usd", "close_price_usd"]:
            df = add_quality_reason(
                df,
                F.col(column_name).isNotNull() & (F.col(column_name) <= zero),
                f"{column_name}_not_positive",
            )
        df = add_quality_reason(df, F.col("volume").isNotNull() & (F.col("volume") < zero), "volume_negative")
        df = add_quality_reason(df, F.col("granularity_seconds").isNotNull() & (F.col("granularity_seconds") != 60), "granularity_not_60_seconds")
        df = add_quality_reason(
            df,
            F.col("candle_start_timestamp").isNotNull()
            & F.col("candle_end_timestamp").isNotNull()
            & (F.col("candle_end_timestamp") != F.expr("candle_start_timestamp + INTERVAL 60 SECONDS")),
            "candle_end_not_start_plus_60_seconds",
        )
        df = add_quality_reason(
            df,
            F.col("high_price_usd").isNotNull()
            & (
                (F.col("high_price_usd") < F.col("open_price_usd"))
                | (F.col("high_price_usd") < F.col("close_price_usd"))
                | (F.col("high_price_usd") < F.col("low_price_usd"))
            ),
            "high_price_below_ohlc_values",
        )
        df = add_quality_reason(
            df,
            F.col("low_price_usd").isNotNull()
            & (
                (F.col("low_price_usd") > F.col("open_price_usd"))
                | (F.col("low_price_usd") > F.col("close_price_usd"))
                | (F.col("low_price_usd") > F.col("high_price_usd"))
            ),
            "low_price_above_ohlc_values",
        )

    if dataset == "fraud_labels":
        df = add_timestamp_order_rule(df, "event_timestamp", "source_timestamp", "event_timestamp_after_source_timestamp")
        df = add_timestamp_order_rule(df, "source_timestamp", "ingestion_timestamp", "source_timestamp_after_ingestion_timestamp")
        df = add_quality_reason(
            df,
            F.col("event_timestamp").isNotNull()
            & F.col("label_timestamp").isNotNull()
            & (F.col("event_timestamp") != F.col("label_timestamp")),
            "event_timestamp_not_equal_label_timestamp",
        )
        df = add_quality_reason(
            df,
            (F.col("is_fraud") == True) & F.col("fraud_type").isNull(),
            "fraud_type_required_for_fraud",
        )
        df = add_quality_reason(
            df,
            (F.col("is_fraud") == False) & F.col("fraud_type").isNotNull(),
            "fraud_type_must_be_null_for_non_fraud",
        )
        df = add_quality_reason(
            df,
            (F.col("investigation_status") == "CONFIRMED_FRAUD") & (F.col("is_fraud") != True),
            "confirmed_fraud_status_requires_is_fraud_true",
        )
        df = add_quality_reason(
            df,
            (F.col("investigation_status") == "CLEARED") & (F.col("is_fraud") != False),
            "cleared_status_requires_is_fraud_false",
        )

    return add_date_derivation_rules(df, DATASETS[dataset])


def add_dedup_status(df: DataFrame, business_key: list[str]) -> DataFrame:
    window = Window.partitionBy(*[F.col(column_name) for column_name in business_key]).orderBy(
        F.col("_ingested_at").desc_nulls_last(),
        F.col("_source_file").desc_nulls_last(),
    )
    return df.withColumn("_dedup_rank", F.row_number().over(window)).withColumn(
        "_duplicate_key_group_size",
        F.count(F.lit(1)).over(Window.partitionBy(*[F.col(column_name) for column_name in business_key])),
    )


def latest_records(df: DataFrame) -> DataFrame:
    return df.filter(F.col("_dedup_rank") == 1)


def duplicate_records(df: DataFrame) -> DataFrame:
    return add_quality_reason(
        df.filter(F.col("_dedup_rank") > 1),
        F.lit(True),
        "duplicate_business_key_non_latest",
    )


def valid_pre_cross(df: DataFrame) -> DataFrame:
    return df.filter(F.size(F.col("_quality_reasons")) == 0)


def add_fk_reason(df: DataFrame, column_name: str, ref_df: DataFrame, ref_column_name: str, reason: str) -> DataFrame:
    ref = ref_df.select(F.col(ref_column_name).alias("_ref_key")).dropDuplicates()
    joined = df.join(F.broadcast(ref), F.col(column_name) == F.col("_ref_key"), "left")
    with_reason = add_quality_reason(
        joined,
        F.col(column_name).isNotNull() & F.col("_ref_key").isNull(),
        reason,
    )
    return with_reason.drop("_ref_key")


def add_wallet_type_reason(
    df: DataFrame,
    wallet_df: DataFrame,
    wallet_id_column: str,
    alias_prefix: str,
    condition: Any,
    reason: str,
) -> DataFrame:
    ref = wallet_df.select(
        F.col("wallet_id").alias(f"_{alias_prefix}_wallet_id"),
        F.col("wallet_type").alias(f"_{alias_prefix}_wallet_type"),
    )
    joined = df.join(F.broadcast(ref), F.col(wallet_id_column) == F.col(f"_{alias_prefix}_wallet_id"), "left")
    with_reason = add_quality_reason(
        joined,
        condition(joined, f"_{alias_prefix}_wallet_type"),
        reason,
    )
    return with_reason.drop(f"_{alias_prefix}_wallet_id", f"_{alias_prefix}_wallet_type")


def apply_cross_table_rules(latest_by_dataset: dict[str, DataFrame]) -> dict[str, DataFrame]:
    output = dict(latest_by_dataset)
    accounts = valid_pre_cross(output["accounts"]).cache()
    accounts.count()

    output["devices"] = add_fk_reason(output["devices"], "primary_account_id", accounts, "account_id", "primary_account_id_missing_in_accounts")
    output["wallets"] = add_fk_reason(output["wallets"], "owner_account_id", accounts, "account_id", "owner_account_id_missing_in_accounts")

    devices = valid_pre_cross(output["devices"]).cache()
    wallets = valid_pre_cross(output["wallets"]).cache()
    devices.count()
    wallets.count()

    output["authentication_events"] = add_fk_reason(
        output["authentication_events"],
        "account_id",
        accounts,
        "account_id",
        "authentication_account_id_missing_in_accounts",
    )
    output["authentication_events"] = add_fk_reason(
        output["authentication_events"],
        "device_id",
        devices,
        "device_id",
        "authentication_device_id_missing_in_devices",
    )
    output["customer_transactions"] = add_fk_reason(
        output["customer_transactions"],
        "account_id",
        accounts,
        "account_id",
        "transaction_account_id_missing_in_accounts",
    )
    output["customer_transactions"] = add_fk_reason(
        output["customer_transactions"],
        "device_id",
        devices,
        "device_id",
        "transaction_device_id_missing_in_devices",
    )
    output["customer_transactions"] = add_fk_reason(
        output["customer_transactions"],
        "source_wallet_id",
        wallets,
        "wallet_id",
        "source_wallet_id_missing_in_wallets",
    )
    output["customer_transactions"] = add_fk_reason(
        output["customer_transactions"],
        "destination_wallet_id",
        wallets,
        "wallet_id",
        "destination_wallet_id_missing_in_wallets",
    )
    output["customer_transactions"] = add_wallet_type_reason(
        output["customer_transactions"],
        wallets,
        "destination_wallet_id",
        "dest",
        lambda _df, wallet_type_col: (F.col("transaction_type") == "DEPOSIT")
        & F.col("destination_wallet_id").isNotNull()
        & (F.col(wallet_type_col) != "CUSTOMER"),
        "deposit_destination_wallet_not_customer_wallet",
    )
    output["customer_transactions"] = add_wallet_type_reason(
        output["customer_transactions"],
        wallets,
        "source_wallet_id",
        "src",
        lambda _df, wallet_type_col: (F.col("transaction_type").isin(["WITHDRAWAL", "TRANSFER"]))
        & F.col("source_wallet_id").isNotNull()
        & (F.col(wallet_type_col) != "CUSTOMER"),
        "outgoing_source_wallet_not_customer_wallet",
    )

    transactions = valid_pre_cross(output["customer_transactions"]).cache()
    transactions.count()

    output["fraud_labels"] = add_fk_reason(
        output["fraud_labels"],
        "transaction_id",
        transactions,
        "transaction_id",
        "fraud_label_transaction_id_missing_in_customer_transactions",
    )
    txn_ref = transactions.select(
        F.col("transaction_id").alias("_transaction_id"),
        F.col("event_timestamp").alias("_transaction_event_timestamp"),
    )
    fraud_joined = output["fraud_labels"].join(
        F.broadcast(txn_ref),
        F.col("transaction_id") == F.col("_transaction_id"),
        "left",
    )
    fraud_joined = add_quality_reason(
        fraud_joined,
        F.col("_transaction_event_timestamp").isNotNull()
        & F.col("label_timestamp").isNotNull()
        & (F.col("label_timestamp") < F.col("_transaction_event_timestamp")),
        "label_timestamp_before_transaction_event_timestamp",
    )
    output["fraud_labels"] = fraud_joined.drop("_transaction_id", "_transaction_event_timestamp")
    return output

def final_silver_columns(config: dict[str, Any]) -> list[str]:
    return config["business_columns"] + BRONZE_METADATA_COLUMNS + SILVER_METADATA_COLUMNS


def final_quarantine_columns(config: dict[str, Any]) -> list[str]:
    return config["business_columns"] + BRONZE_METADATA_COLUMNS + ["_record_hash", "_quality_status"] + QUARANTINE_METADATA_COLUMNS


def prepare_valid_df(df: DataFrame, config: dict[str, Any]) -> DataFrame:
    return (
        df.filter(F.size(F.col("_quality_reasons")) == 0)
        .withColumn("_silver_processed_at", F.current_timestamp())
        .withColumn("_quality_status", F.lit("VALID"))
        .select(*final_silver_columns(config))
    )


def prepare_quarantine_df(df: DataFrame, config: dict[str, Any], dataset: str) -> DataFrame:
    return (
        df.filter(F.size(F.col("_quality_reasons")) > 0)
        .withColumn("_quality_status", F.lit("INVALID"))
        .withColumn("_quarantined_at", F.current_timestamp())
        .withColumn("_quarantine_reason", F.concat_ws(";", F.col("_quality_reasons")))
        .withColumn("_source_bronze_table", F.lit(full_table(BRONZE_SCHEMA, dataset)))
        .withColumn("_silver_run_id", F.lit(SILVER_RUN_ID))
        .select(*final_quarantine_columns(config))
    )


def create_silver_table_if_missing(table_name: str, df: DataFrame) -> bool:
    if table_exists(table_name):
        return False
    df.limit(0).write.format("delta").saveAsTable(table_name)
    return True


def count_rows(df: DataFrame) -> int:
    return int(df.count())


def merge_to_silver(dataset: str, config: dict[str, Any], valid_df: DataFrame) -> dict[str, Any]:
    silver_table = full_table(SILVER_SCHEMA, dataset)
    table_created = create_silver_table_if_missing(silver_table, valid_df)
    target_df = spark.table(silver_table)
    pre_count = count_rows(target_df)

    key_columns = config["business_key"]
    existing_keys = target_df.select(*key_columns, "_record_hash")
    insert_count = count_rows(valid_df.join(existing_keys.select(*key_columns), key_columns, "left_anti"))
    update_condition = None
    joined = valid_df.alias("s").join(existing_keys.alias("t"), key_columns, "inner")
    update_count = count_rows(joined.filter(F.col("s._record_hash") != F.col("t._record_hash")))

    merge_condition = " AND ".join([f"t.`{column_name}` <=> s.`{column_name}`" for column_name in key_columns])
    update_condition = "t.`_record_hash` <> s.`_record_hash`"
    (
        DeltaTable.forName(spark, silver_table)
        .alias("t")
        .merge(valid_df.alias("s"), merge_condition)
        .whenMatchedUpdateAll(condition=update_condition)
        .whenNotMatchedInsertAll()
        .execute()
    )

    final_count = count_rows(spark.table(silver_table))
    detail = spark.sql(f"DESCRIBE DETAIL {quote_identifier(silver_table)}").collect()[0].asDict()
    return {
        "table_created": table_created,
        "pre_silver_count": pre_count,
        "inserted_count": insert_count,
        "updated_count": update_count,
        "final_silver_count": final_count,
        "partitionColumns": detail.get("partitionColumns") or [],
    }


def write_quarantine(dataset: str, config: dict[str, Any], quarantine_df: DataFrame) -> dict[str, Any]:
    path = quarantine_path(dataset)
    invalid_count = count_rows(quarantine_df)
    path_is_delta = False
    try:
        path_is_delta = DeltaTable.isDeltaTable(spark, path)
    except Exception:
        path_is_delta = False
    if invalid_count == 0:
        return {
            "path": path,
            "invalid_count": 0,
            "quarantine_inserted_count": 0,
            "final_quarantine_count": 0 if not path_is_delta else count_rows(spark.read.format("delta").load(path)),
            "written": False,
        }

    if not path_is_delta:
        quarantine_df.write.format("delta").mode("append").save(path)
        final_count = count_rows(spark.read.format("delta").load(path))
        return {
            "path": path,
            "invalid_count": invalid_count,
            "quarantine_inserted_count": invalid_count,
            "final_quarantine_count": final_count,
            "written": True,
        }

    existing = spark.read.format("delta").load(path)
    merge_keys = ["_source_bronze_table", "_record_hash", "_quarantine_reason"]
    inserted_count = count_rows(quarantine_df.join(existing.select(*merge_keys), merge_keys, "left_anti"))
    condition = " AND ".join([f"t.`{column_name}` <=> s.`{column_name}`" for column_name in merge_keys])
    (
        DeltaTable.forPath(spark, path)
        .alias("t")
        .merge(quarantine_df.alias("s"), condition)
        .whenNotMatchedInsertAll()
        .execute()
    )
    final_count = count_rows(spark.read.format("delta").load(path))
    return {
        "path": path,
        "invalid_count": invalid_count,
        "quarantine_inserted_count": inserted_count,
        "final_quarantine_count": final_count,
        "written": inserted_count > 0,
    }


def duplicate_non_latest_count(df: DataFrame) -> int:
    return count_rows(df.filter(F.col("_dedup_rank") > 1))


def count_duplicate_keys_in_table(df: DataFrame, key_columns: list[str]) -> int:
    return count_rows(df.groupBy(*key_columns).count().filter(F.col("count") > 1))


def null_key_count(df: DataFrame, key_columns: list[str]) -> int:
    condition = None
    for column_name in key_columns:
        current = F.col(column_name).isNull()
        condition = current if condition is None else condition | current
    return count_rows(df.filter(condition))


def quality_reason_counts(df: DataFrame) -> dict[str, int]:
    if count_rows(df) == 0:
        return {}
    rows = (
        df.select(F.explode("_quality_reasons").alias("reason"))
        .groupBy("reason")
        .count()
        .orderBy("reason")
        .collect()
    )
    return {row["reason"]: int(row["count"]) for row in rows}


def source_schema_snapshot(schema_name: str) -> dict[str, Any]:
    snapshot = {}
    for dataset in DATASETS:
        full_name = full_table(schema_name, dataset)
        exists = table_exists(full_name)
        snapshot[dataset] = {"table": full_name, "exists": exists}
        if exists:
            df = spark.table(full_name)
            snapshot[dataset].update({"count": count_rows(df), "schema": df.dtypes})
    return snapshot


def build_transformation_matrix(bronze_schema_snapshot: dict[str, Any], silver_schema_snapshot: dict[str, Any]) -> list[dict[str, str]]:
    matrix = []
    for dataset, config in DATASETS.items():
        bronze_types = dict(bronze_schema_snapshot[dataset]["schema"])
        silver_types = dict(silver_schema_snapshot[dataset]["schema"])
        for column_name in config["business_columns"]:
            transformation = "preserve value"
            rule = "required/contract compatibility"
            if column_name in config.get("string_columns", []):
                transformation = "trim whitespace"
                if column_name in config.get("upper_columns", []):
                    transformation += "; uppercase normalization"
                if column_name in config.get("lower_columns", []):
                    transformation += "; lowercase normalization"
                if column_name in config.get("nullable_empty_to_null", []):
                    transformation += "; empty string to null"
            if column_name in config.get("array_upper_columns", []):
                transformation = "trim and uppercase array elements"
            if column_name in config.get("timestamp_columns", []):
                transformation = "epoch integer to UTC timestamp"
                rule = "non-null conversion and timestamp ordering where applicable"
            if column_name in config.get("date_columns", []):
                transformation = "cast to date and validate against canonical timestamp"
                rule = "partition date must match derived UTC date"
            if column_name in config.get("decimal_columns", {}):
                transformation = f"cast to {config['decimal_columns'][column_name]}"
                rule = "numeric range and conversion validation"
            if column_name in config.get("double_columns", []):
                transformation = "cast to double"
                rule = "numeric range and conversion validation"
            if column_name in config.get("long_columns", []):
                transformation = "cast to bigint"
                rule = "const/range and conversion validation"
            if column_name in config.get("enums", {}) or column_name in config.get("array_enums", {}):
                rule += "; approved enum values only"
            if column_name in config.get("uuid_columns", []) or column_name in config.get("nullable_uuid_columns", []):
                rule += "; UUID format"
            if column_name in config.get("country_columns", []):
                rule += "; two-letter country code"
            matrix.append(
                {
                    "dataset": dataset,
                    "source_column": column_name,
                    "bronze_data_type": bronze_types.get(column_name, "missing"),
                    "silver_data_type": silver_types.get(column_name, "missing"),
                    "transformation_applied": transformation,
                    "quality_rule_applied": rule,
                }
            )
    return matrix


def cross_table_integrity_results() -> dict[str, int]:
    accounts = spark.table(full_table(SILVER_SCHEMA, "accounts"))
    devices = spark.table(full_table(SILVER_SCHEMA, "devices"))
    wallets = spark.table(full_table(SILVER_SCHEMA, "wallets"))
    auth = spark.table(full_table(SILVER_SCHEMA, "authentication_events"))
    trx = spark.table(full_table(SILVER_SCHEMA, "customer_transactions"))
    labels = spark.table(full_table(SILVER_SCHEMA, "fraud_labels"))

    account_keys = accounts.select("account_id").dropDuplicates()
    device_keys = devices.select("device_id").dropDuplicates()
    wallet_keys = wallets.select("wallet_id").dropDuplicates()
    transaction_keys = trx.select("transaction_id").dropDuplicates()

    return {
        "devices.primary_account_id_missing_in_accounts": count_rows(
            devices.filter(F.col("primary_account_id").isNotNull()).join(
                account_keys,
                "account_id" if False else devices["primary_account_id"] == account_keys["account_id"],
                "left_anti",
            )
        ),
        "wallets.owner_account_id_missing_in_accounts": count_rows(
            wallets.filter(F.col("owner_account_id").isNotNull()).join(
                account_keys,
                wallets["owner_account_id"] == account_keys["account_id"],
                "left_anti",
            )
        ),
        "authentication_events.account_id_missing_in_accounts": count_rows(
            auth.join(account_keys, "account_id", "left_anti")
        ),
        "authentication_events.device_id_missing_in_devices": count_rows(
            auth.join(device_keys, "device_id", "left_anti")
        ),
        "customer_transactions.account_id_missing_in_accounts": count_rows(
            trx.join(account_keys, "account_id", "left_anti")
        ),
        "customer_transactions.device_id_missing_in_devices": count_rows(
            trx.join(device_keys, "device_id", "left_anti")
        ),
        "customer_transactions.source_wallet_id_missing_in_wallets": count_rows(
            trx.filter(F.col("source_wallet_id").isNotNull()).join(
                wallet_keys,
                trx["source_wallet_id"] == wallet_keys["wallet_id"],
                "left_anti",
            )
        ),
        "customer_transactions.destination_wallet_id_missing_in_wallets": count_rows(
            trx.filter(F.col("destination_wallet_id").isNotNull()).join(
                wallet_keys,
                trx["destination_wallet_id"] == wallet_keys["wallet_id"],
                "left_anti",
            )
        ),
        "fraud_labels.transaction_id_missing_in_customer_transactions": count_rows(
            labels.join(transaction_keys, "transaction_id", "left_anti")
        ),
    }


def summarize_final_table(dataset: str, config: dict[str, Any]) -> dict[str, int]:
    table = spark.table(full_table(SILVER_SCHEMA, dataset))
    return {
        "null_primary_keys": null_key_count(table, config["business_key"]),
        "duplicate_business_keys": count_duplicate_keys_in_table(table, config["business_key"]),
        "quality_status_invalid_rows": count_rows(table.filter(F.col("_quality_status") != "VALID")),
    }


spark.sql(f"USE CATALOG {quote_identifier(CATALOG)}")

schema_check = spark.sql(f"SHOW SCHEMAS IN {quote_identifier(CATALOG)} LIKE '{SILVER_SCHEMA}'").collect()
if not schema_check:
    raise RuntimeError(f"Required schema {CATALOG}.{SILVER_SCHEMA} does not exist.")

bronze_schema_snapshot = source_schema_snapshot(BRONZE_SCHEMA)

normalized_by_dataset: dict[str, DataFrame] = {}
latest_by_dataset: dict[str, DataFrame] = {}
duplicate_by_dataset: dict[str, DataFrame] = {}
timestamp_units_by_dataset: dict[str, dict[str, str]] = {}

for dataset, config in DATASETS.items():
    if not table_exists(full_table(BRONZE_SCHEMA, dataset)):
        raise RuntimeError(f"Required Bronze table is missing: {full_table(BRONZE_SCHEMA, dataset)}")
    normalized, timestamp_units = normalize_dataset(dataset, config)
    normalized = add_dedup_status(normalized, config["business_key"]).cache()
    normalized.count()
    normalized_by_dataset[dataset] = normalized
    latest_by_dataset[dataset] = latest_records(normalized).cache()
    duplicate_by_dataset[dataset] = duplicate_records(normalized).cache()
    latest_by_dataset[dataset].count()
    duplicate_by_dataset[dataset].count()
    timestamp_units_by_dataset[dataset] = timestamp_units

latest_with_cross_rules = apply_cross_table_rules(latest_by_dataset)

run_summary = []
type_changes = []
date_mismatch_counts = {}

for dataset, config in DATASETS.items():
    latest_df = latest_with_cross_rules[dataset].cache()
    duplicate_df = duplicate_by_dataset[dataset]
    latest_df.count()

    valid_df = prepare_valid_df(latest_df, config).cache()
    quarantine_df = prepare_quarantine_df(
        latest_df.unionByName(duplicate_df, allowMissingColumns=True),
        config,
        dataset,
    ).cache()
    valid_count = count_rows(valid_df)
    invalid_count = count_rows(quarantine_df)

    silver_result = merge_to_silver(dataset, config, valid_df)
    quarantine_result = write_quarantine(dataset, config, quarantine_df)
    final_checks = summarize_final_table(dataset, config)

    bronze_count = bronze_schema_snapshot[dataset]["count"]
    null_conversion_failures = sum(
        value
        for reason, value in quality_reason_counts(latest_df).items()
        if reason.endswith("_conversion")
    )
    for date_column, timestamp_column in config.get("date_derivations", {}).items():
        reason_name = f"{date_column}_mismatch_from_{timestamp_column}"
        date_mismatch_counts[f"{dataset}.{reason_name}"] = quality_reason_counts(latest_df).get(reason_name, 0)

    row = {
        "dataset": dataset,
        "bronze_table": full_table(BRONZE_SCHEMA, dataset),
        "silver_table": full_table(SILVER_SCHEMA, dataset),
        "bronze_row_count": bronze_count,
        "expected_bronze_rows": config["expected_bronze_rows"],
        "valid_silver_input_count": valid_count,
        "quarantined_count": invalid_count,
        "quarantine_path": quarantine_result["path"],
        "quarantine_inserted_count": quarantine_result["quarantine_inserted_count"],
        "final_quarantine_count": quarantine_result["final_quarantine_count"],
        "inserted_count": silver_result["inserted_count"],
        "updated_count": silver_result["updated_count"],
        "final_silver_count": silver_result["final_silver_count"],
        "table_created": silver_result["table_created"],
        "partitionColumns": silver_result["partitionColumns"],
        "null_primary_keys": final_checks["null_primary_keys"],
        "duplicate_business_keys": final_checks["duplicate_business_keys"],
        "duplicate_rejected_versions": duplicate_non_latest_count(normalized_by_dataset[dataset]),
        "null_conversion_failures": null_conversion_failures,
        "quality_status_invalid_rows": final_checks["quality_status_invalid_rows"],
        "quality_reason_counts": quality_reason_counts(latest_df),
        "timestamp_units": timestamp_units_by_dataset[dataset],
        "status": "PASS",
    }
    failure_indicators = [
        row["bronze_row_count"] != row["expected_bronze_rows"],
        row["quarantined_count"] != 0,
        row["final_silver_count"] != row["expected_bronze_rows"],
        row["null_primary_keys"] != 0,
        row["duplicate_business_keys"] != 0,
        row["null_conversion_failures"] != 0,
        row["quality_status_invalid_rows"] != 0,
    ]
    if any(failure_indicators):
        row["status"] = "FAIL"
    run_summary.append(row)

silver_schema_snapshot = source_schema_snapshot(SILVER_SCHEMA)
for dataset, config in DATASETS.items():
    bronze_types = dict(bronze_schema_snapshot[dataset]["schema"])
    silver_types = dict(silver_schema_snapshot[dataset]["schema"])
    for column_name in config["business_columns"]:
        if bronze_types.get(column_name) != silver_types.get(column_name):
            type_changes.append(
                {
                    "dataset": dataset,
                    "column": column_name,
                    "bronze_type": bronze_types.get(column_name),
                    "silver_type": silver_types.get(column_name),
                }
            )

cross_table_results = cross_table_integrity_results()
cross_table_failure_count = sum(cross_table_results.values())
transformation_matrix = build_transformation_matrix(bronze_schema_snapshot, silver_schema_snapshot)

failed_checks = [
    {
        "dataset": row["dataset"],
        "status": row["status"],
        "quality_reason_counts": row["quality_reason_counts"],
    }
    for row in run_summary
    if row["status"] != "PASS"
]
if cross_table_failure_count:
    failed_checks.append({"dataset": "cross_table_integrity", "status": "FAIL", "failures": cross_table_results})

contract_notes = [
    "The approved account contract and Bronze schema do not contain an email column, so no account email validation was applied.",
    "Fraud labels remain separate in crypto_fraud.silver.fraud_labels and are not joined into silver.customer_transactions.",
    "Silver tables are managed Delta tables with no physical partition columns for this small pilot.",
]

output = {
    "notebook": "02_silver_transformations",
    "silver_run_id": SILVER_RUN_ID,
    "cluster_id": "0803-061312-78fw66xn",
    "bronze_schema_snapshot": bronze_schema_snapshot,
    "silver_schema_snapshot": silver_schema_snapshot,
    "run_summary": run_summary,
    "cross_table_integrity": cross_table_results,
    "cross_table_failure_count": cross_table_failure_count,
    "date_mismatch_counts": date_mismatch_counts,
    "data_type_changes": type_changes,
    "transformation_matrix": transformation_matrix,
    "contract_notes": contract_notes,
    "failed_checks": failed_checks,
    "overall_status": "PASS" if not failed_checks else "FAIL",
}

print(json.dumps(output, indent=2, sort_keys=True, default=str))
dbutils.notebook.exit(json.dumps(output, sort_keys=True, default=str))
