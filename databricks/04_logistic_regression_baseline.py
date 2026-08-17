# Databricks notebook source
# MAGIC %md
# MAGIC # 04 Logistic Regression Baseline
# MAGIC
# MAGIC Phase 9A creates a leakage-safe chronological baseline from the Phase 8
# MAGIC training dataset. This notebook does not train XGBoost, tune thresholds,
# MAGIC register models, serve models, or start any real-time integration work.

# COMMAND ----------

from __future__ import annotations

import hashlib
import importlib
import inspect
import json
import math
import os
import platform
import shutil
import subprocess
import sys
import uuid
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any


PACKAGE_IMPORTS = {
    "pandas": ("pandas", "pandas"),
    "numpy": ("numpy", "numpy"),
    "scikit-learn": ("sklearn", "scikit-learn"),
    "mlflow": ("mlflow", "mlflow"),
    "matplotlib": ("matplotlib", "matplotlib"),
}


def inspect_package(import_name: str) -> dict[str, Any]:
    try:
        module = importlib.import_module(import_name)
        return {
            "available": True,
            "version": getattr(module, "__version__", "unknown"),
            "module": import_name,
        }
    except ImportError as exc:
        return {"available": False, "version": None, "module": import_name, "error": str(exc)}


package_status_before_install = {
    package_name: inspect_package(import_name)
    for package_name, (import_name, _) in PACKAGE_IMPORTS.items()
}

installed_packages: list[str] = []
for package_name, (import_name, install_name) in PACKAGE_IMPORTS.items():
    if package_status_before_install[package_name]["available"]:
        continue
    subprocess.check_call([sys.executable, "-m", "pip", "install", install_name])
    installed_packages.append(install_name)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import sklearn
from mlflow.models.signature import infer_signature
from pyspark.sql import functions as F
from pyspark.sql import types as T
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

try:
    import cloudpickle
except ImportError:
    cloudpickle = None


package_status_after_install = {
    package_name: inspect_package(import_name)
    for package_name, (import_name, _) in PACKAGE_IMPORTS.items()
}

# COMMAND ----------

CATALOG = "crypto_fraud"
FEATURE_SCHEMA = "features"
FEATURE_TABLE = "crypto_fraud.features.transaction_features_offline"
TRAINING_TABLE = "crypto_fraud.features.transaction_training_dataset"
TARGET_COLUMN = "target_is_fraud"
FEATURE_VERSION = "v1"
FAILED_SPLIT_VERSION = "time_split_v1"
SPLIT_VERSION = "rare_event_aware_time_split_v2"
EXPERIMENT_PATH = "/Users/akanaskhan1506@gmail.com/crypto-fraud-phase9a"
WORKSPACE_NOTEBOOK_PATH = "/Users/akanaskhan1506@gmail.com/04_logistic_regression_baseline"
CLUSTER_ID = "0803-061312-78fw66xn"
RUN_ID = str(uuid.uuid4())
CLASSIFICATION_THRESHOLD = 0.5
RANDOM_STATE = 42
EXPECTED_MODEL_CANDIDATE_COUNT = 55
EXPECTED_TARGET_DISTRIBUTION = {"fraud": 40, "normal": 4439}

DEFAULT_CONFIG_PATH = "/Workspace/Users/akanaskhan1506@gmail.com/config/feature_definitions.json"
DEFAULT_OUTPUT_DBFS_DIR = f"dbfs:/FileStore/crypto_fraud_phase9a/{RUN_ID}"

try:
    dbutils.widgets.text("config_path", DEFAULT_CONFIG_PATH)
    dbutils.widgets.text("output_dbfs_dir", DEFAULT_OUTPUT_DBFS_DIR)
except Exception:
    pass


def get_widget(name: str, default: str) -> str:
    try:
        value = dbutils.widgets.get(name)
        return value or default
    except Exception:
        return default


CONFIG_PATH = get_widget("config_path", DEFAULT_CONFIG_PATH)
OUTPUT_DBFS_DIR = get_widget("output_dbfs_dir", DEFAULT_OUTPUT_DBFS_DIR)


def dbfs_parent(path_value: str) -> str:
    return path_value.rstrip("/").rsplit("/", 1)[0]


OUTPUT_ROOT = Path("/tmp") / "crypto_fraud_phase9a" / RUN_ID
REPORTS_DIR = OUTPUT_ROOT / "reports"
PLOTS_DIR = REPORTS_DIR / "phase9a_plots"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

spark.conf.set("spark.sql.session.timeZone", "UTC")
spark.conf.set("spark.sql.legacy.parquet.nanosAsLong", "true")


def sync_reports_to_dbfs() -> dict[str, Any]:
    workspace_report_root = Path("/Workspace/Users/akanaskhan1506@gmail.com/phase9a_reports/phase9a_20260805_01")
    workspace_result: dict[str, Any]
    copied_files = []
    try:
        for local_file in REPORTS_DIR.rglob("*"):
            if not local_file.is_file():
                continue
            relative_path = local_file.relative_to(REPORTS_DIR)
            target_path = workspace_report_root / relative_path
            target_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(local_file, target_path)
            copied_files.append(relative_path.as_posix())
        workspace_result = {
            "status": "PASS",
            "workspace_reports_path": "/Users/akanaskhan1506@gmail.com/phase9a_reports/phase9a_20260805_01",
            "copied_file_count": len(copied_files),
            "copied_files": copied_files,
        }
    except Exception as exc:
        workspace_result = {
            "status": "FAIL",
            "workspace_reports_path": "/Users/akanaskhan1506@gmail.com/phase9a_reports/phase9a_20260805_01",
            "error": repr(exc),
        }

    dbfs_result: dict[str, Any] = {
        "status": "SKIPPED",
        "reason": "Public DBFS root is disabled in this workspace; reports are exported through workspace files and MLflow artifacts.",
        "output_dbfs_dir": OUTPUT_DBFS_DIR,
    }
    if OUTPUT_DBFS_DIR.startswith("dbfs:/Volumes/"):
        try:
            dbfs_copied_files = []
            for local_file in REPORTS_DIR.rglob("*"):
                if not local_file.is_file():
                    continue
                relative_path = local_file.relative_to(REPORTS_DIR).as_posix()
                target_path = f"{OUTPUT_DBFS_DIR.rstrip('/')}/reports/{relative_path}"
                dbutils.fs.mkdirs(dbfs_parent(target_path))
                dbutils.fs.cp(f"file:{local_file}", target_path, True)
                dbfs_copied_files.append(relative_path)
            dbfs_result = {
                "status": "PASS",
                "output_dbfs_dir": OUTPUT_DBFS_DIR,
                "reports_dbfs_dir": f"{OUTPUT_DBFS_DIR.rstrip('/')}/reports",
                "copied_file_count": len(dbfs_copied_files),
                "copied_files": dbfs_copied_files,
            }
        except Exception as exc:
            dbfs_result = {
                "status": "FAIL",
                "output_dbfs_dir": OUTPUT_DBFS_DIR,
                "error": repr(exc),
            }

    return {
        "status": "PASS" if workspace_result.get("status") == "PASS" or dbfs_result.get("status") == "PASS" else "FAIL",
        "workspace_files": workspace_result,
        "dbfs": dbfs_result,
    }

# COMMAND ----------


def notebook_path() -> str | None:
    try:
        return (
            dbutils.notebook.entry_point.getDbutils()
            .notebook()
            .getContext()
            .notebookPath()
            .get()
        )
    except Exception:
        return None


def load_feature_contract(path_hint: str) -> tuple[dict[str, Any], str]:
    candidates: list[Path] = []
    if path_hint:
        candidates.append(Path(path_hint))

    current_notebook_path = notebook_path()
    if current_notebook_path:
        workspace_dir = Path("/Workspace" + str(Path(current_notebook_path).parent))
        candidates.extend(
            [
                workspace_dir / "config" / "feature_definitions.json",
                workspace_dir.parent / "config" / "feature_definitions.json",
            ]
        )

    candidates.extend(
        [
            Path("config/feature_definitions.json"),
            Path("../config/feature_definitions.json"),
            Path(DEFAULT_CONFIG_PATH),
        ]
    )

    seen: set[str] = set()
    for candidate in candidates:
        key = str(candidate)
        if key in seen:
            continue
        seen.add(key)
        if candidate.exists():
            with candidate.open("r", encoding="utf-8") as handle:
                return json.load(handle), str(candidate)

    raise FileNotFoundError(
        "Could not locate config/feature_definitions.json. Checked: "
        + ", ".join(str(path) for path in candidates)
    )


feature_contract, feature_contract_path = load_feature_contract(CONFIG_PATH)
definition_by_feature = {
    item["feature_name"]: item for item in feature_contract.get("definitions", [])
}
contract_candidate_features = list(feature_contract.get("model_input_candidates", []))
definition_candidate_features = [
    item["feature_name"]
    for item in feature_contract.get("definitions", [])
    if item.get("model_input_candidate") is True
]
feature_version_from_contract = str(feature_contract.get("feature_version", FEATURE_VERSION))

# COMMAND ----------


def quote_identifier(name: str) -> str:
    return ".".join(f"`{part}`" for part in name.split("."))


def count_rows(df) -> int:
    return int(df.count())


def as_jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): as_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [as_jsonable(item) for item in value]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        if not math.isfinite(float(value)):
            return None
        return float(value)
    if isinstance(value, np.ndarray):
        return [as_jsonable(item) for item in value.tolist()]
    if isinstance(value, (pd.Timestamp, datetime)):
        if pd.isna(value):
            return None
        return value.isoformat()
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, float):
        if not math.isfinite(value):
            return None
        return value
    if hasattr(value, "asDict"):
        return as_jsonable(value.asDict(recursive=True))
    return value


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(as_jsonable(payload), handle, indent=2, sort_keys=True)
        handle.write("\n")


def spark_row_to_dict(row: Any) -> dict[str, Any]:
    return {key: as_jsonable(value) for key, value in row.asDict(recursive=True).items()}


def collect_categorical_counts(df, columns: list[str]) -> dict[str, list[dict[str, Any]]]:
    output: dict[str, list[dict[str, Any]]] = {}
    for column_name in columns:
        if column_name not in df.columns:
            output[column_name] = []
            continue
        rows = (
            df.groupBy(column_name)
            .count()
            .orderBy(F.desc("count"), F.asc_nulls_last(column_name))
            .collect()
        )
        output[column_name] = [
            {"value": as_jsonable(row[column_name]), "count": int(row["count"])}
            for row in rows
        ]
    return output


def normalize_target(value: Any) -> int | None:
    if value is None or pd.isna(value):
        return None
    if isinstance(value, (bool, np.bool_)):
        return int(value)
    if isinstance(value, (int, np.integer)):
        if int(value) in (0, 1):
            return int(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "1"}:
            return 1
        if normalized in {"false", "0"}:
            return 0
    return None


def target_distribution_from_rows(rows: list[Any]) -> dict[str, int]:
    distribution = {"fraud": 0, "normal": 0, "other_or_null": 0}
    for row in rows:
        normalized = normalize_target(row[TARGET_COLUMN])
        if normalized == 1:
            distribution["fraud"] += int(row["count"])
        elif normalized == 0:
            distribution["normal"] += int(row["count"])
        else:
            distribution["other_or_null"] += int(row["count"])
    return distribution


def estimate_row_size_bytes(schema: T.StructType) -> int:
    total = 0
    for field in schema.fields:
        dtype = field.dataType
        if isinstance(dtype, (T.ByteType, T.BooleanType)):
            total += 1
        elif isinstance(dtype, T.ShortType):
            total += 2
        elif isinstance(dtype, (T.IntegerType, T.FloatType, T.DateType)):
            total += 4
        elif isinstance(dtype, (T.LongType, T.DoubleType, T.TimestampType)):
            total += 8
        elif isinstance(dtype, T.DecimalType):
            total += 16
        elif isinstance(dtype, T.StringType):
            total += 64
        else:
            total += 32
    return total


def delta_latest_version(table_name: str) -> dict[str, Any]:
    history = (
        spark.sql(f"DESCRIBE HISTORY {quote_identifier(table_name)}")
        .orderBy(F.desc("version"))
        .limit(1)
        .collect()
    )
    if not history:
        return {}
    row = history[0].asDict(recursive=True)
    return {
        "version": as_jsonable(row.get("version")),
        "timestamp": as_jsonable(row.get("timestamp")),
        "operation": as_jsonable(row.get("operation")),
    }


def distinct_target_values(rows: list[Any]) -> list[Any]:
    return [as_jsonable(row[TARGET_COLUMN]) for row in rows]

# COMMAND ----------

training_delta_before = delta_latest_version(TRAINING_TABLE)
feature_delta_before = delta_latest_version(FEATURE_TABLE)

training_df = spark.table(TRAINING_TABLE)
feature_df = spark.table(FEATURE_TABLE)

training_row_count = count_rows(training_df)
training_column_count = len(training_df.columns)
training_unique_transactions = count_rows(training_df.select("transaction_id").dropDuplicates())
training_duplicate_transactions = training_row_count - training_unique_transactions

schema_report = [
    {"name": field.name, "data_type": field.dataType.simpleString(), "nullable": field.nullable}
    for field in training_df.schema.fields
]
schema_by_column = {field["name"]: field for field in schema_report}

target_rows = training_df.groupBy(TARGET_COLUMN).count().collect()
target_distribution = target_distribution_from_rows(target_rows)
target_distinct_values = distinct_target_values(target_rows)

timestamp_bounds_row = (
    training_df.select(
        F.min("feature_timestamp").alias("min_feature_timestamp"),
        F.max("feature_timestamp").alias("max_feature_timestamp"),
    )
    .collect()[0]
    .asDict()
)

categorical_candidate_features = [
    feature_name
    for feature_name in contract_candidate_features
    if str(definition_by_feature.get(feature_name, {}).get("data_type", "")).lower()
    in {"string", "category"}
]
numeric_candidate_features = [
    feature_name
    for feature_name in contract_candidate_features
    if feature_name not in categorical_candidate_features
]

categorical_distributions = collect_categorical_counts(
    training_df,
    [
        column_name
        for column_name in categorical_candidate_features
        if column_name in training_df.columns
    ]
    + [
        column_name
        for column_name in ["investigation_status", "fraud_type", "label_source"]
        if column_name in training_df.columns
    ],
)

feature_version_distribution = collect_categorical_counts(
    training_df, ["_feature_version"] if "_feature_version" in training_df.columns else []
)

numeric_non_finite: dict[str, dict[str, int]] = {}
numeric_summary: dict[str, dict[str, Any]] = {}
for column_name in numeric_candidate_features:
    if column_name not in training_df.columns:
        numeric_non_finite[column_name] = {
            "missing_from_table": 1,
            "null_count": 0,
            "nan_count": 0,
            "infinite_count": 0,
        }
        continue
    cast_col = F.col(column_name).cast("double")
    row = (
        training_df.select(
            F.sum(F.when(F.col(column_name).isNull(), F.lit(1)).otherwise(F.lit(0))).alias("null_count"),
            F.sum(F.when(F.isnan(cast_col), F.lit(1)).otherwise(F.lit(0))).alias("nan_count"),
            F.sum(
                F.when(
                    (~F.isnan(cast_col))
                    & (F.abs(cast_col) == F.lit(float("inf"))),
                    F.lit(1),
                ).otherwise(F.lit(0))
            ).alias("infinite_count"),
            F.min(cast_col).alias("min"),
            F.max(cast_col).alias("max"),
            F.avg(cast_col).alias("mean"),
            F.stddev(cast_col).alias("stddev"),
        )
        .collect()[0]
        .asDict()
    )
    numeric_non_finite[column_name] = {
        "missing_from_table": 0,
        "null_count": int(row["null_count"] or 0),
        "nan_count": int(row["nan_count"] or 0),
        "infinite_count": int(row["infinite_count"] or 0),
    }
    numeric_summary[column_name] = {
        "null_count": int(row["null_count"] or 0),
        "nan_count": int(row["nan_count"] or 0),
        "infinite_count": int(row["infinite_count"] or 0),
        "min": as_jsonable(row["min"]),
        "max": as_jsonable(row["max"]),
        "mean": as_jsonable(row["mean"]),
        "stddev": as_jsonable(row["stddev"]),
    }

sample_columns = [
    column_name
    for column_name in [
        "transaction_id",
        "account_id",
        "device_id",
        "feature_timestamp",
        TARGET_COLUMN,
        "asset",
        "transaction_type",
        "country",
        "customer_risk_tier",
        "_feature_version",
    ]
    if column_name in training_df.columns
]
sample_rows = [
    spark_row_to_dict(row)
    for row in training_df.select(*sample_columns).orderBy("feature_timestamp", "transaction_id").limit(5).collect()
]

training_rows_missing_in_feature_table = count_rows(
    training_df.select("transaction_id")
    .join(feature_df.select("transaction_id"), "transaction_id", "left_anti")
)

null_transaction_ids = count_rows(training_df.filter(F.col("transaction_id").isNull()))
null_feature_timestamps = count_rows(training_df.filter(F.col("feature_timestamp").isNull()))
null_targets = count_rows(training_df.filter(F.col(TARGET_COLUMN).isNull()))

fraud_label_semantics_checks: dict[str, int] = {}
if "investigation_status" in training_df.columns:
    fraud_label_semantics_checks["confirmed_fraud_status_target_violations"] = count_rows(
        training_df.filter(
            (F.col("investigation_status") == F.lit("CONFIRMED_FRAUD"))
            & (F.col(TARGET_COLUMN) != F.lit(True))
        )
    )
    fraud_label_semantics_checks["cleared_status_target_violations"] = count_rows(
        training_df.filter(
            (F.col("investigation_status") == F.lit("CLEARED"))
            & (F.col(TARGET_COLUMN) != F.lit(False))
        )
    )
if "fraud_type" in training_df.columns:
    fraud_label_semantics_checks["fraud_target_null_fraud_type_count"] = count_rows(
        training_df.filter((F.col(TARGET_COLUMN) == F.lit(True)) & F.col("fraud_type").isNull())
    )
    fraud_label_semantics_checks["normal_target_non_null_fraud_type_count"] = count_rows(
        training_df.filter((F.col(TARGET_COLUMN) == F.lit(False)) & F.col("fraud_type").isNotNull())
    )

# COMMAND ----------

explicit_forbidden_model_inputs = {
    "transaction_id",
    "account_id",
    "device_id",
    "source_wallet_id",
    "destination_wallet_id",
    "feature_timestamp",
    "event_date",
    TARGET_COLUMN,
    "is_fraud",
    "fraud_type",
    "label_timestamp",
    "label_status",
    "label_source",
    "investigation_status",
    "investigation_outcome",
    "investigation_result",
    "scenario_id",
    "scenario_assignment",
    "scenario_assignment_id",
    "scenario_execution_id",
    "fraud_scenario",
    "generator_fraud_flag",
    "generator_is_fraud",
    "_feature_generated_at",
    "_feature_version",
    "_feature_hash",
    "_source_silver_transaction_hash",
    "_source_silver_label_hash",
    "_source_silver_label_event_id",
    "_training_generated_at",
    "_training_row_hash",
}
forbidden_name_fragments = [
    "label_",
    "_label",
    "investigation",
    "scenario",
    "generator_fraud",
    "training_row_hash",
    "ingestion_metadata",
]

feature_candidate_set = set(contract_candidate_features)
definition_candidate_set = set(definition_candidate_features)
missing_candidate_definitions = sorted(feature_candidate_set - set(definition_by_feature))
definition_top_level_mismatch = sorted(feature_candidate_set.symmetric_difference(definition_candidate_set))
candidate_columns_missing_from_table = sorted(feature_candidate_set - set(training_df.columns))
leakage_candidate_hits = sorted(
    [
        column_name
        for column_name in contract_candidate_features
        if column_name in explicit_forbidden_model_inputs
        or any(fragment in column_name.lower() for fragment in forbidden_name_fragments)
    ]
)
excluded_columns = sorted([column_name for column_name in training_df.columns if column_name not in feature_candidate_set])

validation_failed_checks: list[str] = []
if training_row_count <= 0:
    validation_failed_checks.append("training_table_empty")
if training_duplicate_transactions != 0:
    validation_failed_checks.append("duplicate_transaction_ids")
if null_transaction_ids != 0:
    validation_failed_checks.append("null_transaction_ids")
if null_feature_timestamps != 0:
    validation_failed_checks.append("null_feature_timestamps")
if null_targets != 0:
    validation_failed_checks.append("null_targets")
if target_distribution["other_or_null"] != 0:
    validation_failed_checks.append("target_not_binary")
if target_distribution["fraud"] == 0 or target_distribution["normal"] == 0:
    validation_failed_checks.append("target_missing_class")
if training_rows_missing_in_feature_table != 0:
    validation_failed_checks.append("training_transactions_missing_in_feature_table")
if len(contract_candidate_features) != EXPECTED_MODEL_CANDIDATE_COUNT:
    validation_failed_checks.append("model_candidate_count_mismatch")
if feature_version_from_contract != FEATURE_VERSION:
    validation_failed_checks.append("feature_contract_version_mismatch")
if missing_candidate_definitions:
    validation_failed_checks.append("missing_candidate_definitions")
if definition_top_level_mismatch:
    validation_failed_checks.append("definition_top_level_candidate_mismatch")
if candidate_columns_missing_from_table:
    validation_failed_checks.append("candidate_columns_missing_from_table")
if leakage_candidate_hits:
    validation_failed_checks.append("leakage_or_identity_candidate_columns")
if sum(item["nan_count"] for item in numeric_non_finite.values()) != 0:
    validation_failed_checks.append("numeric_nan_values_present")
if sum(item["infinite_count"] for item in numeric_non_finite.values()) != 0:
    validation_failed_checks.append("numeric_infinite_values_present")
if any(value != 0 for value in fraud_label_semantics_checks.values()):
    validation_failed_checks.append("fraud_label_semantics_violations")

if target_distribution["fraud"] != EXPECTED_TARGET_DISTRIBUTION["fraud"] or target_distribution["normal"] != EXPECTED_TARGET_DISTRIBUTION["normal"]:
    validation_distribution_warning = {
        "expected": EXPECTED_TARGET_DISTRIBUTION,
        "actual": {
            "fraud": target_distribution["fraud"],
            "normal": target_distribution["normal"],
        },
        "reason": "The live approved Delta table is treated as source of truth when binary label semantics are valid.",
    }
else:
    validation_distribution_warning = None

required_audit_columns = [
    column_name
    for column_name in [
        "account_id",
        "device_id",
        "_feature_version",
        "label_timestamp",
        "investigation_status",
        "fraud_type",
        "label_source",
        "_training_row_hash",
    ]
    if column_name in training_df.columns
]
selected_columns = []
for column_name in ["transaction_id", "feature_timestamp"] + contract_candidate_features + [TARGET_COLUMN] + required_audit_columns:
    if column_name not in selected_columns:
        selected_columns.append(column_name)

selected_schema = T.StructType([training_df.schema[column_name] for column_name in selected_columns])
estimated_selected_bytes = estimate_row_size_bytes(selected_schema) * training_row_count

training_data_profile = {
    "phase": "9A",
    "source_table": TRAINING_TABLE,
    "supporting_feature_table": FEATURE_TABLE,
    "contract_path": feature_contract_path,
    "contract_name": feature_contract.get("contract_name"),
    "source_row_count": training_row_count,
    "source_column_count": training_column_count,
    "schema": schema_report,
    "sample_rows": sample_rows,
    "unique_transaction_count": training_unique_transactions,
    "duplicate_transaction_count": training_duplicate_transactions,
    "null_transaction_ids": null_transaction_ids,
    "null_feature_timestamps": null_feature_timestamps,
    "null_target_count": null_targets,
    "target_column": TARGET_COLUMN,
    "target_distinct_values": target_distinct_values,
    "target_distribution": target_distribution,
    "target_distribution_warning": validation_distribution_warning,
    "fraud_label_semantics_checks": fraud_label_semantics_checks,
    "feature_version_distribution": feature_version_distribution.get("_feature_version", []),
    "feature_version": feature_version_from_contract,
    "timestamp_range": {
        "min_feature_timestamp": as_jsonable(timestamp_bounds_row["min_feature_timestamp"]),
        "max_feature_timestamp": as_jsonable(timestamp_bounds_row["max_feature_timestamp"]),
    },
    "categorical_distinct_values": categorical_distributions,
    "numeric_nan_and_infinite_counts": numeric_non_finite,
    "numeric_summary": numeric_summary,
    "candidate_feature_count": len(contract_candidate_features),
    "candidate_features": contract_candidate_features,
    "excluded_columns": excluded_columns,
    "selected_column_count_for_pandas_conversion": len(selected_columns),
    "estimated_selected_in_memory_size_bytes": estimated_selected_bytes,
    "estimated_selected_in_memory_size_mb": estimated_selected_bytes / (1024 * 1024),
    "training_rows_missing_in_feature_table": training_rows_missing_in_feature_table,
    "failed_checks": validation_failed_checks,
}

training_data_profile_path = REPORTS_DIR / "phase9a_training_data_profile.json"
candidate_features_path = REPORTS_DIR / "phase9a_candidate_features.json"
excluded_columns_path = REPORTS_DIR / "phase9a_excluded_columns.json"
write_json(training_data_profile_path, training_data_profile)
write_json(
    candidate_features_path,
    {
        "source": "config/feature_definitions.json",
        "candidate_feature_count": len(contract_candidate_features),
        "candidate_features": contract_candidate_features,
    },
)
write_json(
    excluded_columns_path,
    {
        "source_table": TRAINING_TABLE,
        "excluded_column_count": len(excluded_columns),
        "excluded_columns": excluded_columns,
        "explicit_forbidden_model_inputs": sorted(explicit_forbidden_model_inputs),
        "leakage_candidate_hits": leakage_candidate_hits,
    },
)

if validation_failed_checks:
    failure_summary = {
        "phase": "9A",
        "overall_status": "FAIL",
        "failed_checks": validation_failed_checks,
        "message": "Stopped before training because training-data validation or feature-contract validation failed.",
        "training_data_profile": training_data_profile,
    }
    write_json(REPORTS_DIR / "phase9a_logistic_baseline_summary.json", failure_summary)
    sync_reports_to_dbfs()
    print("PHASE9A_RESULT_JSON=" + json.dumps(as_jsonable(failure_summary), sort_keys=True))
    raise RuntimeError(f"Phase 9A validation failed before training: {validation_failed_checks}")

# COMMAND ----------

selected_sdf = training_df.select(*selected_columns).orderBy("feature_timestamp", "transaction_id")
training_pdf = selected_sdf.toPandas()
actual_selected_memory_bytes = int(training_pdf.memory_usage(deep=True).sum())
training_pdf["feature_timestamp"] = pd.to_datetime(training_pdf["feature_timestamp"], utc=True)
training_pdf["_target_int"] = training_pdf[TARGET_COLUMN].apply(normalize_target)

if training_pdf["_target_int"].isna().any():
    raise RuntimeError("Target conversion produced null values after Spark validation.")

training_pdf = (
    training_pdf.sort_values(["feature_timestamp", "transaction_id"], kind="mergesort")
    .reset_index(drop=True)
)

row_count = len(training_pdf)
ORIGINAL_SPLIT_PROPORTIONS = {"train": 0.70, "validation": 0.15, "test": 0.15}
REMEDIATED_MIN_PROPORTIONS = {"train": 0.60, "validation": 0.10, "test": 0.10}
REMEDIATED_MIN_FRAUD = {"validation": 3, "test": 3}
REMEDIATED_PREFERRED_TRAIN_FRAUD = 25


def iso_min(series: pd.Series) -> str | None:
    if series.empty:
        return None
    value = series.min()
    if pd.isna(value):
        return None
    return value.isoformat()


def iso_max(series: pd.Series) -> str | None:
    if series.empty:
        return None
    value = series.max()
    if pd.isna(value):
        return None
    return value.isoformat()


def assign_splits(base_pdf: pd.DataFrame, train_end_index: int, validation_end_index: int) -> pd.DataFrame:
    assigned = base_pdf.copy()
    assigned["split"] = "test"
    assigned.loc[assigned.index < train_end_index, "split"] = "train"
    assigned.loc[(assigned.index >= train_end_index) & (assigned.index < validation_end_index), "split"] = "validation"
    return assigned


def split_stats(split_name: str, split_df: pd.DataFrame) -> dict[str, Any]:
    fraud_count = int((split_df["_target_int"] == 1).sum())
    normal_count = int((split_df["_target_int"] == 0).sum())
    stats = {
        "row_count": int(len(split_df)),
        "percentage": float(len(split_df) / row_count) if row_count else 0.0,
        "fraud_count": fraud_count,
        "normal_count": normal_count,
        "fraud_rate": float(fraud_count / len(split_df)) if len(split_df) else None,
        "min_feature_timestamp": iso_min(split_df["feature_timestamp"]),
        "max_feature_timestamp": iso_max(split_df["feature_timestamp"]),
        "unique_account_count": None,
        "unique_device_count": None,
        "duplicate_transaction_count": int(split_df["transaction_id"].duplicated().sum()),
        "null_target_count": int(split_df["_target_int"].isna().sum()),
    }
    if "account_id" in split_df.columns:
        stats["unique_account_count"] = int(split_df["account_id"].nunique(dropna=True))
    if "device_id" in split_df.columns:
        stats["unique_device_count"] = int(split_df["device_id"].nunique(dropna=True))
    return stats


def build_split_hash(assigned_pdf: pd.DataFrame, split_version: str) -> str:
    split_hash_input = []
    hash_columns = ["transaction_id", "split"]
    if "_feature_version" in assigned_pdf.columns:
        hash_columns.append("_feature_version")
    for _, row in assigned_pdf[hash_columns].sort_values("transaction_id", kind="mergesort").iterrows():
        split_hash_input.append(
            "|".join(
                [
                    str(row["transaction_id"]),
                    str(row["split"]),
                    str(row.get("_feature_version", feature_version_from_contract)),
                    split_version,
                ]
            )
        )
    return hashlib.sha256("\n".join(split_hash_input).encode("utf-8")).hexdigest()


def build_split_artifacts(
    assigned_pdf: pd.DataFrame,
    train_end_index: int,
    validation_end_index: int,
    split_version: str,
    split_policy: str,
    selection_algorithm: str,
    min_validation_fraud: int,
    min_test_fraud: int,
    extra_manifest_fields: dict[str, Any] | None = None,
) -> dict[str, Any]:
    train_frame = assigned_pdf[assigned_pdf["split"] == "train"].copy()
    validation_frame = assigned_pdf[assigned_pdf["split"] == "validation"].copy()
    test_frame = assigned_pdf[assigned_pdf["split"] == "test"].copy()
    split_summary_local = {
        "train": split_stats("train", train_frame),
        "validation": split_stats("validation", validation_frame),
        "test": split_stats("test", test_frame),
    }
    transaction_sets_local = {
        split_name: set(split_df["transaction_id"].astype(str))
        for split_name, split_df in {
            "train": train_frame,
            "validation": validation_frame,
            "test": test_frame,
        }.items()
    }
    overlap_checks_local = {
        "train_validation_overlap_count": len(transaction_sets_local["train"].intersection(transaction_sets_local["validation"])),
        "train_test_overlap_count": len(transaction_sets_local["train"].intersection(transaction_sets_local["test"])),
        "validation_test_overlap_count": len(transaction_sets_local["validation"].intersection(transaction_sets_local["test"])),
        "all_transactions_assigned_once": int(
            len(transaction_sets_local["train"] | transaction_sets_local["validation"] | transaction_sets_local["test"])
            == row_count
        ),
    }
    time_boundary_checks_local = {
        "train_max_lte_validation_min": bool(
            train_frame["feature_timestamp"].max() <= validation_frame["feature_timestamp"].min()
        ),
        "validation_max_lte_test_min": bool(
            validation_frame["feature_timestamp"].max() <= test_frame["feature_timestamp"].min()
        ),
    }
    failed_checks_local: list[str] = []
    if any(value != 0 for key, value in overlap_checks_local.items() if key.endswith("_overlap_count")):
        failed_checks_local.append("split_transaction_id_overlap")
    if overlap_checks_local["all_transactions_assigned_once"] != 1:
        failed_checks_local.append("transactions_not_assigned_exactly_once")
    if any(stats["duplicate_transaction_count"] != 0 for stats in split_summary_local.values()):
        failed_checks_local.append("duplicate_transaction_ids_within_split")
    if any(stats["null_target_count"] != 0 for stats in split_summary_local.values()):
        failed_checks_local.append("null_targets_within_split")
    if split_summary_local["train"]["fraud_count"] == 0 or split_summary_local["train"]["normal_count"] == 0:
        failed_checks_local.append("training_split_missing_target_class")
    if split_summary_local["validation"]["fraud_count"] < min_validation_fraud:
        failed_checks_local.append(
            "validation_split_zero_fraud" if split_summary_local["validation"]["fraud_count"] == 0 else "validation_split_fewer_than_required_fraud"
        )
    if split_summary_local["test"]["fraud_count"] < min_test_fraud:
        failed_checks_local.append(
            "test_split_zero_fraud" if split_summary_local["test"]["fraud_count"] == 0 else "test_split_fewer_than_required_fraud"
        )
    if not all(time_boundary_checks_local.values()):
        failed_checks_local.append("chronological_boundary_violation")

    warnings_local = []
    if 0 < split_summary_local["validation"]["fraud_count"] < 3:
        warnings_local.append("validation_split_has_fewer_than_three_fraud_cases")
    if 0 < split_summary_local["test"]["fraud_count"] < 3:
        warnings_local.append("test_split_has_fewer_than_three_fraud_cases")
    if split_summary_local["train"]["fraud_count"] < REMEDIATED_PREFERRED_TRAIN_FRAUD:
        warnings_local.append("training_split_has_fewer_than_preferred_twenty_five_fraud_cases")

    selected_boundaries = {
        "train_end_index_exclusive": int(train_end_index),
        "validation_end_index_exclusive": int(validation_end_index),
        "validation_start_index": int(train_end_index),
        "test_start_index": int(validation_end_index),
        "train_max_timestamp": split_summary_local["train"]["max_feature_timestamp"],
        "validation_min_timestamp": split_summary_local["validation"]["min_feature_timestamp"],
        "validation_max_timestamp": split_summary_local["validation"]["max_feature_timestamp"],
        "test_min_timestamp": split_summary_local["test"]["min_feature_timestamp"],
        "validation_start_transaction_id": str(assigned_pdf.iloc[train_end_index]["transaction_id"]) if train_end_index < row_count else None,
        "test_start_transaction_id": str(assigned_pdf.iloc[validation_end_index]["transaction_id"]) if validation_end_index < row_count else None,
    }
    manifest = {
        "phase": "9A",
        "source_table": TRAINING_TABLE,
        "split_policy": split_policy,
        "split_version": split_version,
        "selection_algorithm": selection_algorithm,
        "ordering_columns": ["feature_timestamp ascending", "transaction_id ascending"],
        "deterministic_ordering": ["feature_timestamp ascending", "transaction_id ascending"],
        "classification_threshold": CLASSIFICATION_THRESHOLD,
        "selected_time_boundaries": selected_boundaries,
        "split_boundaries": split_summary_local,
        "row_counts": {split_name: stats["row_count"] for split_name, stats in split_summary_local.items()},
        "fraud_normal_counts": {
            split_name: {
                "fraud_count": stats["fraud_count"],
                "normal_count": stats["normal_count"],
                "fraud_rate": stats["fraud_rate"],
            }
            for split_name, stats in split_summary_local.items()
        },
        "percentages": {split_name: stats["percentage"] for split_name, stats in split_summary_local.items()},
        "entity_counts": {
            split_name: {
                "unique_account_count": stats["unique_account_count"],
                "unique_device_count": stats["unique_device_count"],
            }
            for split_name, stats in split_summary_local.items()
        },
        "transaction_id_counts": {
            split_name: int(len(transaction_sets_local[split_name]))
            for split_name in ["train", "validation", "test"]
        },
        "overlap_checks": overlap_checks_local,
        "time_boundary_checks": time_boundary_checks_local,
        "split_hash": build_split_hash(assigned_pdf, split_version),
        "warnings": warnings_local,
        "failed_checks": failed_checks_local,
    }
    if extra_manifest_fields:
        manifest.update(extra_manifest_fields)
    return {
        "assigned_pdf": assigned_pdf,
        "train_pdf": train_frame,
        "validation_pdf": validation_frame,
        "test_pdf": test_frame,
        "split_summary": split_summary_local,
        "overlap_checks": overlap_checks_local,
        "time_boundary_checks": time_boundary_checks_local,
        "split_hash": manifest["split_hash"],
        "split_warnings": warnings_local,
        "split_failed_checks": failed_checks_local,
        "manifest": manifest,
    }


# Preserve the failed fixed chronological split evidence as time_split_v1.
time_split_v1_train_end = int(row_count * 0.70)
time_split_v1_validation_end = int(row_count * 0.85)
time_split_v1_assigned = assign_splits(training_pdf, time_split_v1_train_end, time_split_v1_validation_end)
time_split_v1_artifacts = build_split_artifacts(
    time_split_v1_assigned,
    time_split_v1_train_end,
    time_split_v1_validation_end,
    FAILED_SPLIT_VERSION,
    "oldest 70 percent train, next 15 percent validation, newest 15 percent test",
    "Fixed chronological row cut at 70 and 85 percent. This split failed because the latest 15 percent contained zero fraud cases.",
    min_validation_fraud=1,
    min_test_fraud=1,
)
original_time_split_v1_manifest = time_split_v1_artifacts["manifest"]
original_split_manifest_path = REPORTS_DIR / "phase9a_split_manifest.json"
write_json(original_split_manifest_path, original_time_split_v1_manifest)

# Deterministic rare-event-aware chronological boundary search. Labels are used only to choose contiguous time boundaries.
target_values = training_pdf["_target_int"].astype(int).to_numpy()
fraud_cumulative = np.concatenate([[0], np.cumsum(target_values)])


def range_fraud_count(start_index: int, end_index: int) -> int:
    return int(fraud_cumulative[end_index] - fraud_cumulative[start_index])


def range_normal_count(start_index: int, end_index: int) -> int:
    return int((end_index - start_index) - range_fraud_count(start_index, end_index))


def proportion_distance(train_end_index: int, validation_end_index: int) -> float:
    train_pct = train_end_index / row_count
    validation_pct = (validation_end_index - train_end_index) / row_count
    test_pct = (row_count - validation_end_index) / row_count
    return float(
        abs(train_pct - ORIGINAL_SPLIT_PROPORTIONS["train"])
        + abs(validation_pct - ORIGINAL_SPLIT_PROPORTIONS["validation"])
        + abs(test_pct - ORIGINAL_SPLIT_PROPORTIONS["test"])
    )


def candidate_details(train_end_index: int, validation_end_index: int) -> dict[str, Any]:
    return {
        "train_end_index_exclusive": int(train_end_index),
        "validation_end_index_exclusive": int(validation_end_index),
        "train_rows": int(train_end_index),
        "validation_rows": int(validation_end_index - train_end_index),
        "test_rows": int(row_count - validation_end_index),
        "train_fraud_count": range_fraud_count(0, train_end_index),
        "validation_fraud_count": range_fraud_count(train_end_index, validation_end_index),
        "test_fraud_count": range_fraud_count(validation_end_index, row_count),
        "train_normal_count": range_normal_count(0, train_end_index),
        "validation_normal_count": range_normal_count(train_end_index, validation_end_index),
        "test_normal_count": range_normal_count(validation_end_index, row_count),
        "proportion_distance_from_70_15_15": proportion_distance(train_end_index, validation_end_index),
        "validation_start_transaction_id": str(training_pdf.iloc[train_end_index]["transaction_id"]) if train_end_index < row_count else None,
        "test_start_transaction_id": str(training_pdf.iloc[validation_end_index]["transaction_id"]) if validation_end_index < row_count else None,
        "validation_start_timestamp": training_pdf.iloc[train_end_index]["feature_timestamp"].isoformat() if train_end_index < row_count else None,
        "test_start_timestamp": training_pdf.iloc[validation_end_index]["feature_timestamp"].isoformat() if validation_end_index < row_count else None,
    }


min_train_rows = int(math.ceil(row_count * REMEDIATED_MIN_PROPORTIONS["train"]))
min_validation_rows = int(math.ceil(row_count * REMEDIATED_MIN_PROPORTIONS["validation"]))
min_test_rows = int(math.ceil(row_count * REMEDIATED_MIN_PROPORTIONS["test"]))
valid_candidates: list[dict[str, Any]] = []
latest_test_start_index: int | None = None
for validation_end_candidate in range(row_count - min_test_rows, min_train_rows + min_validation_rows - 1, -1):
    if range_fraud_count(validation_end_candidate, row_count) < REMEDIATED_MIN_FRAUD["test"]:
        continue
    candidates_at_test_start: list[dict[str, Any]] = []
    for train_end_candidate in range(min_train_rows, validation_end_candidate - min_validation_rows + 1):
        train_fraud = range_fraud_count(0, train_end_candidate)
        train_normal = range_normal_count(0, train_end_candidate)
        validation_fraud = range_fraud_count(train_end_candidate, validation_end_candidate)
        if train_fraud <= 0 or train_normal <= 0:
            continue
        if validation_fraud < REMEDIATED_MIN_FRAUD["validation"]:
            continue
        detail = candidate_details(train_end_candidate, validation_end_candidate)
        detail["training_preferred_fraud_count_met"] = detail["train_fraud_count"] >= REMEDIATED_PREFERRED_TRAIN_FRAUD
        candidates_at_test_start.append(detail)
    if candidates_at_test_start:
        latest_test_start_index = validation_end_candidate
        valid_candidates = candidates_at_test_start
        break


def candidate_missing_constraints(candidate: dict[str, Any]) -> list[str]:
    missing = []
    if candidate["train_rows"] < min_train_rows:
        missing.append("training_rows_below_60_percent")
    if candidate["validation_rows"] < min_validation_rows:
        missing.append("validation_rows_below_10_percent")
    if candidate["test_rows"] < min_test_rows:
        missing.append("test_rows_below_10_percent")
    if candidate["train_fraud_count"] <= 0 or candidate["train_normal_count"] <= 0:
        missing.append("training_split_missing_target_class")
    if candidate["validation_fraud_count"] < REMEDIATED_MIN_FRAUD["validation"]:
        missing.append("validation_split_fewer_than_three_fraud")
    if candidate["test_fraud_count"] < REMEDIATED_MIN_FRAUD["test"]:
        missing.append("test_split_fewer_than_three_fraud")
    return missing


if not valid_candidates:
    closest_candidate: dict[str, Any] | None = None
    best_score: tuple[Any, ...] | None = None
    for validation_end_candidate in range(row_count - min_test_rows, min_train_rows + min_validation_rows - 1, -1):
        for train_end_candidate in range(min_train_rows, validation_end_candidate - min_validation_rows + 1):
            detail = candidate_details(train_end_candidate, validation_end_candidate)
            satisfied_fraud_score = min(detail["validation_fraud_count"], 3) + min(detail["test_fraud_count"], 3)
            score = (
                satisfied_fraud_score,
                int(detail["train_fraud_count"] >= REMEDIATED_PREFERRED_TRAIN_FRAUD),
                detail["train_fraud_count"],
                -detail["proportion_distance_from_70_15_15"],
                validation_end_candidate,
            )
            if best_score is None or score > best_score:
                best_score = score
                closest_candidate = detail
    fraud_rows = training_pdf[training_pdf["_target_int"] == 1].copy()
    fraud_counts_by_day = [
        {"date": str(day), "fraud_count": int(count)}
        for day, count in fraud_rows.groupby(fraud_rows["feature_timestamp"].dt.date).size().sort_index().items()
    ]
    fraud_timestamps = [
        {
            "transaction_id": str(row["transaction_id"]),
            "feature_timestamp": row["feature_timestamp"].isoformat(),
        }
        for _, row in fraud_rows.sort_values(["feature_timestamp", "transaction_id"], kind="mergesort").iterrows()
    ]
    failed_v2_manifest = {
        "phase": "9A",
        "overall_status": "FAIL",
        "split_version": SPLIT_VERSION,
        "selection_algorithm": "No valid contiguous chronological rare-event-aware split satisfied all constraints.",
        "ordering_columns": ["feature_timestamp ascending", "transaction_id ascending"],
        "min_rows": {"train": min_train_rows, "validation": min_validation_rows, "test": min_test_rows},
        "failed_time_split_v1": original_time_split_v1_manifest,
        "fraud_counts_by_day": fraud_counts_by_day,
        "fraud_timestamps": fraud_timestamps,
        "closest_candidate_split": closest_candidate,
        "closest_candidate_missing_constraints": candidate_missing_constraints(closest_candidate) if closest_candidate else ["no_candidate_with_minimum_row_counts"],
        "failed_checks": ["no_valid_rare_event_aware_time_split_v2"],
    }
    split_manifest_path = REPORTS_DIR / "phase9a_split_manifest_v2.json"
    write_json(split_manifest_path, failed_v2_manifest)
    failure_summary = {
        "phase": "9A",
        "overall_status": "FAIL",
        "failed_checks": failed_v2_manifest["failed_checks"],
        "message": "Stopped before training because no valid rare-event-aware chronological split satisfied the constraints.",
        "training_data_profile": training_data_profile,
        "original_time_split_v1": original_time_split_v1_manifest,
        "rare_event_aware_time_split_v2": failed_v2_manifest,
    }
    write_json(REPORTS_DIR / "phase9a_logistic_baseline_summary.json", failure_summary)
    sync_reports_to_dbfs()
    print("PHASE9A_RESULT_JSON=" + json.dumps(as_jsonable(failure_summary), sort_keys=True))
    raise RuntimeError("No valid rare_event_aware_time_split_v2 was found.")

selected_candidate = sorted(
    valid_candidates,
    key=lambda item: (
        item["proportion_distance_from_70_15_15"],
        item["test_start_transaction_id"] or "",
        item["validation_start_transaction_id"] or "",
        item["train_end_index_exclusive"],
    ),
)[0]
training_pdf = assign_splits(
    training_pdf,
    selected_candidate["train_end_index_exclusive"],
    selected_candidate["validation_end_index_exclusive"],
)
v2_extra_manifest_fields = {
    "min_rows": {"train": min_train_rows, "validation": min_validation_rows, "test": min_test_rows},
    "min_fraud_constraints": REMEDIATED_MIN_FRAUD,
    "preferred_training_fraud_count": REMEDIATED_PREFERRED_TRAIN_FRAUD,
    "selected_candidate": selected_candidate,
    "candidate_count_at_latest_test_start": len(valid_candidates),
    "latest_possible_test_start_index": latest_test_start_index,
    "comparison_with_failed_time_split_v1": {
        "split_version": FAILED_SPLIT_VERSION,
        "failed_checks": original_time_split_v1_manifest["failed_checks"],
        "row_counts": original_time_split_v1_manifest["row_counts"],
        "fraud_normal_counts": original_time_split_v1_manifest["fraud_normal_counts"],
        "split_hash": original_time_split_v1_manifest["split_hash"],
    },
    "target_aware_boundary_limitation": {
        "is_target_aware": True,
        "records_reordered": False,
        "contiguous_latest_time_test_block": True,
        "warning": "Labels were used only to choose valid contiguous chronological split boundaries in this small rare-event pilot. Evaluation estimates may be optimistic or unstable and are suitable for pipeline and portfolio validation only. A larger production dataset should use a pre-locked temporal test period containing naturally occurring fraud cases.",
    },
}
v2_artifacts = build_split_artifacts(
    training_pdf,
    selected_candidate["train_end_index_exclusive"],
    selected_candidate["validation_end_index_exclusive"],
    SPLIT_VERSION,
    "rare-event-aware contiguous chronological time blocks with minimum split sizes and fraud-count constraints",
    "Sort by feature_timestamp and transaction_id, scan deterministic row boundaries, choose the latest possible test-start timestamp with at least three test fraud cases, then choose the closest candidate to 70/15/15, then transaction_id tie-breakers.",
    min_validation_fraud=REMEDIATED_MIN_FRAUD["validation"],
    min_test_fraud=REMEDIATED_MIN_FRAUD["test"],
    extra_manifest_fields=v2_extra_manifest_fields,
)
train_pdf = v2_artifacts["train_pdf"]
validation_pdf = v2_artifacts["validation_pdf"]
test_pdf = v2_artifacts["test_pdf"]
split_summary = v2_artifacts["split_summary"]
overlap_checks = v2_artifacts["overlap_checks"]
time_boundary_checks = v2_artifacts["time_boundary_checks"]
split_hash = v2_artifacts["split_hash"]
split_failed_checks = v2_artifacts["split_failed_checks"]
split_warnings = v2_artifacts["split_warnings"]
split_manifest = v2_artifacts["manifest"]
split_manifest_path = REPORTS_DIR / "phase9a_split_manifest_v2.json"
write_json(split_manifest_path, split_manifest)

if split_failed_checks:
    failure_summary = {
        "phase": "9A",
        "overall_status": "FAIL",
        "failed_checks": split_failed_checks,
        "message": "Stopped before training because rare_event_aware_time_split_v2 quality gates failed.",
        "training_data_profile": training_data_profile,
        "original_time_split_v1": original_time_split_v1_manifest,
        "rare_event_aware_time_split_v2": split_manifest,
    }
    write_json(REPORTS_DIR / "phase9a_logistic_baseline_summary.json", failure_summary)
    sync_reports_to_dbfs()
    print("PHASE9A_RESULT_JSON=" + json.dumps(as_jsonable(failure_summary), sort_keys=True))
    raise RuntimeError(f"Phase 9A v2 split validation failed before training: {split_failed_checks}")

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


def is_binary_indicator_name(feature_name: str) -> bool:
    return (
        feature_name.startswith("is_")
        or feature_name.startswith("has_")
        or feature_name.endswith("_available")
        or feature_name.endswith("_applicable")
        or "_flag" in feature_name
        or "mismatch" in feature_name
    )


X_train = train_pdf[contract_candidate_features].copy()
y_train = train_pdf["_target_int"].astype(int).to_numpy()
X_validation = validation_pdf[contract_candidate_features].copy()
y_validation = validation_pdf["_target_int"].astype(int).to_numpy()
X_test = test_pdf[contract_candidate_features].copy()
y_test = test_pdf["_target_int"].astype(int).to_numpy()

categorical_features = [
    feature_name
    for feature_name in contract_candidate_features
    if feature_name in categorical_candidate_features
    or str(X_train[feature_name].dtype) in {"object", "category", "string"}
]
numeric_features = [feature_name for feature_name in contract_candidate_features if feature_name not in categorical_features]

numeric_conversion_issues: dict[str, dict[str, int]] = {}
numeric_training_all_null = []
for feature_name in numeric_features:
    split_issues: dict[str, int] = {}
    for split_name, frame in [
        ("train", X_train),
        ("validation", X_validation),
        ("test", X_test),
    ]:
        original = frame[feature_name]
        converted = pd.to_numeric(original, errors="coerce")
        bad_numeric = int((original.notna() & converted.isna()).sum())
        infinite_count = int(np.isinf(converted.dropna().to_numpy(dtype=float)).sum())
        split_issues[f"{split_name}_bad_numeric_count"] = bad_numeric
        split_issues[f"{split_name}_infinite_count"] = infinite_count
    train_converted = pd.to_numeric(X_train[feature_name], errors="coerce").replace([np.inf, -np.inf], np.nan)
    if int(train_converted.notna().sum()) == 0:
        numeric_training_all_null.append(feature_name)
    numeric_conversion_issues[feature_name] = split_issues

binary_features = []
for feature_name in numeric_features:
    converted = pd.to_numeric(X_train[feature_name], errors="coerce").dropna().unique()
    distinct_values = {float(value) for value in converted.tolist()}
    if distinct_values and distinct_values.issubset({0.0, 1.0}) and is_binary_indicator_name(feature_name):
        binary_features.append(feature_name)
continuous_numeric_features = [feature_name for feature_name in numeric_features if feature_name not in binary_features]

preprocessing_failed_checks: list[str] = []
if numeric_training_all_null:
    preprocessing_failed_checks.append("numeric_features_all_null_in_training_split")
if any(
    value != 0
    for issue in numeric_conversion_issues.values()
    for key, value in issue.items()
    if key.endswith("_bad_numeric_count") or key.endswith("_infinite_count")
):
    preprocessing_failed_checks.append("numeric_conversion_or_infinite_values_present")

if preprocessing_failed_checks:
    failure_summary = {
        "phase": "9A",
        "overall_status": "FAIL",
        "failed_checks": preprocessing_failed_checks,
        "numeric_training_all_null": numeric_training_all_null,
        "numeric_conversion_issues": numeric_conversion_issues,
        "message": "Stopped before training because preprocessing validation failed.",
    }
    write_json(REPORTS_DIR / "phase9a_logistic_baseline_summary.json", failure_summary)
    sync_reports_to_dbfs()
    print("PHASE9A_RESULT_JSON=" + json.dumps(as_jsonable(failure_summary), sort_keys=True))
    raise RuntimeError(f"Phase 9A preprocessing validation failed before training: {preprocessing_failed_checks}")

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
        ("numeric", numeric_transformer, numeric_features),
        ("categorical", categorical_transformer, categorical_features),
    ],
    remainder="drop",
    verbose_feature_names_out=True,
)

# COMMAND ----------


def metric_value(value: Any) -> float | int | None:
    if value is None:
        return None
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        value = float(value)
        if not math.isfinite(value):
            return None
        return value
    return value


def positive_probability(model: Any, X: pd.DataFrame) -> np.ndarray:
    probabilities = model.predict_proba(X)
    classes = list(model.classes_)
    if 1 in classes:
        return probabilities[:, classes.index(1)]
    if True in classes:
        return probabilities[:, classes.index(True)]
    return np.zeros(len(X), dtype=float)


def evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray) -> dict[str, Any]:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    both_classes = len(np.unique(y_true)) == 2
    metrics = {
        "average_precision": metric_value(average_precision_score(y_true, y_prob)) if np.sum(y_true) > 0 else None,
        "roc_auc": metric_value(roc_auc_score(y_true, y_prob)) if both_classes else None,
        "fraud_precision": metric_value(precision_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "fraud_recall": metric_value(recall_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "fraud_f1": metric_value(f1_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "accuracy": metric_value(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": metric_value(balanced_accuracy_score(y_true, y_pred)),
        "specificity": metric_value(tn / (tn + fp)) if (tn + fp) else None,
        "false_positive_rate": metric_value(fp / (fp + tn)) if (fp + tn) else None,
        "false_negative_rate": metric_value(fn / (fn + tp)) if (fn + tp) else None,
        "true_positives": int(tp),
        "false_positives": int(fp),
        "true_negatives": int(tn),
        "false_negatives": int(fn),
        "predicted_fraud_count": int(np.sum(y_pred == 1)),
        "predicted_normal_count": int(np.sum(y_pred == 0)),
        "threshold": CLASSIFICATION_THRESHOLD,
        "row_count": int(len(y_true)),
        "fraud_count": int(np.sum(y_true == 1)),
        "normal_count": int(np.sum(y_true == 0)),
        "confusion_matrix": {
            "true_negative": int(tn),
            "false_positive": int(fp),
            "false_negative": int(fn),
            "true_positive": int(tp),
        },
    }
    return metrics


def evaluate_model(model: Any, X: pd.DataFrame, y_true: np.ndarray) -> dict[str, Any]:
    y_prob = positive_probability(model, X)
    y_pred = (y_prob >= CLASSIFICATION_THRESHOLD).astype(int)
    return {
        "metrics": evaluate_predictions(y_true, y_pred, y_prob),
        "y_pred": y_pred,
        "y_prob": y_prob,
    }


def save_confusion_matrix_plot(metrics: dict[str, Any], title: str, path: Path) -> None:
    matrix = np.array(
        [
            [
                metrics["confusion_matrix"]["true_negative"],
                metrics["confusion_matrix"]["false_positive"],
            ],
            [
                metrics["confusion_matrix"]["false_negative"],
                metrics["confusion_matrix"]["true_positive"],
            ],
        ]
    )
    plt.figure(figsize=(5.5, 4.5))
    plt.imshow(matrix, interpolation="nearest", cmap="Blues")
    plt.title(title)
    plt.colorbar()
    plt.xticks([0, 1], ["Pred normal", "Pred fraud"])
    plt.yticks([0, 1], ["Actual normal", "Actual fraud"])
    for row_index in range(2):
        for col_index in range(2):
            plt.text(
                col_index,
                row_index,
                str(matrix[row_index, col_index]),
                ha="center",
                va="center",
                color="white" if matrix[row_index, col_index] > matrix.max() / 2 else "black",
            )
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def save_precision_recall_plot(y_true: np.ndarray, y_prob: np.ndarray, title: str, path: Path) -> None:
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    ap = average_precision_score(y_true, y_prob) if np.sum(y_true) > 0 else None
    plt.figure(figsize=(6, 4.5))
    plt.plot(recall, precision, linewidth=2)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    suffix = f" (AP={ap:.4f})" if ap is not None else ""
    plt.title(title + suffix)
    plt.xlim(0, 1)
    plt.ylim(0, 1.02)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def save_roc_plot(y_true: np.ndarray, y_prob: np.ndarray, title: str, path: Path) -> bool:
    if len(np.unique(y_true)) < 2:
        return False
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc = roc_auc_score(y_true, y_prob)
    plt.figure(figsize=(6, 4.5))
    plt.plot(fpr, tpr, linewidth=2)
    plt.plot([0, 1], [0, 1], linestyle="--", linewidth=1, color="gray")
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.title(f"{title} (ROC-AUC={auc:.4f})")
    plt.xlim(0, 1)
    plt.ylim(0, 1.02)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    return True


def save_probability_distribution_plot(y_true: np.ndarray, y_prob: np.ndarray, title: str, path: Path) -> None:
    plt.figure(figsize=(6, 4.5))
    normal_probs = y_prob[y_true == 0]
    fraud_probs = y_prob[y_true == 1]
    plt.hist(normal_probs, bins=20, alpha=0.65, label="Normal", color="#4C78A8")
    plt.hist(fraud_probs, bins=20, alpha=0.75, label="Fraud", color="#E45756")
    plt.axvline(CLASSIFICATION_THRESHOLD, color="black", linewidth=1, linestyle="--")
    plt.xlabel("Predicted fraud probability")
    plt.ylabel("Transaction count")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def save_coefficient_plot(coefficient_rows: list[dict[str, Any]], path: Path) -> None:
    top_rows = sorted(coefficient_rows, key=lambda item: abs(item["coefficient"]), reverse=True)[:20]
    top_rows = list(reversed(top_rows))
    labels = [row["transformed_feature_name"] for row in top_rows]
    values = [row["coefficient"] for row in top_rows]
    colors = ["#E45756" if value > 0 else "#4C78A8" for value in values]
    plt.figure(figsize=(9, max(5, len(labels) * 0.32)))
    plt.barh(range(len(labels)), values, color=colors)
    plt.yticks(range(len(labels)), labels, fontsize=8)
    plt.xlabel("Logistic Regression coefficient")
    plt.title("Top Logistic Regression coefficient magnitudes")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def save_model_plots(model_name: str, split_name: str, y_true: np.ndarray, eval_result: dict[str, Any]) -> list[str]:
    y_pred = eval_result["y_pred"]
    y_prob = eval_result["y_prob"]
    metrics = eval_result["metrics"]
    paths = []
    cm_path = PLOTS_DIR / f"{model_name}_{split_name}_confusion_matrix.png"
    pr_path = PLOTS_DIR / f"{model_name}_{split_name}_precision_recall_curve.png"
    roc_path = PLOTS_DIR / f"{model_name}_{split_name}_roc_curve.png"
    prob_path = PLOTS_DIR / f"{model_name}_{split_name}_probability_distribution.png"
    save_confusion_matrix_plot(metrics, f"{model_name} {split_name} confusion matrix", cm_path)
    save_precision_recall_plot(y_true, y_prob, f"{model_name} {split_name} precision-recall", pr_path)
    if save_roc_plot(y_true, y_prob, f"{model_name} {split_name} ROC", roc_path):
        paths.append(str(roc_path))
    save_probability_distribution_plot(y_true, y_prob, f"{model_name} {split_name} fraud probabilities", prob_path)
    paths.extend([str(cm_path), str(pr_path), str(prob_path)])
    return paths


def log_metric_dict(prefix: str, metrics: dict[str, Any]) -> None:
    for key, value in metrics.items():
        if isinstance(value, dict):
            continue
        cleaned = metric_value(value)
        if isinstance(cleaned, (int, float)) and cleaned is not None:
            mlflow.log_metric(f"{prefix}_{key}", float(cleaned))


def base_mlflow_params(model_type: str) -> dict[str, Any]:
    params = {
        "model_type": model_type,
        "training_table": TRAINING_TABLE,
        "feature_table": FEATURE_TABLE,
        "feature_version": feature_version_from_contract,
        "split_version": SPLIT_VERSION,
        "split_hash": split_hash,
        "classification_threshold": CLASSIFICATION_THRESHOLD,
        "candidate_feature_count": len(contract_candidate_features),
        "numeric_feature_count": len(numeric_features),
        "categorical_feature_count": len(categorical_features),
        "binary_feature_count": len(binary_features),
        "train_row_count": split_summary["train"]["row_count"],
        "validation_row_count": split_summary["validation"]["row_count"],
        "test_row_count": split_summary["test"]["row_count"],
        "train_fraud_count": split_summary["train"]["fraud_count"],
        "validation_fraud_count": split_summary["validation"]["fraud_count"],
        "test_fraud_count": split_summary["test"]["fraud_count"],
        "train_normal_count": split_summary["train"]["normal_count"],
        "validation_normal_count": split_summary["validation"]["normal_count"],
        "test_normal_count": split_summary["test"]["normal_count"],
    }
    return params


def log_params(params: dict[str, Any]) -> None:
    for key, value in params.items():
        if value is None:
            continue
        mlflow.log_param(key, value)

# COMMAND ----------

dummy_classifier = DummyClassifier(strategy="most_frequent")
dummy_classifier.fit(X_train, y_train)
dummy_validation_eval = evaluate_model(dummy_classifier, X_validation, y_validation)
dummy_test_eval = evaluate_model(dummy_classifier, X_test, y_test)
dummy_plot_paths = []
dummy_plot_paths.extend(save_model_plots("dummy_most_frequent_v1", "validation", y_validation, dummy_validation_eval))
dummy_plot_paths.extend(save_model_plots("dummy_most_frequent_v1", "test", y_test, dummy_test_eval))

dummy_metrics_report = {
    "model_name": "dummy_most_frequent_v1",
    "strategy": "most_frequent",
    "purpose": "Demonstrate why raw accuracy is misleading for this imbalanced fraud dataset.",
    "validation_metrics": dummy_validation_eval["metrics"],
    "test_metrics": dummy_test_eval["metrics"],
    "plot_paths": dummy_plot_paths,
}
dummy_metrics_path = REPORTS_DIR / "phase9a_dummy_baseline_metrics.json"
write_json(dummy_metrics_path, dummy_metrics_report)

mlflow.set_tracking_uri("databricks")
mlflow.set_experiment(EXPERIMENT_PATH)

with mlflow.start_run(run_name="dummy_most_frequent_v1") as dummy_run:
    dummy_run_id = dummy_run.info.run_id
    params = base_mlflow_params("DummyClassifier")
    params.update({"strategy": "most_frequent"})
    log_params(params)
    log_metric_dict("validation", dummy_validation_eval["metrics"])
    log_metric_dict("test", dummy_test_eval["metrics"])
    mlflow.log_artifact(str(training_data_profile_path), artifact_path="reports")
    mlflow.log_artifact(str(original_split_manifest_path), artifact_path="reports")
    mlflow.log_artifact(str(split_manifest_path), artifact_path="reports")
    mlflow.log_artifact(str(candidate_features_path), artifact_path="reports")
    mlflow.log_artifact(str(excluded_columns_path), artifact_path="reports")
    mlflow.log_artifact(str(dummy_metrics_path), artifact_path="reports")
    mlflow.log_artifacts(str(PLOTS_DIR), artifact_path="reports/phase9a_plots")

# COMMAND ----------

logistic_regression = LogisticRegression(
    class_weight="balanced",
    solver="liblinear",
    max_iter=1000,
    random_state=RANDOM_STATE,
)
logistic_pipeline = Pipeline(
    steps=[
        ("preprocess", preprocessor),
        ("classifier", logistic_regression),
    ]
)
logistic_pipeline.fit(X_train, y_train)

transformed_train = logistic_pipeline.named_steps["preprocess"].transform(X_train)
transformed_feature_count = int(transformed_train.shape[1])

one_hot_encoder = (
    logistic_pipeline.named_steps["preprocess"]
    .named_transformers_["categorical"]
    .named_steps["one_hot"]
)
learned_categorical_values = {
    feature_name: [as_jsonable(value) for value in values.tolist()]
    for feature_name, values in zip(categorical_features, one_hot_encoder.categories_)
}

preprocessing_summary = {
    "preprocessing_fit_scope": "training split only",
    "sklearn_pipeline": "ColumnTransformer inside fitted sklearn Pipeline",
    "numeric_feature_count": len(numeric_features),
    "numeric_features": numeric_features,
    "continuous_numeric_features": continuous_numeric_features,
    "binary_feature_count": len(binary_features),
    "binary_features": binary_features,
    "categorical_feature_count": len(categorical_features),
    "categorical_features": categorical_features,
    "null_handling_rules": {
        "numeric": "Coerce approved numeric inputs, reject bad non-null numeric values before fit, impute allowed nulls with training median, then standardize.",
        "categorical": "Fill missing categorical values with __MISSING__, one-hot encode categories learned from training data only, and ignore unknown categories at validation/test/inference time.",
    },
    "learned_categorical_values": learned_categorical_values,
    "final_transformed_feature_dimension": transformed_feature_count,
    "numeric_conversion_issues": numeric_conversion_issues,
    "numeric_training_all_null": numeric_training_all_null,
}
preprocessing_summary_path = REPORTS_DIR / "phase9a_preprocessing_summary.json"
write_json(preprocessing_summary_path, preprocessing_summary)

logistic_validation_eval = evaluate_model(logistic_pipeline, X_validation, y_validation)
logistic_test_eval = evaluate_model(logistic_pipeline, X_test, y_test)
logistic_plot_paths = []
logistic_plot_paths.extend(save_model_plots("logistic_regression_balanced_v1", "validation", y_validation, logistic_validation_eval))
logistic_plot_paths.extend(save_model_plots("logistic_regression_balanced_v1", "test", y_test, logistic_test_eval))

transformed_feature_names = list(
    logistic_pipeline.named_steps["preprocess"].get_feature_names_out()
)
coefficients = logistic_pipeline.named_steps["classifier"].coef_[0]
coefficient_rows = [
    {
        "transformed_feature_name": transformed_feature_names[index],
        "coefficient": float(coefficients[index]),
        "absolute_coefficient": float(abs(coefficients[index])),
    }
    for index in range(len(transformed_feature_names))
]
coefficient_rows_sorted = sorted(coefficient_rows, key=lambda item: item["coefficient"], reverse=True)
largest_positive_coefficients = coefficient_rows_sorted[:20]
largest_negative_coefficients = sorted(coefficient_rows, key=lambda item: item["coefficient"])[:20]
coefficient_report = {
    "model_name": "logistic_regression_balanced_v1",
    "coefficient_interpretation_warning": "Coefficients represent associations learned from synthetic training data and do not establish causation.",
    "largest_positive_coefficients": largest_positive_coefficients,
    "largest_negative_coefficients": largest_negative_coefficients,
    "top_absolute_coefficients": sorted(coefficient_rows, key=lambda item: item["absolute_coefficient"], reverse=True)[:30],
}
coefficient_report_path = REPORTS_DIR / "phase9a_logistic_coefficient_report.json"
coefficient_report_csv_path = REPORTS_DIR / "phase9a_logistic_coefficients.csv"
write_json(coefficient_report_path, coefficient_report)
pd.DataFrame(coefficient_rows).sort_values("absolute_coefficient", ascending=False).to_csv(
    coefficient_report_csv_path, index=False
)
coefficient_plot_path = PLOTS_DIR / "logistic_regression_balanced_v1_top_coefficient_magnitudes.png"
save_coefficient_plot(coefficient_rows, coefficient_plot_path)
logistic_plot_paths.append(str(coefficient_plot_path))

# COMMAND ----------


def segment_metric_block(
    frame: pd.DataFrame,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    segment_name: str,
    segment_value: Any,
) -> dict[str, Any]:
    row_count_segment = int(len(frame))
    fraud_count_segment = int(np.sum(y_true == 1))
    high_risk_count = int(np.sum(y_pred == 1))
    result = {
        "segment_name": segment_name,
        "segment_value": as_jsonable(segment_value),
        "row_count": row_count_segment,
        "fraud_count": fraud_count_segment,
        "average_predicted_fraud_probability": metric_value(float(np.mean(y_prob))) if row_count_segment else None,
        "predicted_high_risk_count_at_threshold_0_5": high_risk_count,
        "precision_defined": high_risk_count > 0,
        "recall_defined": fraud_count_segment > 0,
        "fraud_precision": None,
        "fraud_recall": None,
    }
    if high_risk_count > 0:
        result["fraud_precision"] = metric_value(precision_score(y_true, y_pred, pos_label=1, zero_division=0))
    if fraud_count_segment > 0:
        result["fraud_recall"] = metric_value(recall_score(y_true, y_pred, pos_label=1, zero_division=0))
    return result


def segment_analysis_for_split(split_name: str, split_df: pd.DataFrame, eval_result: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    output: dict[str, list[dict[str, Any]]] = {}
    y_pred = eval_result["y_pred"]
    y_prob = eval_result["y_prob"]
    y_true = split_df["_target_int"].astype(int).to_numpy()
    analysis_frame = split_df.reset_index(drop=True)

    for segment_column in ["asset", "transaction_type", "customer_risk_tier"]:
        if segment_column not in analysis_frame.columns:
            continue
        rows = []
        segment_values = analysis_frame[segment_column].where(
            analysis_frame[segment_column].notna(), "__MISSING__"
        )
        for value in sorted(segment_values.astype(str).unique().tolist()):
            mask = segment_values.astype(str) == value
            rows.append(
                segment_metric_block(
                    analysis_frame.loc[mask],
                    y_true[mask.to_numpy()],
                    y_pred[mask.to_numpy()],
                    y_prob[mask.to_numpy()],
                    segment_column,
                    value,
                )
            )
        output[segment_column] = rows

    if "has_previous_transaction" in analysis_frame.columns:
        cold_start_values = pd.to_numeric(analysis_frame["has_previous_transaction"], errors="coerce").fillna(0)
        cold_start_labels = np.where(cold_start_values == 0, "cold_start", "existing_history")
        rows = []
        for value in ["cold_start", "existing_history"]:
            mask_array = cold_start_labels == value
            rows.append(
                segment_metric_block(
                    analysis_frame.loc[mask_array],
                    y_true[mask_array],
                    y_pred[mask_array],
                    y_prob[mask_array],
                    "cold_start_status",
                    value,
                )
            )
        output["cold_start_status"] = rows

    return output


segment_analysis = {
    "model_name": "logistic_regression_balanced_v1",
    "scope": "validation and locked test splits at threshold 0.5",
    "limitations": "Segment metrics are unstable when segment fraud counts are small.",
    "validation": segment_analysis_for_split("validation", validation_pdf, logistic_validation_eval),
    "test": segment_analysis_for_split("test", test_pdf, logistic_test_eval),
}
segment_analysis_path = REPORTS_DIR / "phase9a_segment_analysis.json"
write_json(segment_analysis_path, segment_analysis)

logistic_metrics_report = {
    "model_name": "logistic_regression_balanced_v1",
    "fixed_configuration": {
        "class_weight": "balanced",
        "solver": "liblinear",
        "max_iter": 1000,
        "random_state": RANDOM_STATE,
        "threshold": CLASSIFICATION_THRESHOLD,
    },
    "validation_metrics": logistic_validation_eval["metrics"],
    "test_metrics": logistic_test_eval["metrics"],
    "plot_paths": logistic_plot_paths,
}
logistic_metrics_path = REPORTS_DIR / "phase9a_logistic_baseline_metrics.json"
write_json(logistic_metrics_path, logistic_metrics_report)

package_versions = {
    "python": sys.version.split()[0],
    "python_full": sys.version,
    "platform": platform.platform(),
    "pandas": pd.__version__,
    "numpy": np.__version__,
    "scikit_learn": sklearn.__version__,
    "mlflow": mlflow.__version__,
    "matplotlib": matplotlib.__version__,
    "pyspark": spark.version,
    "cloudpickle": getattr(cloudpickle, "__version__", None) if cloudpickle else None,
    "package_status_before_install": package_status_before_install,
    "package_status_after_install": package_status_after_install,
    "notebook_scoped_packages_installed": installed_packages,
}
package_versions_path = REPORTS_DIR / "phase9a_package_versions.json"
write_json(package_versions_path, package_versions)

# COMMAND ----------

target_class_mapping = {
    "positive_class": 1,
    "positive_class_name": "fraud",
    "negative_class": 0,
    "negative_class_name": "normal",
    "source_target_column": TARGET_COLUMN,
    "approved_semantics": "target_is_fraud=True means confirmed fraud; target_is_fraud=False means cleared normal.",
}
target_class_mapping_path = REPORTS_DIR / "phase9a_target_class_mapping.json"
write_json(target_class_mapping_path, target_class_mapping)

model_uri = None
reload_test_results: dict[str, Any] | None = None

with mlflow.start_run(run_name="logistic_regression_balanced_v1") as logistic_run:
    logistic_run_id = logistic_run.info.run_id
    params = base_mlflow_params("LogisticRegression")
    params.update(
        {
            "class_weight": "balanced",
            "solver": "liblinear",
            "max_iter": 1000,
            "random_state": RANDOM_STATE,
            "transformed_feature_count": transformed_feature_count,
        }
    )
    log_params(params)
    log_metric_dict("validation", logistic_validation_eval["metrics"])
    log_metric_dict("test", logistic_test_eval["metrics"])

    input_example = X_train.head(min(10, len(X_train))).copy()
    signature = infer_signature(input_example, logistic_pipeline.predict(input_example))
    pip_requirements = [
        f"mlflow=={mlflow.__version__}",
        f"scikit-learn=={sklearn.__version__}",
        f"pandas=={pd.__version__}",
        f"numpy=={np.__version__}",
    ]
    if cloudpickle is not None and getattr(cloudpickle, "__version__", None):
        pip_requirements.append(f"cloudpickle=={cloudpickle.__version__}")

    try:
        mlflow.sklearn.log_model(
            sk_model=logistic_pipeline,
            artifact_path="model",
            signature=signature,
            input_example=input_example,
            pip_requirements=pip_requirements,
        )
        model_uri = f"runs:/{logistic_run_id}/model"
    except TypeError:
        mlflow.sklearn.log_model(
            sk_model=logistic_pipeline,
            name="model",
            signature=signature,
            input_example=input_example,
            pip_requirements=pip_requirements,
        )
        model_uri = f"runs:/{logistic_run_id}/model"

    deterministic_sample_ids: list[str] = []

    def add_sample_rows(frame: pd.DataFrame, condition: pd.Series, limit: int) -> None:
        rows = frame.loc[condition].sort_values(["feature_timestamp", "transaction_id"]).head(limit)
        for transaction_id in rows["transaction_id"].astype(str).tolist():
            if transaction_id not in deterministic_sample_ids:
                deterministic_sample_ids.append(transaction_id)

    add_sample_rows(test_pdf, test_pdf["_target_int"] == 1, 5)
    add_sample_rows(test_pdf, test_pdf["_target_int"] == 0, 5)
    if "asset" in test_pdf.columns:
        add_sample_rows(test_pdf, test_pdf["asset"].astype(str) == "BTC", 2)
        add_sample_rows(test_pdf, test_pdf["asset"].astype(str) == "ETH", 2)
    if "transaction_type" in test_pdf.columns:
        for transaction_type in sorted(test_pdf["transaction_type"].dropna().astype(str).unique().tolist()):
            add_sample_rows(test_pdf, test_pdf["transaction_type"].astype(str) == transaction_type, 1)

    reload_sample = (
        test_pdf[test_pdf["transaction_id"].astype(str).isin(deterministic_sample_ids)]
        .sort_values(["feature_timestamp", "transaction_id"])
        .reset_index(drop=True)
    )
    sample_X = reload_sample[contract_candidate_features].copy()
    original_classes = logistic_pipeline.predict(sample_X).astype(int)
    original_probabilities = positive_probability(logistic_pipeline, sample_X)
    reloaded_model = mlflow.sklearn.load_model(model_uri)
    reloaded_classes = reloaded_model.predict(sample_X).astype(int)
    reloaded_probabilities = positive_probability(reloaded_model, sample_X)
    max_probability_difference = float(np.max(np.abs(original_probabilities - reloaded_probabilities))) if len(sample_X) else 0.0
    reload_test_passed = bool(
        np.array_equal(original_classes, reloaded_classes)
        and max_probability_difference <= 1e-12
    )
    reload_test_results = {
        "status": "PASS" if reload_test_passed else "FAIL",
        "sample_row_count": int(len(sample_X)),
        "sample_transaction_ids": reload_sample["transaction_id"].astype(str).tolist(),
        "sample_coverage": {
            "fraud_examples": int((reload_sample["_target_int"] == 1).sum()),
            "normal_examples": int((reload_sample["_target_int"] == 0).sum()),
            "btc_examples": int((reload_sample["asset"].astype(str) == "BTC").sum()) if "asset" in reload_sample.columns else None,
            "eth_examples": int((reload_sample["asset"].astype(str) == "ETH").sum()) if "asset" in reload_sample.columns else None,
            "transaction_types": sorted(reload_sample["transaction_type"].dropna().astype(str).unique().tolist()) if "transaction_type" in reload_sample.columns else [],
        },
        "predicted_classes_match_exactly": bool(np.array_equal(original_classes, reloaded_classes)),
        "max_probability_difference": max_probability_difference,
        "probability_tolerance": 1e-12,
    }
    reload_test_path = REPORTS_DIR / "phase9a_model_reload_test.json"
    write_json(reload_test_path, reload_test_results)

    training_delta_after = delta_latest_version(TRAINING_TABLE)
    feature_delta_after = delta_latest_version(FEATURE_TABLE)
    source_delta_unchanged = {
        "training_table_version_unchanged": training_delta_before.get("version") == training_delta_after.get("version"),
        "feature_table_version_unchanged": feature_delta_before.get("version") == feature_delta_after.get("version"),
        "training_table_before": training_delta_before,
        "training_table_after": training_delta_after,
        "feature_table_before": feature_delta_before,
        "feature_table_after": feature_delta_after,
    }

    quality_gate_checks = {
        "source_training_dataset_validation_passes": not validation_failed_checks,
        "approved_feature_count_reconciled": len(contract_candidate_features) == EXPECTED_MODEL_CANDIDATE_COUNT,
        "no_identity_or_label_leakage_columns_used": not leakage_candidate_hits,
        "chronological_splits_have_no_overlap": all(
            value == 0 for key, value in overlap_checks.items() if key.endswith("_overlap_count")
        ),
        "required_target_classes_available_in_splits": (
            split_summary["train"]["fraud_count"] > 0
            and split_summary["train"]["normal_count"] > 0
            and split_summary["validation"]["fraud_count"] >= REMEDIATED_MIN_FRAUD["validation"]
            and split_summary["test"]["fraud_count"] >= REMEDIATED_MIN_FRAUD["test"]
        ),
        "preprocessing_fit_on_training_only": True,
        "dummy_classifier_run_logged": bool(dummy_run_id),
        "logistic_regression_run_logged": bool(logistic_run_id),
        "logistic_regression_model_artifact_logged": bool(model_uri),
        "validation_and_test_metrics_produced": True,
        "confusion_matrices_and_curves_saved": len(list(PLOTS_DIR.glob("*.png"))) >= 9,
        "model_reload_predictions_match": reload_test_results["status"] == "PASS",
        "source_delta_tables_remain_unchanged": (
            source_delta_unchanged["training_table_version_unchanged"]
            and source_delta_unchanged["feature_table_version_unchanged"]
        ),
        "xgboost_model_trained": False,
        "model_registered": False,
        "real_time_work_started": False,
    }
    failed_quality_gates = [
        key
        for key, value in quality_gate_checks.items()
        if (key in {"xgboost_model_trained", "model_registered", "real_time_work_started"} and value is not False)
        or (key not in {"xgboost_model_trained", "model_registered", "real_time_work_started"} and value is not True)
    ]

    warnings_and_limitations = [
        "The full approved dataset contains only 40 confirmed fraud examples, so validation and test metrics are unstable.",
        "Metrics are suitable for pipeline and portfolio validation, not production-performance claims.",
        "Accuracy is not treated as the primary fraud metric because the dataset is highly imbalanced.",
        "Logistic Regression coefficients are associations learned from synthetic data and do not establish causation.",
        "The classification threshold is fixed at 0.5; threshold optimization is intentionally deferred to Phase 9B.",
        "The rare_event_aware_time_split_v2 boundary is target-aware: labels were used only to choose contiguous chronological boundaries, not to reorder individual rows.",
        "The v2 test split remains a contiguous latest-time block, but evaluation estimates may be optimistic or unstable.",
        "A larger production dataset should use a pre-locked temporal test period containing naturally occurring fraud cases.",
    ] + split_warnings

    commands_executed = [
        "databricks --profile crypto-fraud-dev current-user me",
        "databricks --profile crypto-fraud-dev clusters get 0803-061312-78fw66xn",
        "databricks --profile crypto-fraud-dev clusters start 0803-061312-78fw66xn",
        "databricks --profile crypto-fraud-dev workspace import --file config/feature_definitions.json --format RAW --overwrite /Users/akanaskhan1506@gmail.com/config/feature_definitions.json",
        "databricks --profile crypto-fraud-dev workspace import --file databricks/04_logistic_regression_baseline.py --format SOURCE --language PYTHON --overwrite /Users/akanaskhan1506@gmail.com/04_logistic_regression_baseline",
        "databricks --profile crypto-fraud-dev jobs submit --json .tmp/phase9a_job_submit.json",
        "databricks --profile crypto-fraud-dev workspace export /Users/akanaskhan1506@gmail.com/phase9a_reports/phase9a_20260805_01/<report>.json --file reports/<report>.json --format AUTO",
    ]

    phase9a_summary = {
        "phase": "9A",
        "overall_status": "PASS" if not failed_quality_gates else "FAIL",
        "notebook": "04_logistic_regression_baseline",
        "local_notebook_path": "databricks/04_logistic_regression_baseline.py",
        "workspace_notebook_path": WORKSPACE_NOTEBOOK_PATH,
        "cluster": {
            "cluster_id": CLUSTER_ID,
            "cluster_name": "crypto-fraud-dev-compute",
        },
        "mlflow_experiment_path": EXPERIMENT_PATH,
        "dummy_classifier_run_id": dummy_run_id,
        "logistic_regression_run_id": logistic_run_id,
        "logged_logistic_regression_model_uri": model_uri,
        "output_dbfs_dir": OUTPUT_DBFS_DIR,
        "reports_dbfs_dir": f"{OUTPUT_DBFS_DIR.rstrip('/')}/reports",
        "source_table": TRAINING_TABLE,
        "supporting_feature_table": FEATURE_TABLE,
        "target_column": TARGET_COLUMN,
        "source_row_count": training_row_count,
        "source_column_count": training_column_count,
        "candidate_feature_count": len(contract_candidate_features),
        "excluded_column_count": len(excluded_columns),
        "feature_list": contract_candidate_features,
        "excluded_columns": excluded_columns,
        "preprocessing_configuration": preprocessing_summary,
        "remediation_applied": "rare_event_aware_time_split_v2",
        "original_time_split_v1_result": original_time_split_v1_manifest,
        "rare_event_aware_time_split_v2_result": split_manifest,
        "target_aware_boundary_warning": split_manifest["target_aware_boundary_limitation"],
        "split_statistics": split_manifest,
        "dummy_classifier": dummy_metrics_report,
        "logistic_regression": logistic_metrics_report,
        "validation_metrics": {
            "dummy_classifier": dummy_validation_eval["metrics"],
            "logistic_regression": logistic_validation_eval["metrics"],
        },
        "test_metrics": {
            "dummy_classifier": dummy_test_eval["metrics"],
            "logistic_regression": logistic_test_eval["metrics"],
        },
        "confusion_matrices": {
            "dummy_validation": dummy_validation_eval["metrics"]["confusion_matrix"],
            "dummy_test": dummy_test_eval["metrics"]["confusion_matrix"],
            "logistic_validation": logistic_validation_eval["metrics"]["confusion_matrix"],
            "logistic_test": logistic_test_eval["metrics"]["confusion_matrix"],
        },
        "segment_results": segment_analysis,
        "coefficient_summary": coefficient_report,
        "reload_test_results": reload_test_results,
        "package_versions": package_versions,
        "source_delta_unchanged": source_delta_unchanged,
        "warnings_and_limitations": warnings_and_limitations,
        "quality_gate_checks": quality_gate_checks,
        "failed_checks": failed_quality_gates,
        "implementation_defects_and_fixes": [],
        "commands_executed": commands_executed,
        "phase9b_scope_guardrails": {
            "xgboost_trained": False,
            "hyperparameter_tuning_performed": False,
            "threshold_optimization_performed": False,
            "model_registered": False,
            "model_serving_created": False,
            "real_time_inference_created": False,
            "event_hubs_integration_created": False,
        },
        "artifacts_created": {
            "reports": [
                "reports/phase9a_training_data_profile.json",
                "reports/phase9a_split_manifest.json",
                "reports/phase9a_split_manifest_v2.json",
                "reports/phase9a_logistic_baseline_summary.json",
                "reports/phase9a_candidate_features.json",
                "reports/phase9a_excluded_columns.json",
                "reports/phase9a_preprocessing_summary.json",
                "reports/phase9a_package_versions.json",
                "reports/phase9a_segment_analysis.json",
                "reports/phase9a_logistic_coefficient_report.json",
                "reports/phase9a_logistic_coefficients.csv",
                "reports/phase9a_model_reload_test.json",
                "reports/phase9a_dummy_baseline_metrics.json",
                "reports/phase9a_logistic_baseline_metrics.json",
                "reports/phase9a_target_class_mapping.json",
            ],
            "plots_directory": "reports/phase9a_plots/",
        },
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }

    phase9a_summary_path = REPORTS_DIR / "phase9a_logistic_baseline_summary.json"
    phase9a_summary["artifact_dbfs_sync"] = {"status": "PENDING"}
    write_json(phase9a_summary_path, phase9a_summary)
    dbfs_sync_result = sync_reports_to_dbfs()
    phase9a_summary["artifact_dbfs_sync"] = dbfs_sync_result
    write_json(phase9a_summary_path, phase9a_summary)
    sync_reports_to_dbfs()

    mlflow.log_artifacts(str(REPORTS_DIR), artifact_path="reports")
    mlflow.set_tag("phase", "9A")
    mlflow.set_tag("model_registered", "false")
    mlflow.set_tag("threshold_optimization", "false")
    mlflow.set_tag("xgboost_trained", "false")

if failed_quality_gates:
    print("PHASE9A_RESULT_JSON=" + json.dumps(as_jsonable(phase9a_summary), sort_keys=True))
    raise RuntimeError(f"Phase 9A quality gates failed: {failed_quality_gates}")

result_json = json.dumps(as_jsonable(phase9a_summary), indent=2, sort_keys=True)
print("PHASE9A_RESULT_JSON=" + result_json)
dbutils.notebook.exit(result_json)

