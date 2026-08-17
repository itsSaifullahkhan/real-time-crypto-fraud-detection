# Databricks notebook source
# MAGIC %md
# MAGIC # 05 XGBoost Model Selection
# MAGIC
# MAGIC Phase 9B reuses the approved Phase 9A feature set and chronological split,
# MAGIC trains one fixed XGBoost classifier, compares it with the existing
# MAGIC Logistic Regression MLflow model, locks a validation-selected threshold,
# MAGIC evaluates the selected model once on test, and attempts Unity Catalog
# MAGIC registration only if the existing environment supports it.

# COMMAND ----------

from __future__ import annotations

import hashlib
import importlib
import inspect
import json
import math
import platform
import shutil
import subprocess
import sys
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any


PACKAGE_IMPORTS = {
    "xgboost": "xgboost",
    "sklearn": "sklearn",
    "mlflow": "mlflow",
    "pandas": "pandas",
    "numpy": "numpy",
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
    for package_name, import_name in PACKAGE_IMPORTS.items()
}

installed_packages: list[str] = []
if not package_status_before_install["xgboost"]["available"]:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "xgboost"])
    installed_packages.append("xgboost")

package_status_after_install = {
    package_name: inspect_package(import_name)
    for package_name, import_name in PACKAGE_IMPORTS.items()
}

missing_required = [
    package_name
    for package_name, status in package_status_after_install.items()
    if not status["available"]
]
if missing_required:
    raise RuntimeError(f"Missing required Phase 9B packages: {missing_required}")

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import sklearn
import xgboost
from mlflow.models.signature import infer_signature
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

try:
    import cloudpickle
except ImportError:
    cloudpickle = None

# COMMAND ----------

TRAINING_TABLE = "crypto_fraud.features.transaction_training_dataset"
TARGET_COLUMN = "target_is_fraud"
FEATURE_VERSION = "v1"
SPLIT_VERSION = "rare_event_aware_time_split_v2"
APPROVED_SPLIT_HASH = "6576d4aa251135aa3666c01396b5eeaafe3be1a741b2610262c5f4aebf7bbf01"
TRAIN_END_INDEX_EXCLUSIVE = 2726
VALIDATION_END_INDEX_EXCLUSIVE = 3270
EXPECTED_SPLIT_COUNTS = {
    "train": {"row_count": 2726, "fraud_count": 31, "normal_count": 2695},
    "validation": {"row_count": 544, "fraud_count": 6, "normal_count": 538},
    "test": {"row_count": 1209, "fraud_count": 3, "normal_count": 1206},
}
EXPECTED_FEATURE_COUNT = 55
RANDOM_STATE = 42
THRESHOLDS = [0.30, 0.40, 0.50, 0.60, 0.70, 0.80]
MIN_VALIDATION_RECALL = 4 / 6
LOGISTIC_MODEL_URI = "runs:/9f89dbd9ce374008974f1b6e8562aa57/model"
EXPERIMENT_PATH = "/Users/akanaskhan1506@gmail.com/crypto-fraud-phase9b"
WORKSPACE_NOTEBOOK_PATH = "/Users/akanaskhan1506@gmail.com/05_xgboost_model_selection"
WORKSPACE_REPORT_PATH = "/Users/akanaskhan1506@gmail.com/phase9b_reports/phase9b_latest"
WORKSPACE_REPORT_ROOT = Path("/Workspace" + WORKSPACE_REPORT_PATH)
REGISTERED_MODEL_NAME = "crypto_fraud.models.fraud_detection_model"
RUN_NAME = "xgboost_baseline_v1"

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
NUMERIC_FEATURES = [feature_name for feature_name in APPROVED_FEATURES if feature_name not in CATEGORICAL_FEATURES]

OUTPUT_ROOT = Path("/tmp") / "crypto_fraud_phase9b"
REPORTS_DIR = OUTPUT_ROOT / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

spark.conf.set("spark.sql.session.timeZone", "UTC")
spark.conf.set("spark.sql.legacy.parquet.nanosAsLong", "true")

# COMMAND ----------


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


def sync_reports_to_workspace() -> dict[str, Any]:
    copied_files = []
    try:
        WORKSPACE_REPORT_ROOT.mkdir(parents=True, exist_ok=True)
        for filename in ["phase9b_model_comparison.json", "phase9b_summary.json"]:
            source_path = REPORTS_DIR / filename
            if not source_path.exists():
                continue
            target_path = WORKSPACE_REPORT_ROOT / filename
            shutil.copy2(source_path, target_path)
            copied_files.append(filename)
        return {
            "status": "PASS",
            "workspace_reports_path": WORKSPACE_REPORT_PATH,
            "copied_files": copied_files,
        }
    except Exception as exc:
        return {
            "status": "FAIL",
            "workspace_reports_path": WORKSPACE_REPORT_PATH,
            "error": repr(exc),
        }


def finish_notebook(summary: dict[str, Any]) -> None:
    result_json = json.dumps(as_jsonable(summary), indent=2, sort_keys=True)
    print("PHASE9B_RESULT_JSON=" + result_json)
    dbutils.notebook.exit(result_json)


def fail_phase(failed_checks: list[str], message: str, extra: dict[str, Any] | None = None) -> None:
    summary = {
        "phase": "9B",
        "overall_status": "FAIL",
        "failed_checks": failed_checks,
        "message": message,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    if extra:
        summary.update(extra)
    write_json(REPORTS_DIR / "phase9b_summary.json", summary)
    summary["workspace_report_sync"] = sync_reports_to_workspace()
    write_json(REPORTS_DIR / "phase9b_summary.json", summary)
    finish_notebook(summary)


def metric_value(value: Any) -> float | int | None:
    if value is None:
        return None
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        numeric = float(value)
        if not math.isfinite(numeric):
            return None
        return numeric
    return value


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


def assign_splits(base_pdf: pd.DataFrame) -> pd.DataFrame:
    assigned = base_pdf.copy()
    assigned["split"] = "test"
    assigned.loc[assigned.index < TRAIN_END_INDEX_EXCLUSIVE, "split"] = "train"
    assigned.loc[
        (assigned.index >= TRAIN_END_INDEX_EXCLUSIVE)
        & (assigned.index < VALIDATION_END_INDEX_EXCLUSIVE),
        "split",
    ] = "validation"
    return assigned


def build_split_hash(assigned_pdf: pd.DataFrame) -> str:
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
                    str(row.get("_feature_version", FEATURE_VERSION)),
                    SPLIT_VERSION,
                ]
            )
        )
    return hashlib.sha256("\n".join(split_hash_input).encode("utf-8")).hexdigest()


def split_count_block(split_df: pd.DataFrame) -> dict[str, int]:
    fraud_count = int((split_df["_target_int"] == 1).sum())
    normal_count = int((split_df["_target_int"] == 0).sum())
    return {
        "row_count": int(len(split_df)),
        "fraud_count": fraud_count,
        "normal_count": normal_count,
    }


def split_overlap_checks(assigned_pdf: pd.DataFrame) -> dict[str, int]:
    train_ids = set(assigned_pdf.loc[assigned_pdf["split"] == "train", "transaction_id"].astype(str))
    validation_ids = set(assigned_pdf.loc[assigned_pdf["split"] == "validation", "transaction_id"].astype(str))
    test_ids = set(assigned_pdf.loc[assigned_pdf["split"] == "test", "transaction_id"].astype(str))
    return {
        "train_validation_overlap_count": len(train_ids & validation_ids),
        "train_test_overlap_count": len(train_ids & test_ids),
        "validation_test_overlap_count": len(validation_ids & test_ids),
    }


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


def positive_probability(model: Any, X: pd.DataFrame) -> np.ndarray:
    probabilities = model.predict_proba(X)
    classes = list(model.classes_)
    if 1 in classes:
        return probabilities[:, classes.index(1)]
    if True in classes:
        return probabilities[:, classes.index(True)]
    return np.zeros(len(X), dtype=float)


def metrics_at_threshold(y_true: np.ndarray, y_prob: np.ndarray, threshold: float) -> dict[str, Any]:
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "threshold": float(threshold),
        "pr_auc": metric_value(average_precision_score(y_true, y_prob)) if int(np.sum(y_true == 1)) > 0 else None,
        "precision": metric_value(precision_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "recall": metric_value(recall_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "f1": metric_value(f1_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "true_positives": int(tp),
        "false_positives": int(fp),
        "true_negatives": int(tn),
        "false_negatives": int(fn),
        "row_count": int(len(y_true)),
        "fraud_count": int(np.sum(y_true == 1)),
        "normal_count": int(np.sum(y_true == 0)),
    }


def select_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> dict[str, Any]:
    metrics = [metrics_at_threshold(y_true, y_prob, threshold) for threshold in THRESHOLDS]
    eligible = [item for item in metrics if float(item["recall"]) >= MIN_VALIDATION_RECALL]
    if eligible:
        selected = min(
            eligible,
            key=lambda item: (
                int(item["false_positives"]),
                -float(item["precision"]),
                -float(item["threshold"]),
            ),
        )
        status = "PASS"
        rule = "validation recall >= 0.6667, then fewest false positives, higher precision, higher threshold"
    else:
        selected = min(
            metrics,
            key=lambda item: (
                -float(item["recall"]),
                int(item["false_positives"]),
                -float(item["precision"]),
                -float(item["threshold"]),
            ),
        )
        status = "FAIL"
        rule = "no threshold reached validation recall >= 0.6667; fallback kept highest recall for diagnostics"
    return {
        "status": status,
        "selected_threshold": float(selected["threshold"]),
        "selected_metrics": selected,
        "threshold_grid": metrics,
        "selection_rule": rule,
    }


def log_metric_dict(prefix: str, metrics: dict[str, Any]) -> None:
    for key, value in metrics.items():
        if key in {"row_count", "fraud_count", "normal_count", "threshold"}:
            continue
        cleaned = metric_value(value)
        if isinstance(cleaned, (int, float)) and cleaned is not None:
            mlflow.log_metric(f"{prefix}_{key}", float(cleaned))


def log_params(params: dict[str, Any]) -> None:
    for key, value in params.items():
        if value is None:
            continue
        mlflow.log_param(key, value)


def registration_manual_action(error_text: str) -> str:
    lowered = error_text.lower()
    if "access mode" in lowered or "unity catalog" in lowered and "cluster" in lowered:
        return "Use a Unity Catalog-compatible access mode on the existing cluster, then rerun only Phase 9B registration."
    if "create model" in lowered or "permission" in lowered or "privilege" in lowered:
        return "Grant CREATE MODEL plus USE CATALOG/USE SCHEMA on crypto_fraud.models, then rerun only Phase 9B registration."
    if "schema" in lowered and ("not found" in lowered or "does not exist" in lowered):
        return "Create schema crypto_fraud.models or grant CREATE SCHEMA on catalog crypto_fraud, then rerun only Phase 9B registration."
    return "Resolve the reported Unity Catalog Model Registry permission or access blocker, then rerun only Phase 9B registration."


def attempt_uc_registration(model_uri: str) -> dict[str, Any]:
    try:
        mlflow.set_registry_uri("databricks-uc")
        schema_rows = spark.sql("SHOW SCHEMAS IN crypto_fraud LIKE 'models'").collect()
        schema_created = False
        if not schema_rows:
            spark.sql("CREATE SCHEMA IF NOT EXISTS crypto_fraud.models")
            schema_created = True
        registered_model = mlflow.register_model(model_uri, REGISTERED_MODEL_NAME)
        client = mlflow.tracking.MlflowClient()
        client.set_registered_model_alias(REGISTERED_MODEL_NAME, "candidate", registered_model.version)
        return {
            "status": "PASS",
            "registered_model_name": REGISTERED_MODEL_NAME,
            "registered_model_version": str(registered_model.version),
            "candidate_alias_status": "PASS",
            "schema_created": schema_created,
        }
    except Exception as exc:
        error_text = repr(exc)
        return {
            "status": "BLOCKED-MANUAL",
            "registered_model_name": REGISTERED_MODEL_NAME,
            "registered_model_version": None,
            "candidate_alias_status": "NOT_SET",
            "blocker": error_text,
            "manual_action": registration_manual_action(error_text),
        }

# COMMAND ----------

if len(APPROVED_FEATURES) != EXPECTED_FEATURE_COUNT:
    fail_phase(
        ["approved_feature_count_mismatch"],
        "Approved Phase 9A feature list does not contain exactly 55 features.",
        {"approved_feature_count": len(APPROVED_FEATURES)},
    )

source_df = spark.table(TRAINING_TABLE)
source_columns = set(source_df.columns)
required_columns = ["transaction_id", "feature_timestamp", TARGET_COLUMN] + APPROVED_FEATURES
missing_columns = [column for column in required_columns if column not in source_columns]
if missing_columns:
    fail_phase(
        ["required_columns_missing"],
        "Training table does not contain the approved Phase 9A feature and split columns.",
        {"missing_columns": missing_columns},
    )

select_columns = required_columns + (["_feature_version"] if "_feature_version" in source_columns else [])
training_pdf = (
    source_df.select(*select_columns)
    .orderBy("feature_timestamp", "transaction_id")
    .toPandas()
    .reset_index(drop=True)
)
training_pdf["_target_int"] = training_pdf[TARGET_COLUMN].apply(normalize_target)
if training_pdf["_target_int"].isna().any():
    fail_phase(
        ["target_normalization_failed"],
        "target_is_fraud contains values outside the approved binary target semantics.",
        {"null_or_invalid_target_count": int(training_pdf["_target_int"].isna().sum())},
    )

assigned_pdf = assign_splits(training_pdf)
split_hash = build_split_hash(assigned_pdf)
train_pdf = assigned_pdf[assigned_pdf["split"] == "train"].copy()
validation_pdf = assigned_pdf[assigned_pdf["split"] == "validation"].copy()
test_pdf = assigned_pdf[assigned_pdf["split"] == "test"].copy()

split_summary = {
    "train": split_count_block(train_pdf),
    "validation": split_count_block(validation_pdf),
    "test": split_count_block(test_pdf),
}
overlap_checks = split_overlap_checks(assigned_pdf)

split_failed_checks = []
for split_name, expected_counts in EXPECTED_SPLIT_COUNTS.items():
    for key, expected_value in expected_counts.items():
        if split_summary[split_name][key] != expected_value:
            split_failed_checks.append(f"{split_name}_{key}_mismatch")
if any(value != 0 for value in overlap_checks.values()):
    split_failed_checks.append("transaction_overlap_detected")
if split_hash != APPROVED_SPLIT_HASH:
    split_failed_checks.append("split_hash_mismatch")

if split_failed_checks:
    fail_phase(
        split_failed_checks,
        "Approved Phase 9A rare_event_aware_time_split_v2 did not reproduce exactly; stopped before training.",
        {
            "split_version": SPLIT_VERSION,
            "approved_split_hash": APPROVED_SPLIT_HASH,
            "actual_split_hash": split_hash,
            "split_summary": split_summary,
            "overlap_checks": overlap_checks,
        },
    )

# COMMAND ----------

X_train = train_pdf[APPROVED_FEATURES].copy()
y_train = train_pdf["_target_int"].astype(int).to_numpy()
X_validation = validation_pdf[APPROVED_FEATURES].copy()
y_validation = validation_pdf["_target_int"].astype(int).to_numpy()
X_test = test_pdf[APPROVED_FEATURES].copy()
y_test = test_pdf["_target_int"].astype(int).to_numpy()

numeric_conversion_issues: dict[str, dict[str, int]] = {}
numeric_training_all_null = []
for feature_name in NUMERIC_FEATURES:
    split_issues: dict[str, int] = {}
    for split_name, frame in [("train", X_train), ("validation", X_validation), ("test", X_test)]:
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
    fail_phase(
        preprocessing_failed_checks,
        "Phase 9A preprocessing checks failed; stopped before training.",
        {
            "numeric_training_all_null": numeric_training_all_null,
            "numeric_conversion_issues": numeric_conversion_issues,
        },
    )

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

binary_features = [feature_name for feature_name in NUMERIC_FEATURES if is_binary_indicator_name(feature_name)]
scale_pos_weight = float(split_summary["train"]["normal_count"] / split_summary["train"]["fraud_count"])

XGB_PARAMS = {
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
    "scale_pos_weight": scale_pos_weight,
    "verbosity": 0,
}

xgb_pipeline = Pipeline(
    steps=[
        ("preprocess", preprocessor),
        ("model", XGBClassifier(**XGB_PARAMS)),
    ]
)

# COMMAND ----------

mlflow.set_experiment(EXPERIMENT_PATH)

with mlflow.start_run(run_name=RUN_NAME) as xgb_run:
    xgb_run_id = xgb_run.info.run_id
    xgb_pipeline.fit(X_train, y_train)
    transformed_feature_count = int(xgb_pipeline.named_steps["preprocess"].transform(X_train.head(1)).shape[1])

    xgb_validation_prob = positive_probability(xgb_pipeline, X_validation)
    xgb_threshold_selection = select_threshold(y_validation, xgb_validation_prob)

    logistic_model = mlflow.sklearn.load_model(LOGISTIC_MODEL_URI)
    logistic_validation_prob = positive_probability(logistic_model, X_validation)
    logistic_threshold_selection = select_threshold(y_validation, logistic_validation_prob)

    xgb_validation_metrics = xgb_threshold_selection["selected_metrics"]
    logistic_validation_metrics = logistic_threshold_selection["selected_metrics"]

    candidates = [
        {
            "model": "logistic_regression",
            "selected_threshold": logistic_threshold_selection["selected_threshold"],
            "metrics": logistic_validation_metrics,
            "model_uri": LOGISTIC_MODEL_URI,
        },
        {
            "model": "xgboost",
            "selected_threshold": xgb_threshold_selection["selected_threshold"],
            "metrics": xgb_validation_metrics,
            "model_uri": f"runs:/{xgb_run_id}/model",
        },
    ]
    eligible_candidates = [
        candidate
        for candidate in candidates
        if float(candidate["metrics"]["recall"]) >= MIN_VALIDATION_RECALL
    ]
    if not eligible_candidates:
        comparison = {
            "phase": "9B",
            "overall_status": "FAIL",
            "logistic_regression": {
                "validation_metrics": logistic_validation_metrics,
                "selected_threshold": logistic_threshold_selection["selected_threshold"],
                "threshold_selection": logistic_threshold_selection,
            },
            "xgboost": {
                "validation_metrics": xgb_validation_metrics,
                "selected_threshold": xgb_threshold_selection["selected_threshold"],
                "threshold_selection": xgb_threshold_selection,
            },
            "selected_model": None,
            "selection_reason": "No candidate achieved validation recall >= 0.6667.",
            "final_test_metrics": None,
        }
        mlflow.log_dict(as_jsonable(comparison), "phase9b_model_comparison.json")
        write_json(REPORTS_DIR / "phase9b_model_comparison.json", comparison)
        summary = {
            "phase": "9B",
            "overall_status": "FAIL",
            "failed_checks": ["no_model_met_validation_recall_floor"],
            "notebook_path": WORKSPACE_NOTEBOOK_PATH,
            "xgboost_run_id": xgb_run_id,
            "selected_model": None,
            "selected_model_uri": None,
            "comparison": comparison,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
        }
        write_json(REPORTS_DIR / "phase9b_summary.json", summary)
        summary["workspace_report_sync"] = sync_reports_to_workspace()
        write_json(REPORTS_DIR / "phase9b_summary.json", summary)
        finish_notebook(summary)

    selected_candidate = min(
        eligible_candidates,
        key=lambda candidate: (
            int(candidate["metrics"]["false_positives"]),
            -float(candidate["metrics"]["pr_auc"]),
            -float(candidate["metrics"]["precision"]),
        ),
    )
    selected_model = selected_candidate["model"]
    selected_threshold = float(selected_candidate["selected_threshold"])
    selected_model_uri = str(selected_candidate["model_uri"])

    if selected_model == "xgboost":
        selected_test_model = xgb_pipeline
    else:
        selected_test_model = logistic_model
    selected_test_prob = positive_probability(selected_test_model, X_test)
    final_test_metrics = metrics_at_threshold(y_test, selected_test_prob, selected_threshold)

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

    xgb_model_uri = f"runs:/{xgb_run_id}/model"
    if selected_model == "xgboost":
        selected_model_uri = xgb_model_uri

    selection_reason = (
        f"{selected_model} selected by validation priority: recall >= 0.6667, "
        "then lower false positives, higher PR-AUC, higher precision."
    )

    registration_result = attempt_uc_registration(selected_model_uri)
    overall_status = "PASS" if registration_result["status"] == "PASS" else "BLOCKED-MANUAL"

    comparison = {
        "phase": "9B",
        "logistic_regression": {
            "model_uri": LOGISTIC_MODEL_URI,
            "validation_metrics": logistic_validation_metrics,
            "selected_threshold": logistic_threshold_selection["selected_threshold"],
            "threshold_selection": logistic_threshold_selection,
            "retrained": False,
        },
        "xgboost": {
            "model_uri": xgb_model_uri,
            "validation_metrics": xgb_validation_metrics,
            "selected_threshold": xgb_threshold_selection["selected_threshold"],
            "threshold_selection": xgb_threshold_selection,
            "fixed_parameters": XGB_PARAMS,
        },
        "selected_model": selected_model,
        "selected_threshold": selected_threshold,
        "selected_model_uri": selected_model_uri,
        "selection_reason": selection_reason,
        "final_test_metrics": final_test_metrics,
        "test_set_warning": "The locked test split contains only 3 fraud cases, so performance is unstable and portfolio-level, not production-level.",
    }

    mlflow.log_dict(as_jsonable(comparison), "phase9b_model_comparison.json")
    log_params(
        {
            "model_type": "XGBClassifier",
            "objective": XGB_PARAMS["objective"],
            "eval_metric": XGB_PARAMS["eval_metric"],
            "n_estimators": XGB_PARAMS["n_estimators"],
            "max_depth": XGB_PARAMS["max_depth"],
            "learning_rate": XGB_PARAMS["learning_rate"],
            "subsample": XGB_PARAMS["subsample"],
            "colsample_bytree": XGB_PARAMS["colsample_bytree"],
            "min_child_weight": XGB_PARAMS["min_child_weight"],
            "reg_lambda": XGB_PARAMS["reg_lambda"],
            "random_state": XGB_PARAMS["random_state"],
            "n_jobs": XGB_PARAMS["n_jobs"],
            "tree_method": XGB_PARAMS["tree_method"],
            "scale_pos_weight": scale_pos_weight,
            "feature_count": len(APPROVED_FEATURES),
            "transformed_feature_count": transformed_feature_count,
            "split_version": SPLIT_VERSION,
            "split_hash": split_hash,
        }
    )
    log_metric_dict("validation", xgb_validation_metrics)
    mlflow.log_metric("validation_selected_threshold", float(xgb_threshold_selection["selected_threshold"]))
    mlflow.set_tag("phase", "9B")
    mlflow.set_tag("selected_model", selected_model)
    mlflow.set_tag("selected_model_uri", selected_model_uri)
    mlflow.set_tag("registration_status", registration_result["status"])
    mlflow.set_tag("phase10_started", "false")
    mlflow.set_tag("event_hubs_started", "false")

    preprocessing_summary = {
        "preprocessing_fit_scope": "training split only",
        "numeric_features": NUMERIC_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "binary_features": binary_features,
        "numeric_transformer": "NumericCoercer, SimpleImputer(strategy='median'), StandardScaler",
        "categorical_transformer": "CategoricalCleaner('__MISSING__'), OneHotEncoder(handle_unknown='ignore')",
        "final_transformed_feature_dimension": transformed_feature_count,
    }

    quality_gate_checks = {
        "approved_phase9a_split_reused_exactly": split_hash == APPROVED_SPLIT_HASH,
        "approved_feature_set_reused": len(APPROVED_FEATURES) == EXPECTED_FEATURE_COUNT,
        "preprocessing_fit_on_train_only": True,
        "one_xgboost_model_trained": True,
        "logistic_regression_retrained": False,
        "threshold_selected_using_validation_only": True,
        "test_used_only_after_model_and_threshold_selection": True,
        "xgboost_run_logged_to_mlflow": bool(xgb_run_id),
        "selected_model_artifact_available": bool(selected_model_uri),
        "phase10_started": False,
        "event_hubs_started": False,
    }

    summary = {
        "phase": "9B",
        "overall_status": overall_status,
        "notebook_path": WORKSPACE_NOTEBOOK_PATH,
        "cluster": {
            "cluster_id": "0803-061312-78fw66xn",
            "cluster_name": "crypto-fraud-dev-compute",
        },
        "mlflow_experiment_path": EXPERIMENT_PATH,
        "xgboost_run_id": xgb_run_id,
        "xgboost_model_uri": xgb_model_uri,
        "selected_model": selected_model,
        "selected_threshold": selected_threshold,
        "selected_model_uri": selected_model_uri,
        "selection_reason": selection_reason,
        "validation_metrics": {
            "logistic_regression": logistic_validation_metrics,
            "xgboost": xgb_validation_metrics,
        },
        "selected_thresholds": {
            "logistic_regression": logistic_threshold_selection["selected_threshold"],
            "xgboost": xgb_threshold_selection["selected_threshold"],
        },
        "final_locked_test_metrics": final_test_metrics,
        "test_set_warning": comparison["test_set_warning"],
        "split_version": SPLIT_VERSION,
        "split_hash": split_hash,
        "split_summary": split_summary,
        "overlap_checks": overlap_checks,
        "feature_count": len(APPROVED_FEATURES),
        "preprocessing_summary": preprocessing_summary,
        "xgboost_fixed_parameters": XGB_PARAMS,
        "scale_pos_weight": scale_pos_weight,
        "package_versions": {
            "python": sys.version.split()[0],
            "python_full": sys.version,
            "platform": platform.platform(),
            "xgboost": xgboost.__version__,
            "scikit_learn": sklearn.__version__,
            "mlflow": mlflow.__version__,
            "pandas": pd.__version__,
            "numpy": np.__version__,
            "package_status_before_install": package_status_before_install,
            "package_status_after_install": package_status_after_install,
            "notebook_scoped_packages_installed": installed_packages,
        },
        "registration": registration_result,
        "quality_gate_checks": quality_gate_checks,
        "failed_checks": [],
        "artifacts_created": {
            "reports": [
                "reports/phase9b_model_comparison.json",
                "reports/phase9b_summary.json",
            ],
            "mlflow_comparison_artifact": "phase9b_model_comparison.json",
        },
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }

    write_json(REPORTS_DIR / "phase9b_model_comparison.json", comparison)
    write_json(REPORTS_DIR / "phase9b_summary.json", summary)
    summary["workspace_report_sync"] = sync_reports_to_workspace()
    write_json(REPORTS_DIR / "phase9b_summary.json", summary)
    sync_reports_to_workspace()

finish_notebook(summary)
