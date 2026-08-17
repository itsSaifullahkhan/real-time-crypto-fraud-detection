# Databricks notebook source
# MAGIC %md
# MAGIC # 14 Demo Prepare

# COMMAND ----------

from __future__ import annotations

import importlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from typing import Any

import mlflow

from pyspark.sql import functions as F

spark.conf.set("spark.sql.session.timeZone", "UTC")
spark.conf.set("spark.sql.shuffle.partitions", "1")

dbutils.widgets.text("demo_duration_seconds", "300")
dbutils.widgets.text("target_transactions", "25")
dbutils.widgets.text("fraud_rate", "0.10")
dbutils.widgets.text("enable_alerting", "true")
dbutils.widgets.text("send_test_alert", "true")
dbutils.widgets.text(
    "package_wheel_path",
    "/Workspace/Users/akanaskhan1506@gmail.com/crypto-fraud-platform/dist/crypto_fraud_platform-0.1.0-py3-none-any.whl",
)
dbutils.widgets.text("install_dependencies", "true")
dbutils.widgets.text("alert_secret_scope", "crypto-fraud-secrets")
dbutils.widgets.text("alert_webhook_secret_key", "fraud-alert-webhook-url")

PACKAGE_WHEEL_PATH = dbutils.widgets.get("package_wheel_path")
INSTALL_DEPENDENCIES = (dbutils.widgets.get("install_dependencies") or "true").strip().lower() == "true"
ALERT_SECRET_SCOPE = dbutils.widgets.get("alert_secret_scope") or "crypto-fraud-secrets"
ALERT_WEBHOOK_SECRET_KEY = dbutils.widgets.get("alert_webhook_secret_key") or "fraud-alert-webhook-url"

BRONZE_TABLES = {
    "market": "crypto_fraud.bronze.live_market_events",
    "transactions": "crypto_fraud.bronze.live_customer_transactions",
    "authentication": "crypto_fraud.bronze.live_authentication_events",
    "fraud_labels": "crypto_fraud.bronze.live_fraud_labels",
    "fraud_decisions": "crypto_fraud.bronze.live_fraud_decisions",
}
REGISTERED_MODEL_NAME = "crypto_fraud.models.fraud_detection_model"


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def set_task_value(key: str, value: object) -> None:
    try:
        dbutils.jobs.taskValues.set(key=key, value=json.dumps(value, sort_keys=True, separators=(",", ":")))
    except Exception:
        pass


def package_available(name: str) -> bool:
    try:
        importlib.import_module(name)
        return True
    except ImportError:
        return False


def table_exists(full_name: str) -> bool:
    return spark.catalog.tableExists(full_name)


def table_count(full_name: str) -> int:
    return int(spark.table(full_name).count()) if table_exists(full_name) else 0


if INSTALL_DEPENDENCIES:
    packages = [
        PACKAGE_WHEEL_PATH,
        "azure-eventhub>=5.13,<6",
        "websocket-client>=1.8,<2",
        "jsonschema>=4.22,<5",
        "PyYAML>=6.0,<7",
        "python-dotenv>=1.0,<2",
        "requests>=2.31,<3",
    ]
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "--disable-pip-version-check", "--quiet", *packages]
    )

missing_after_install = [
    package_name
    for package_name, import_name in {
        "crypto-fraud-platform": "crypto_fraud_platform",
        "azure-eventhub": "azure.eventhub",
        "websocket-client": "websocket",
        "jsonschema": "jsonschema",
        "PyYAML": "yaml",
        "requests": "requests",
    }.items()
    if not package_available(import_name)
]
if missing_after_install:
    raise RuntimeError(f"Missing demo dependencies after install: {missing_after_install}")

try:
    dbutils.secrets.get(scope="crypto-fraud-secrets", key="eventhubs-databricks-connection")
    eventhub_secret_available = True
except Exception:
    eventhub_secret_available = False

try:
    dbutils.secrets.get(scope=ALERT_SECRET_SCOPE, key=ALERT_WEBHOOK_SECRET_KEY)
    alert_webhook_secret_available = True
except Exception:
    alert_webhook_secret_available = False

try:
    mlflow.set_registry_uri("databricks-uc")
    client = mlflow.tracking.MlflowClient()
    candidate_alias_before = str(client.get_model_version_by_alias(REGISTERED_MODEL_NAME, "candidate").version)
except Exception:
    candidate_alias_before = None

pre_counts = {name: table_count(table_name) for name, table_name in BRONZE_TABLES.items()}
demo_start_timestamp = now_utc()

summary: dict[str, Any] = {
    "task": "prepare_demo",
    "status": "PASS" if eventhub_secret_available else "FAIL",
    "demo_start_timestamp": demo_start_timestamp,
    "parameters": {
        "demo_duration_seconds": dbutils.widgets.get("demo_duration_seconds"),
        "target_transactions": dbutils.widgets.get("target_transactions"),
        "fraud_rate": dbutils.widgets.get("fraud_rate"),
        "enable_alerting": dbutils.widgets.get("enable_alerting"),
        "send_test_alert": dbutils.widgets.get("send_test_alert"),
    },
    "package_wheel_path": PACKAGE_WHEEL_PATH,
    "dependencies_installed": INSTALL_DEPENDENCIES,
    "pre_bronze_counts": pre_counts,
    "eventhub_secret_available": eventhub_secret_available,
    "alert_webhook_secret_available": alert_webhook_secret_available,
    "candidate_alias_before": candidate_alias_before,
    "threshold": 0.80,
    "secrets_exposed": False,
}

set_task_value("demo_start_timestamp", demo_start_timestamp)
set_task_value("pre_bronze_counts", pre_counts)
set_task_value("summary", summary)

if not eventhub_secret_available:
    raise RuntimeError("Required Event Hubs Databricks secret is not available")

dbutils.notebook.exit(json.dumps(summary, sort_keys=True, separators=(",", ":")))
