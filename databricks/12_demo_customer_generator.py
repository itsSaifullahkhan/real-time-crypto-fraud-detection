# Databricks notebook source
# MAGIC %md
# MAGIC # 12 Demo Customer Generator

# COMMAND ----------

from __future__ import annotations

import argparse
import importlib
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone

dbutils.widgets.text("demo_duration_seconds", "300")
dbutils.widgets.text("target_transactions", "25")
dbutils.widgets.text("fraud_rate", "0.10")
dbutils.widgets.text("label_delay_seconds", "10")
dbutils.widgets.text("producer_start_delay_seconds", "45")
dbutils.widgets.text("force_fraud_count", "1")
dbutils.widgets.text("random_seed", "42")
dbutils.widgets.text("validate_schema", "true")

DEMO_DURATION_SECONDS = int(dbutils.widgets.get("demo_duration_seconds") or "300")
TARGET_TRANSACTIONS = int(dbutils.widgets.get("target_transactions") or "25")
FRAUD_RATE = float(dbutils.widgets.get("fraud_rate") or "0.10")
LABEL_DELAY_SECONDS = float(dbutils.widgets.get("label_delay_seconds") or "10")
START_DELAY_SECONDS = int(dbutils.widgets.get("producer_start_delay_seconds") or "45")
FORCE_FRAUD_COUNT = int(dbutils.widgets.get("force_fraud_count") or "1")
RANDOM_SEED = int(dbutils.widgets.get("random_seed") or "42")
VALIDATE_SCHEMA = (dbutils.widgets.get("validate_schema") or "true").strip().lower() == "true"
PACKAGE_WHEEL_PATH = (
    "/Workspace/Users/akanaskhan1506@gmail.com/crypto-fraud-platform/dist/"
    "crypto_fraud_platform-0.1.0-py3-none-any.whl"
)
WORKSPACE_PROJECT_ROOT = "/Workspace/Users/akanaskhan1506@gmail.com/crypto-fraud-platform"


def ensure_demo_package() -> None:
    try:
        importlib.import_module("crypto_fraud_platform")
        return
    except ImportError:
        pass
    subprocess.check_call(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--disable-pip-version-check",
            "--quiet",
            PACKAGE_WHEEL_PATH,
            "azure-eventhub>=5.13,<6",
            "jsonschema>=4.22,<5",
            "PyYAML>=6.0,<7",
            "python-dotenv>=1.0,<2",
        ]
    )


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def set_task_value(key: str, value: object) -> None:
    try:
        dbutils.jobs.taskValues.set(key=key, value=json.dumps(value, sort_keys=True, separators=(",", ":")))
    except Exception:
        pass


connection_string = dbutils.secrets.get(
    scope="crypto-fraud-secrets",
    key="eventhubs-databricks-connection",
)
os.environ["EVENT_HUB_CONNECTION_STRING"] = connection_string
os.environ.setdefault("EVENT_HUB_TRANSACTIONS", "transaction-events")
os.environ.setdefault("EVENT_HUB_AUTHENTICATION", "authentication-events")
os.environ.setdefault("EVENT_HUB_FRAUD_LABELS", "fraud-labels")
os.environ["CRYPTO_FRAUD_PROJECT_ROOT"] = WORKSPACE_PROJECT_ROOT

if START_DELAY_SECONDS > 0:
    time.sleep(START_DELAY_SECONDS)

ensure_demo_package()

from crypto_fraud_platform.live_generators import live_customer_generator

interval_seconds = 0.0
if TARGET_TRANSACTIONS > 1:
    interval_seconds = max(float(DEMO_DURATION_SECONDS) / float(TARGET_TRANSACTIONS), 0.1)

started_at = now_utc()
args = argparse.Namespace(
    transaction_count=TARGET_TRANSACTIONS,
    interval=interval_seconds,
    fraud_rate=FRAUD_RATE,
    label_delay=LABEL_DELAY_SECONDS,
    random_seed=RANDOM_SEED,
    force_fraud_count=FORCE_FRAUD_COUNT,
    validate_schema=VALIDATE_SCHEMA,
)
return_code = live_customer_generator.run(args)
finished_at = now_utc()

summary = {
    "task": "customer_generator",
    "status": "PASS" if return_code == 0 else "FAIL",
    "return_code": int(return_code),
    "transaction_event_hub": os.environ["EVENT_HUB_TRANSACTIONS"],
    "authentication_event_hub": os.environ["EVENT_HUB_AUTHENTICATION"],
    "fraud_label_event_hub": os.environ["EVENT_HUB_FRAUD_LABELS"],
    "demo_duration_seconds": DEMO_DURATION_SECONDS,
    "target_transactions": TARGET_TRANSACTIONS,
    "fraud_rate": FRAUD_RATE,
    "label_delay_seconds": LABEL_DELAY_SECONDS,
    "producer_start_delay_seconds": START_DELAY_SECONDS,
    "force_fraud_count": FORCE_FRAUD_COUNT,
    "interval_seconds": interval_seconds,
    "started_at_utc": started_at,
    "finished_at_utc": finished_at,
    "producer_stopped": True,
    "fraud_labels_are_delayed_feedback_only": True,
    "secrets_exposed": False,
}
set_task_value("summary", summary)

if return_code != 0:
    raise RuntimeError("Live customer generator failed")

dbutils.notebook.exit(json.dumps(summary, sort_keys=True, separators=(",", ":")))
