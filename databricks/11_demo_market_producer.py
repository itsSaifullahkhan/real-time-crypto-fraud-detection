# Databricks notebook source
# MAGIC %md
# MAGIC # 11 Demo Market Producer

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
dbutils.widgets.text("producer_start_delay_seconds", "45")
dbutils.widgets.text("validate_schema", "true")
dbutils.widgets.text("read_timeout_seconds", "20")

DEMO_DURATION_SECONDS = int(dbutils.widgets.get("demo_duration_seconds") or "300")
START_DELAY_SECONDS = int(dbutils.widgets.get("producer_start_delay_seconds") or "45")
VALIDATE_SCHEMA = (dbutils.widgets.get("validate_schema") or "true").strip().lower() == "true"
READ_TIMEOUT_SECONDS = int(dbutils.widgets.get("read_timeout_seconds") or "20")
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
            "websocket-client>=1.8,<2",
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
os.environ.setdefault("EVENT_HUB_MARKET", "market-events")
os.environ["CRYPTO_FRAUD_PROJECT_ROOT"] = WORKSPACE_PROJECT_ROOT

if START_DELAY_SECONDS > 0:
    time.sleep(START_DELAY_SECONDS)

ensure_demo_package()

from crypto_fraud_platform.websocket_collector import coinbase_market_producer

started_at = now_utc()
args = argparse.Namespace(
    max_events=0,
    max_seconds=DEMO_DURATION_SECONDS,
    read_timeout=READ_TIMEOUT_SECONDS,
    reconnect_delay=coinbase_market_producer.RECONNECT_DELAY_SECONDS,
    require_products=False,
    validate_schema=VALIDATE_SCHEMA,
)
return_code = coinbase_market_producer.run(args)
finished_at = now_utc()

summary = {
    "task": "market_producer",
    "status": "PASS" if return_code == 0 else "FAIL",
    "return_code": int(return_code),
    "event_hub": os.environ["EVENT_HUB_MARKET"],
    "demo_duration_seconds": DEMO_DURATION_SECONDS,
    "producer_start_delay_seconds": START_DELAY_SECONDS,
    "started_at_utc": started_at,
    "finished_at_utc": finished_at,
    "producer_stopped": True,
    "secrets_exposed": False,
}
set_task_value("summary", summary)

if return_code != 0:
    raise RuntimeError("Coinbase market producer failed")

dbutils.notebook.exit(json.dumps(summary, sort_keys=True, separators=(",", ":")))
