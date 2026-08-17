# Databricks notebook source
# MAGIC %md
# MAGIC # 13 Demo Fraud Alert Monitor

# COMMAND ----------

from __future__ import annotations

import json
import importlib
import re
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
from datetime import datetime, timezone
from typing import Any

PACKAGE_WHEEL_PATH = (
    "/Workspace/Users/akanaskhan1506@gmail.com/crypto-fraud-platform/dist/"
    "crypto_fraud_platform-0.1.0-py3-none-any.whl"
)


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
            "requests>=2.31,<3",
        ]
    )


ensure_demo_package()

from crypto_fraud_platform.alerting import (
    ALERT_THRESHOLD,
    WebhookNotifier,
    build_alert_payload,
    build_test_alert_payload,
    is_high_risk_fraud_decision,
)

dbutils.widgets.text("enable_alerting", "true")
dbutils.widgets.text("send_test_alert", "true")
dbutils.widgets.text("timeout_seconds", "405")
dbutils.widgets.text("real_time_trigger", "5 seconds")
dbutils.widgets.text("alert_secret_scope", "crypto-fraud-secrets")
dbutils.widgets.text("alert_webhook_secret_key", "fraud-alert-webhook-url")

ENABLE_ALERTING = (dbutils.widgets.get("enable_alerting") or "true").strip().lower() == "true"
SEND_TEST_ALERT = (dbutils.widgets.get("send_test_alert") or "true").strip().lower() == "true"
TIMEOUT_SECONDS = int(dbutils.widgets.get("timeout_seconds") or "405")
REAL_TIME_TRIGGER = dbutils.widgets.get("real_time_trigger") or "5 seconds"
ALERT_SECRET_SCOPE = dbutils.widgets.get("alert_secret_scope") or "crypto-fraud-secrets"
ALERT_WEBHOOK_SECRET_KEY = dbutils.widgets.get("alert_webhook_secret_key") or "fraud-alert-webhook-url"

EVENTHUB_SECRET_SCOPE = "crypto-fraud-secrets"
EVENTHUB_SECRET_KEY = "eventhubs-databricks-connection"
FRAUD_DECISION_TOPIC = "fraud-decisions"
CONSUMER_GROUP = "stream-processing"


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def set_task_value(key: str, value: object) -> None:
    try:
        dbutils.jobs.taskValues.set(key=key, value=json.dumps(value, sort_keys=True, separators=(",", ":")))
    except Exception:
        pass


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
        match = re.search(r"sr=https%3A%2F%2F([^%/;\s\"']+)", joined, flags=re.IGNORECASE)
    if not match:
        raise ValueError("Event Hubs namespace endpoint missing from secret")
    return f"{match.group(1)}:9093"


def optional_webhook_url() -> str | None:
    try:
        value = dbutils.secrets.get(scope=ALERT_SECRET_SCOPE, key=ALERT_WEBHOOK_SECRET_KEY)
    except Exception:
        return None
    value = str(value or "").strip()
    return value or None


started_at = now_utc()
webhook_url = optional_webhook_url() if ENABLE_ALERTING else None
notifier = WebhookNotifier(webhook_url) if webhook_url else None

stats: dict[str, Any] = {
    "task": "alert_monitor",
    "status": "DISABLED" if not ENABLE_ALERTING else "PASS" if webhook_url else "BLOCKED_MANUAL",
    "enable_alerting": ENABLE_ALERTING,
    "event_hub": FRAUD_DECISION_TOPIC,
    "consumer_group": CONSUMER_GROUP,
    "threshold": ALERT_THRESHOLD,
    "timeout_seconds": TIMEOUT_SECONDS,
    "decisions_observed": 0,
    "high_risk_alerts_detected": 0,
    "alerts_sent": 0,
    "notification_failures": 0,
    "test_alert_supported": True,
    "test_alert_result": "NOT_REQUESTED",
    "external_alert_channel": (
        f"Databricks secret {ALERT_SECRET_SCOPE}/{ALERT_WEBHOOK_SECRET_KEY}"
        if webhook_url
        else f"BLOCKED_MANUAL: create Databricks secret {ALERT_SECRET_SCOPE}/{ALERT_WEBHOOK_SECRET_KEY}"
    ),
    "started_at_utc": started_at,
    "finished_at_utc": None,
    "monitor_stopped": False,
    "secrets_exposed": False,
}

if SEND_TEST_ALERT:
    if notifier is None:
        stats["test_alert_result"] = "BLOCKED_MANUAL"
    else:
        try:
            notifier.send(build_test_alert_payload())
            stats["test_alert_result"] = "PASS"
        except Exception as exc:
            stats["test_alert_result"] = f"FAIL:{type(exc).__name__}"
            stats["notification_failures"] += 1
            stats["status"] = "FAIL"

seen_event_ids: set[str] = set()


def process_event(raw_payload: str) -> None:
    stats["decisions_observed"] += 1
    try:
        event = json.loads(raw_payload)
    except Exception:
        return
    event_id = str(event.get("event_id") or event.get("transaction_id") or "")
    if event_id in seen_event_ids:
        return
    if not is_high_risk_fraud_decision(event):
        return
    seen_event_ids.add(event_id)
    stats["high_risk_alerts_detected"] += 1
    if notifier is None:
        return
    try:
        notifier.send(build_alert_payload(event))
        stats["alerts_sent"] += 1
    except Exception:
        stats["notification_failures"] += 1
        stats["status"] = "FAIL"


def on_event(partition_context, event) -> None:
    if event is None:
        return
    process_event(event.body_as_str(encoding="UTF-8"))
    try:
        partition_context.update_checkpoint(event)
    except Exception:
        pass


client = None
if ENABLE_ALERTING:
    from azure.eventhub import EventHubConsumerClient

    connection_string = normalize_connection_string(
        dbutils.secrets.get(scope=EVENTHUB_SECRET_SCOPE, key=EVENTHUB_SECRET_KEY)
    )
    client = EventHubConsumerClient.from_connection_string(
        conn_str=connection_string,
        consumer_group=CONSUMER_GROUP,
        eventhub_name=FRAUD_DECISION_TOPIC,
    )

try:
    if client is not None:
        def receive_events() -> None:
            client.receive(
                on_event=on_event,
                starting_position="@latest",
                max_wait_time=5.0,
            )

        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(receive_events)
            try:
                future.result(timeout=TIMEOUT_SECONDS)
            except FutureTimeoutError:
                try:
                    client.close()
                except Exception:
                    pass
                try:
                    future.result(timeout=30)
                except Exception:
                    pass
finally:
    if client is not None:
        try:
            client.close()
        except Exception:
            pass

stats["finished_at_utc"] = now_utc()
stats["monitor_stopped"] = True
set_task_value("summary", stats)
dbutils.notebook.exit(json.dumps(stats, sort_keys=True, separators=(",", ":")))
