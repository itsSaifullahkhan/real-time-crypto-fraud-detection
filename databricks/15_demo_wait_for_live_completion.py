# Databricks notebook source
# MAGIC %md
# MAGIC # 15 Demo Wait For Live Completion

# COMMAND ----------

from __future__ import annotations

import json
import time
from datetime import datetime, timezone

dbutils.widgets.text("label_settle_seconds", "15")
LABEL_SETTLE_SECONDS = int(dbutils.widgets.get("label_settle_seconds") or "15")


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


started_at = now_utc()
if LABEL_SETTLE_SECONDS > 0:
    time.sleep(LABEL_SETTLE_SECONDS)
finished_at = now_utc()

summary = {
    "task": "wait_for_live_completion",
    "status": "PASS",
    "label_settle_seconds": LABEL_SETTLE_SECONDS,
    "started_at_utc": started_at,
    "finished_at_utc": finished_at,
}

try:
    dbutils.jobs.taskValues.set(key="summary", value=json.dumps(summary, sort_keys=True, separators=(",", ":")))
except Exception:
    pass

dbutils.notebook.exit(json.dumps(summary, sort_keys=True, separators=(",", ":")))
