from __future__ import annotations

import json
import re
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EXPECTED_FEATURE_COUNT = 55
EXPECTED_THRESHOLD = "0.80"
EXPECTED_TOPICS = {
    "market-events",
    "transaction-events",
    "authentication-events",
    "fraud-labels",
    "fraud-decisions",
}
EXPECTED_GOLD_TABLES = {
    "crypto_fraud.gold.fraud_transaction_decisions",
    "crypto_fraud.gold.fraud_kpi_summary",
    "crypto_fraud.gold.fraud_activity_timeseries",
    "crypto_fraud.gold.model_monitoring_summary",
}
LEAKAGE_TERMS = {
    "is_fraud",
    "target",
    "fraud_type",
    "fraud_label",
    "label_timestamp",
    "label_source",
    "investigation_status",
}


def rel(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def fail(message: str) -> None:
    print(f"FAIL: {message}")
    raise SystemExit(1)


def assert_true(condition: bool, message: str) -> None:
    if not condition:
        fail(message)


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def load_json(path: Path):
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def model_input_features() -> list[str]:
    payload = load_json(ROOT / "config" / "feature_definitions.json")
    candidates = payload.get("model_input_candidates")
    if not isinstance(candidates, list):
        fail("config/feature_definitions.json missing model_input_candidates list")
    features = []
    for item in candidates:
        if isinstance(item, str):
            features.append(item)
        elif isinstance(item, dict):
            name = item.get("name") or item.get("feature_name")
            if name:
                features.append(str(name))
    return features


def validate_python_syntax() -> int:
    checked = 0
    for folder in ["src", "scripts", "databricks"]:
        for path in (ROOT / folder).rglob("*.py"):
            source = read_text(path)
            compile(source, str(path), "exec")
            checked += 1
    return checked


def validate_json_files() -> int:
    checked = 0
    for folder in ["config", "reports"]:
        root = ROOT / folder
        if not root.exists():
            continue
        for path in root.rglob("*.json"):
            load_json(path)
            checked += 1
    return checked


def validate_feature_contract() -> None:
    features = model_input_features()
    assert_true(len(features) == EXPECTED_FEATURE_COUNT, f"expected 55 model features, found {len(features)}")
    assert_true(len(set(features)) == EXPECTED_FEATURE_COUNT, "model feature list contains duplicates")
    leaked = [name for name in features if any(term in name.lower() for term in LEAKAGE_TERMS)]
    assert_true(not leaked, f"fraud-label leakage terms present in feature names: {leaked}")


def validate_threshold_and_alias_safety() -> None:
    phase13 = read_text(ROOT / "databricks" / "06_realtime_fraud_scoring.py")
    phase16 = read_text(ROOT / "databricks" / "09_feedback_retraining.py")
    phase17 = read_text(ROOT / "databricks" / "10_gold_analytics.py")
    assert_true(f"THRESHOLD = {EXPECTED_THRESHOLD}" in phase13, "Phase 13 threshold is not 0.80")
    assert_true(f"THRESHOLD = {EXPECTED_THRESHOLD}" in phase16, "Phase 16 threshold is not 0.80")
    assert_true(f"THRESHOLD = {EXPECTED_THRESHOLD}" in phase17, "Phase 17 threshold is not 0.80")
    assert_true("@candidate" in phase13, "Phase 13 does not load the candidate alias")
    assert_true("set_registered_model_alias" not in phase16, "Phase 16 attempts to move a registry alias")
    assert_true("set_registered_model_alias" not in phase17, "Phase 17 attempts to move a registry alias")


def validate_eventhub_and_table_refs() -> None:
    combined = "\n".join(
        read_text(ROOT / path)
        for path in [
            Path("src/crypto_fraud_platform/live_generators/live_customer_generator.py"),
            Path("src/crypto_fraud_platform/websocket_collector/coinbase_market_producer.py"),
            Path("databricks/06_realtime_fraud_scoring.py"),
            Path("databricks/07_live_bronze_storage.py"),
            Path("databricks/10_gold_analytics.py"),
            Path("sql/fraud_monitoring_dashboard.sql"),
            Path("reports/phase17_summary.json"),
            Path("docs/ARCHITECTURE.md"),
            Path("README.md"),
        ]
    )
    missing_topics = sorted(topic for topic in EXPECTED_TOPICS if topic not in combined)
    assert_true(not missing_topics, f"missing Event Hub topic references: {missing_topics}")
    assert_true("stream-processing" in combined, "missing stream-processing consumer group reference")
    assert_true("bronze-storage" in combined, "missing bronze-storage consumer group reference")
    missing_tables = sorted(table for table in EXPECTED_GOLD_TABLES if table not in combined)
    assert_true(not missing_tables, f"missing Gold table references: {missing_tables}")


def iter_scanned_files():
    excluded_dirs = {".git", ".tmp", ".venv", "venv", "__pycache__", "data"}
    excluded_suffixes = {".zip", ".parquet", ".pyc", ".pyo"}
    for path in ROOT.rglob("*"):
        if not path.is_file():
            continue
        parts = set(path.relative_to(ROOT).parts)
        if parts & excluded_dirs:
            continue
        if path.name == ".env" or path.suffix.lower() in excluded_suffixes:
            continue
        yield path


SECRET_PATTERNS = [
    re.compile(r"SharedAccessKey\s*=\s*(?!<|\{|YOUR|REDACTED)[^;\\s\"']{16,}", re.IGNORECASE),
    re.compile(r"AccountKey\s*=\s*(?!<|\{|YOUR|REDACTED)[^;\\s\"']{16,}", re.IGNORECASE),
    re.compile(r"DefaultEndpointsProtocol\s*=.*AccountKey\s*=\s*(?!<|\{|YOUR|REDACTED)", re.IGNORECASE),
    re.compile(r"databricks[_-]?token\s*[:=]\s*(?!<|\{|YOUR|REDACTED)[A-Za-z0-9._/-]{20,}", re.IGNORECASE),
    re.compile(r"(password|api[_-]?key|client[_-]?secret)\s*[:=]\s*(?!<|\{|YOUR|REDACTED|placeholder)[A-Za-z0-9._/+=-]{20,}", re.IGNORECASE),
]


def validate_secret_safety() -> None:
    gitignore = read_text(ROOT / ".gitignore")
    assert_true(".env" in gitignore, ".env is not ignored")
    findings: list[str] = []
    for path in iter_scanned_files():
        try:
            text = read_text(path)
        except UnicodeDecodeError:
            continue
        for pattern in SECRET_PATTERNS:
            if pattern.search(text):
                findings.append(rel(path))
                break
    assert_true(not findings, f"possible secrets found in repository files: {sorted(set(findings))}")


def validate_phase_reports() -> None:
    phase17 = load_json(ROOT / "reports" / "phase17_summary.json")
    assert_true(phase17.get("overall_status") == "PASS", "Phase 17 report is not PASS")
    assert_true(phase17.get("gold_tables_queryable") is True, "Gold tables were not queryable in Phase 17")
    assert_true(phase17.get("source_reconciliation") == "PASS", "Phase 17 source reconciliation failed")
    assert_true(phase17.get("candidate_alias_changed") is False, "Phase 17 changed candidate alias")
    assert_true(phase17.get("threshold_changed") is False, "Phase 17 changed threshold")


def main() -> int:
    python_files = validate_python_syntax()
    json_files = validate_json_files()
    validate_feature_contract()
    validate_threshold_and_alias_safety()
    validate_eventhub_and_table_refs()
    validate_secret_safety()
    validate_phase_reports()
    print(
        json.dumps(
            {
                "status": "PASS",
                "python_files_checked": python_files,
                "json_files_checked": json_files,
                "feature_count": EXPECTED_FEATURE_COUNT,
                "threshold": 0.80,
                "gold_tables_verified_from_phase17_report": True,
                "secrets_found": False,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
