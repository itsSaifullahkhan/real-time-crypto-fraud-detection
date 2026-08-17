from __future__ import annotations

from datetime import UTC, datetime
from typing import Any, Mapping

ALERT_THRESHOLD = 0.80


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"true", "1", "yes"}
    return bool(value)


def _as_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _first_present(event: Mapping[str, Any], *keys: str) -> Any:
    for key in keys:
        value = event.get(key)
        if value is not None:
            return value
    return None


def is_high_risk_fraud_decision(
    event: Mapping[str, Any],
    *,
    threshold: float = ALERT_THRESHOLD,
) -> bool:
    if event.get("event_type") != "fraud_decision":
        return False
    if not _as_bool(event.get("predicted_fraud")):
        return False
    probability = _as_float(_first_present(event, "fraud_probability", "risk_score"))
    return probability is not None and probability >= float(threshold)


def build_alert_payload(
    event: Mapping[str, Any],
    *,
    threshold: float = ALERT_THRESHOLD,
    title: str = "High-Risk Crypto Transaction Detected",
) -> dict[str, Any]:
    probability = _as_float(_first_present(event, "fraud_probability", "risk_score"))
    return {
        "title": title,
        "alert_type": "REAL_FRAUD_DECISION",
        "transaction_id": event.get("transaction_id"),
        "fraud_probability": probability,
        "predicted_fraud": _as_bool(event.get("predicted_fraud")),
        "threshold": float(threshold),
        "amount": _first_present(event, "transaction_amount_usd", "amount"),
        "asset": event.get("asset"),
        "country": event.get("country"),
        "decision_timestamp": _first_present(
            event,
            "prediction_timestamp",
            "event_timestamp",
            "decision_timestamp",
        ),
        "model_name": event.get("model_name"),
        "model_version": event.get("model_version"),
        "threshold_policy_version": event.get("threshold_policy_version"),
        "source_event_id": event.get("event_id"),
    }


def build_test_alert_payload() -> dict[str, Any]:
    now = datetime.now(tz=UTC).isoformat(timespec="seconds").replace("+00:00", "Z")
    return {
        "title": "TEST FRAUD ALERT",
        "alert_type": "TEST_ALERT",
        "transaction_id": "TEST-ONLY-NOT-A-MODEL-PREDICTION",
        "fraud_probability": 0.80,
        "predicted_fraud": True,
        "threshold": ALERT_THRESHOLD,
        "amount": 0.0,
        "asset": "TEST",
        "country": "TEST",
        "decision_timestamp": now,
        "model_name": "notification-channel-test",
        "model_version": "not-applicable",
        "threshold_policy_version": "test-alert-not-model-output",
        "source_event_id": "test-alert-not-written-to-event-hubs",
    }


def format_alert_message(payload: Mapping[str, Any]) -> str:
    amount = payload.get("amount")
    amount_text = "not available" if amount in {None, ""} else str(amount)
    probability = payload.get("fraud_probability")
    probability_text = "not available" if probability is None else f"{float(probability):.4f}"
    return "\n".join(
        [
            str(payload.get("title") or "Fraud Alert"),
            "",
            f"Transaction: {payload.get('transaction_id')}",
            f"Fraud probability: {probability_text}",
            f"Amount: {amount_text}",
            f"Asset: {payload.get('asset') or 'not available'}",
            f"Country: {payload.get('country') or 'not available'}",
            f"Decision time: {payload.get('decision_timestamp') or 'not available'}",
            f"Model: {payload.get('model_name') or 'not available'}",
            f"Model version: {payload.get('model_version') or 'not available'}",
            f"Alert type: {payload.get('alert_type') or 'not available'}",
        ]
    )
