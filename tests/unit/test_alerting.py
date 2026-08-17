from crypto_fraud_platform.alerting.alerts import (
    ALERT_THRESHOLD,
    build_alert_payload,
    build_test_alert_payload,
    format_alert_message,
    is_high_risk_fraud_decision,
)


def test_high_risk_alert_requires_model_decision_flag_and_threshold() -> None:
    event = {
        "event_type": "fraud_decision",
        "transaction_id": "tx-1",
        "fraud_probability": ALERT_THRESHOLD,
        "predicted_fraud": True,
    }

    assert is_high_risk_fraud_decision(event)


def test_high_risk_alert_does_not_fire_for_labels() -> None:
    event = {
        "event_type": "fraud_label",
        "transaction_id": "tx-1",
        "is_fraud": True,
        "fraud_probability": 0.99,
        "predicted_fraud": True,
    }

    assert not is_high_risk_fraud_decision(event)


def test_high_risk_alert_does_not_infer_missing_predicted_flag() -> None:
    event = {
        "event_type": "fraud_decision",
        "transaction_id": "tx-1",
        "risk_score": 0.99,
        "decision": "REVIEW",
    }

    assert not is_high_risk_fraud_decision(event)


def test_alert_payload_uses_non_secret_fields() -> None:
    payload = build_alert_payload(
        {
            "event_type": "fraud_decision",
            "transaction_id": "tx-1",
            "fraud_probability": 0.91,
            "predicted_fraud": True,
            "transaction_amount_usd": 1200.0,
            "asset": "BTC",
            "country": "US",
            "prediction_timestamp": "2026-08-12T12:00:00Z",
            "model_name": "crypto_fraud.models.fraud_detection_model",
            "model_version": "1",
        }
    )

    text = format_alert_message(payload)

    assert payload["transaction_id"] == "tx-1"
    assert payload["amount"] == 1200.0
    assert "Transaction: tx-1" in text
    assert "SharedAccessKey" not in text


def test_test_alert_is_clearly_labeled_and_not_a_model_prediction() -> None:
    payload = build_test_alert_payload()

    assert payload["alert_type"] == "TEST_ALERT"
    assert payload["transaction_id"] == "TEST-ONLY-NOT-A-MODEL-PREDICTION"
    assert payload["source_event_id"] == "test-alert-not-written-to-event-hubs"
