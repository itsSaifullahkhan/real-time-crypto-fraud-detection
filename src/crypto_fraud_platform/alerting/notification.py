from __future__ import annotations

from typing import Any, Mapping

from crypto_fraud_platform.alerting.alerts import format_alert_message


class WebhookNotifier:
    def __init__(self, webhook_url: str, *, timeout_seconds: float = 10.0) -> None:
        cleaned = str(webhook_url or "").strip()
        if not cleaned:
            raise ValueError("webhook_url is required")
        self.webhook_url = cleaned
        self.timeout_seconds = float(timeout_seconds)

    def send(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        import requests

        message = format_alert_message(payload)
        response = requests.post(
            self.webhook_url,
            json={
                "title": payload.get("title"),
                "text": message,
                "payload": dict(payload),
            },
            timeout=self.timeout_seconds,
        )
        response.raise_for_status()
        return {
            "status_code": int(response.status_code),
            "ok": bool(response.ok),
        }
