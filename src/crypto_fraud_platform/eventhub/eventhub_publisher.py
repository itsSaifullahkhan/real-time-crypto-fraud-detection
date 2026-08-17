from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from azure.eventhub import EventData, EventHubProducerClient


def _load_local_env() -> None:
    try:
        from dotenv import load_dotenv
    except ImportError:
        return

    project_root = Path(__file__).resolve().parents[3]
    load_dotenv(project_root / ".env", override=False)


class EventHubPublisher:
    def __init__(self) -> None:
        _load_local_env()
        connection_string = os.getenv("EVENT_HUB_CONNECTION_STRING") or os.getenv(
            "AZURE_EVENT_HUB_CONNECTION_STRING"
        )
        if not connection_string:
            raise RuntimeError("EVENT_HUB_CONNECTION_STRING is not configured")
        self._connection_string = connection_string
        self._clients: dict[str, EventHubProducerClient] = {}

    def publish(self, eventhub_name: str, event: dict[str, Any]) -> None:
        payload = json.dumps(event, default=self._json_default, separators=(",", ":"))
        batch = self._client(eventhub_name).create_batch()
        batch.add(EventData(payload))
        self._client(eventhub_name).send_batch(batch)

    def close(self) -> None:
        for client in self._clients.values():
            client.close()
        self._clients.clear()

    def _client(self, eventhub_name: str) -> EventHubProducerClient:
        client = self._clients.get(eventhub_name)
        if client is None:
            client = EventHubProducerClient.from_connection_string(
                conn_str=self._connection_string,
                eventhub_name=eventhub_name,
            )
            self._clients[eventhub_name] = client
        return client

    @staticmethod
    def _json_default(value: Any) -> str:
        if hasattr(value, "isoformat"):
            return value.isoformat()
        raise TypeError(f"Unsupported JSON value: {value!r}")
