from __future__ import annotations

import argparse
import json
import os
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import websocket

SRC_PATH = Path(__file__).resolve().parents[2]
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from crypto_fraud_platform.eventhub.eventhub_publisher import EventHubPublisher


WEBSOCKET_ENDPOINT = "wss://advanced-trade-ws.coinbase.com"
PRODUCT_IDS = ["BTC-USD", "ETH-USD"]
CHANNELS = ["market_trades", "heartbeats"]
SCHEMA_VERSION = "1.0"
SOURCE = "coinbase_websocket"
EVENT_TYPE = "market_event"
RECONNECT_DELAY_SECONDS = 5


def utc_now_z() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds").replace("+00:00", "Z")


def normalize_timestamp(value: Any) -> str | None:
    if not isinstance(value, str) or not value:
        return None
    text = value.strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    if "." in text:
        prefix, suffix = text.split(".", 1)
        offset = "+00:00" if suffix.endswith("+00:00") else ""
        fraction = suffix[: -len(offset)] if offset else suffix
        if "+" in fraction or "-" in fraction:
            sign = "+" if "+" in fraction else "-"
            fraction, timezone_part = fraction.split(sign, 1)
            offset = sign + timezone_part
        text = f"{prefix}.{fraction[:6].ljust(6, '0')}{offset}"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc).isoformat(timespec="microseconds").replace("+00:00", "Z")


def positive_float(value: Any) -> float | None:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if numeric <= 0:
        return None
    return numeric


def sequence_number(message: dict[str, Any], trade: dict[str, Any]) -> int:
    for value in (message.get("sequence_num"), message.get("sequence"), trade.get("sequence")):
        try:
            return max(int(value), 0)
        except (TypeError, ValueError):
            continue
    return 0


def market_trade_payloads(message: dict[str, Any]):
    if message.get("channel") == "market_trades":
        for event in message.get("events", []):
            for trade in event.get("trades", []):
                if isinstance(trade, dict):
                    yield trade
    elif message.get("type") == "market_trades":
        yield message


def market_trade_to_market_event(
    message: dict[str, Any],
    trade: dict[str, Any],
) -> dict[str, Any] | None:
    product_id = trade.get("product_id")
    if product_id not in PRODUCT_IDS:
        return None

    price = positive_float(trade.get("price"))
    size = positive_float(trade.get("size"))
    side = trade.get("side")
    trade_id = trade.get("trade_id")
    trade_timestamp = normalize_timestamp(trade.get("time"))
    message_timestamp = normalize_timestamp(message.get("timestamp") or trade.get("time"))

    if not all([price, size, side, trade_id, message_timestamp, trade_timestamp]):
        return None

    side_value = str(side).upper()
    if side_value not in {"BUY", "SELL"}:
        return None

    ingestion_timestamp = utc_now_z()
    return {
        "event_id": str(uuid.uuid4()),
        "event_type": EVENT_TYPE,
        "schema_version": SCHEMA_VERSION,
        "source": SOURCE,
        "event_timestamp": trade_timestamp,
        "source_timestamp": message_timestamp,
        "ingestion_timestamp": ingestion_timestamp,
        "product_id": product_id,
        "trade_id": str(trade_id),
        "price_usd": price,
        "size": size,
        "side": side_value,
        "trade_timestamp": trade_timestamp,
        "message_timestamp": message_timestamp,
        "sequence_number": sequence_number(message, trade),
    }


def schema_validator():
    configured_root = os.getenv("CRYPTO_FRAUD_PROJECT_ROOT")
    project_root = Path(configured_root).expanduser() if configured_root else Path(__file__).resolve().parents[3]
    src_path = str(project_root / "src")
    if src_path not in sys.path:
        sys.path.insert(0, src_path)
    from crypto_fraud_platform.data_generator.common.schema_validation import SchemaValidator

    return SchemaValidator(project_root)


def send_subscription(ws: websocket.WebSocket) -> None:
    ws.send(json.dumps({"type": "subscribe", "channel": "market_trades", "product_ids": PRODUCT_IDS}))
    ws.send(json.dumps({"type": "subscribe", "channel": "heartbeats"}))


def log(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


def run(args: argparse.Namespace) -> int:
    validator = schema_validator() if args.validate_schema else None
    try:
        publisher = EventHubPublisher()
    except RuntimeError as exc:
        log(str(exc))
        return 1
    eventhub_name = os.getenv("EVENT_HUB_MARKET", "market-events")
    normalized_count = 0
    normalized_products: set[str] = set()
    raw_market_trade_products: set[str] = set()
    deadline = time.monotonic() + args.max_seconds if args.max_seconds else None

    try:
        while True:
            if deadline and time.monotonic() >= deadline:
                break
            try:
                ws = websocket.create_connection(WEBSOCKET_ENDPOINT, timeout=args.read_timeout)
                send_subscription(ws)
                log("connected subscribed")
                while True:
                    if deadline and time.monotonic() >= deadline:
                        ws.close()
                        raise TimeoutError("max_seconds reached")
                    raw_message = ws.recv()
                    message = json.loads(raw_message)

                    if message.get("channel") == "heartbeats" or message.get("type") == "heartbeat":
                        continue

                    for trade in market_trade_payloads(message):
                        product_id = trade.get("product_id")
                        if product_id in PRODUCT_IDS:
                            raw_market_trade_products.add(str(product_id))
                        market_event = market_trade_to_market_event(message, trade)
                        if market_event is None:
                            continue
                        if validator is not None:
                            validator.validate_record("market-event", market_event)
                        try:
                            publisher.publish(eventhub_name, market_event)
                        except Exception as exc:
                            log(f"event hub send failed for {eventhub_name}: {type(exc).__name__}")
                            continue
                        print(
                            json.dumps(
                                {
                                    "sent": eventhub_name,
                                    "product_id": market_event["product_id"],
                                    "trade_id": market_event["trade_id"],
                                },
                                separators=(",", ":"),
                            ),
                            flush=True,
                        )
                        normalized_count += 1
                        normalized_products.add(market_event["product_id"])
                        if args.require_products and set(PRODUCT_IDS).issubset(normalized_products):
                            ws.close()
                            return 0
                        if args.max_events and normalized_count >= args.max_events and not args.require_products:
                            ws.close()
                            return 0
            except KeyboardInterrupt:
                log("shutdown requested")
                return 0
            except TimeoutError:
                break
            except Exception as exc:
                log(f"websocket disconnected: {exc}")
                if deadline and time.monotonic() >= deadline:
                    break
                time.sleep(min(RECONNECT_DELAY_SECONDS, max(args.reconnect_delay, 0)))
    finally:
        publisher.close()

    if args.require_products and not set(PRODUCT_IDS).issubset(normalized_products):
        missing = sorted(set(PRODUCT_IDS) - normalized_products)
        raw_seen = sorted(raw_market_trade_products)
        log(f"missing normalized products: {missing}; raw market trade products seen: {raw_seen}")
        return 1
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Coinbase public market WebSocket collector.")
    parser.add_argument("--max-events", type=int, default=0)
    parser.add_argument("--max-seconds", type=int, default=0)
    parser.add_argument("--read-timeout", type=int, default=20)
    parser.add_argument("--reconnect-delay", type=int, default=RECONNECT_DELAY_SECONDS)
    parser.add_argument("--require-products", action="store_true")
    parser.add_argument("--validate-schema", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(run(parse_args()))
