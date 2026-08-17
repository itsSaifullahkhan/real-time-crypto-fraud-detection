from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np

SRC_PATH = Path(__file__).resolve().parents[2]
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from crypto_fraud_platform.data_generator.common.identifiers import DeterministicIdFactory
from crypto_fraud_platform.data_generator.common.schema_validation import SchemaValidator
from crypto_fraud_platform.data_generator.fraud_scenarios.scenario_engine import ScenarioEngine
from crypto_fraud_platform.data_generator.historical.authentication_generator import (
    _different_country,
    _ip_address,
    make_authentication_event,
)
from crypto_fraud_platform.data_generator.historical.entity_generator import generate_entities
from crypto_fraud_platform.data_generator.historical.label_generator import make_fraud_label
from crypto_fraud_platform.eventhub.eventhub_publisher import EventHubPublisher


TRANSACTION_INTERVAL_SECONDS = float(os.getenv("TRANSACTION_INTERVAL_SECONDS", "1.0"))
DEFAULT_FRAUD_RATE = float(os.getenv("DEFAULT_FRAUD_RATE", "0.10"))
LABEL_DELAY_SECONDS = float(os.getenv("LABEL_DELAY_SECONDS", "10"))
RANDOM_SEED = int(os.getenv("RANDOM_SEED", "42"))

SOURCE = "realtime_customer_generator"
SCHEMA_VERSION = "1.0"
MARKET_PRICES_USD = {"BTC": 65000.0, "ETH": 1900.0}
MIN_ACCOUNT_COUNT = 64


@dataclass
class PendingLabel:
    emit_at: datetime
    transaction_id: str
    is_fraud: bool
    fraud_type: str | None


def utc_now() -> datetime:
    return datetime.now(tz=UTC).replace(microsecond=0)


def json_default(value: Any) -> str:
    if isinstance(value, datetime):
        return value.astimezone(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")
    raise TypeError(f"Unsupported JSON value: {value!r}")


def emit(prefix: str, record: dict[str, Any]) -> None:
    print(f"{prefix}: {json.dumps(record, default=json_default, separators=(',', ':'))}", flush=True)


def project_root() -> Path:
    configured_root = os.getenv("CRYPTO_FRAUD_PROJECT_ROOT")
    if configured_root:
        return Path(configured_root).expanduser()
    return Path(__file__).resolve().parents[3]


class LiveCustomerGenerator:
    def __init__(
        self,
        *,
        fraud_rate: float = DEFAULT_FRAUD_RATE,
        label_delay_seconds: float = LABEL_DELAY_SECONDS,
        random_seed: int = RANDOM_SEED,
        validate_schema: bool = False,
        force_fraud_count: int = 0,
    ) -> None:
        self.rng = np.random.default_rng(random_seed)
        self.id_factory = DeterministicIdFactory(random_seed + 1100)
        self.fraud_rate = fraud_rate
        self.label_delay_seconds = label_delay_seconds
        self.force_fraud_count = force_fraud_count
        root = project_root()
        self.validator = SchemaValidator(root) if validate_schema else None
        self.scenario_engine = ScenarioEngine(project_root=root, rng=self.rng)
        now = utc_now()
        self.entities = generate_entities(
            account_count=MIN_ACCOUNT_COUNT,
            start_timestamp=now - timedelta(days=30),
            end_timestamp=now,
            rng=self.rng,
            id_factory=self.id_factory,
        )
        self.active_accounts = self.entities.accounts[
            self.entities.accounts["account_status"] == "ACTIVE"
        ].to_dict(orient="records")
        self.pending_labels: list[PendingLabel] = []
        self.transaction_index = 0
        self.stats = {
            "transactions": 0,
            "fraud_transactions": 0,
            "auth_events": 0,
            "labels": 0,
            "transaction_schema_pass": True,
            "authentication_schema_pass": True,
            "fraud_label_schema_pass": True,
        }

    def next_transaction_bundle(self) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        account = self._account()
        is_fraud = self._is_fraud()
        scenario_id = self._scenario_id() if is_fraud else None
        auth_events = self._authentication_events(account, scenario_id)
        transaction = self._transaction(account, scenario_id)
        fraud_type = (
            self.scenario_engine.scenarios[scenario_id].fraud_type if scenario_id is not None else None
        )
        self.pending_labels.append(
            PendingLabel(
                emit_at=utc_now() + timedelta(seconds=self.label_delay_seconds),
                transaction_id=transaction["transaction_id"],
                is_fraud=is_fraud,
                fraud_type=fraud_type,
            )
        )
        self.transaction_index += 1
        return auth_events, transaction

    def emit_due_labels(self) -> list[dict[str, Any]]:
        now = utc_now()
        ready = [item for item in self.pending_labels if item.emit_at <= now]
        self.pending_labels = [item for item in self.pending_labels if item.emit_at > now]
        labels = []
        for item in ready:
            label = make_fraud_label(
                id_factory=self.id_factory,
                transaction_id=item.transaction_id,
                label_timestamp=now,
                is_fraud=item.is_fraud,
                fraud_type=item.fraud_type,
                investigation_status="CONFIRMED_FRAUD" if item.is_fraud else "CLEARED",
            )
            self._validate("fraud-label", label)
            labels.append(label)
            self.stats["labels"] += 1
        return labels

    def _account(self) -> dict[str, Any]:
        return self.active_accounts[int(self.rng.integers(0, len(self.active_accounts)))]

    def _is_fraud(self) -> bool:
        if self.force_fraud_count and self.stats["fraud_transactions"] < self.force_fraud_count:
            return True
        return bool(self.rng.random() < self.fraud_rate)

    def _scenario_id(self) -> str:
        scenarios = self.scenario_engine.enabled_scenario_ids
        return str(scenarios[self.stats["fraud_transactions"] % len(scenarios)])

    def _authentication_events(
        self,
        account: dict[str, Any],
        scenario_id: str | None,
    ) -> list[dict[str, Any]]:
        account_id = str(account["account_id"])
        home_country = str(account["home_country"])
        event_time = utc_now()
        events: list[dict[str, Any]] = []

        if scenario_id in {"account_takeover", "shared_suspicious_device"}:
            device_id = str(self.rng.choice(self.entities.shared_device_ids))
            country = _different_country(home_country, self.rng)
            for offset in range(2):
                events.append(
                    make_authentication_event(
                        id_factory=self.id_factory,
                        event_timestamp=event_time - timedelta(seconds=3 - offset),
                        account_id=account_id,
                        device_id=device_id,
                        country=country,
                        ip_address=_ip_address(self.rng),
                        login_success=False,
                        mfa_success=False,
                        password_reset_flag=offset == 1 and scenario_id == "account_takeover",
                        failure_reason="MFA_FAILED",
                    )
                )
            events.append(
                make_authentication_event(
                    id_factory=self.id_factory,
                    event_timestamp=event_time,
                    account_id=account_id,
                    device_id=device_id,
                    country=country,
                    ip_address=_ip_address(self.rng),
                    login_success=True,
                    mfa_success=True,
                    password_reset_flag=False,
                    failure_reason=None,
                )
            )
            return self._validated_auths(events)

        device_id = str(self.rng.choice(self.entities.account_device_ids[account_id]))
        success = bool(self.rng.choice([True, False], p=[0.94, 0.06]))
        events.append(
            make_authentication_event(
                id_factory=self.id_factory,
                event_timestamp=event_time,
                account_id=account_id,
                device_id=device_id,
                country=home_country,
                ip_address=_ip_address(self.rng),
                login_success=success,
                mfa_success=success,
                password_reset_flag=False,
                failure_reason=None if success else "INVALID_PASSWORD",
            )
        )
        return self._validated_auths(events)

    def _validated_auths(self, events: list[dict[str, Any]]) -> list[dict[str, Any]]:
        for event in events:
            self._validate("authentication-event", event)
            self.stats["auth_events"] += 1
        return events

    def _transaction(self, account: dict[str, Any], scenario_id: str | None) -> dict[str, Any]:
        account_id = str(account["account_id"])
        home_country = str(account["home_country"])
        asset = str(self.rng.choice(list(account["preferred_assets"])))
        transaction_type = str(self.rng.choice(["DEPOSIT", "WITHDRAWAL", "TRANSFER"], p=[0.34, 0.46, 0.20]))
        amount_usd = self._normal_amount(account)
        device_id = str(self.rng.choice(self.entities.account_device_ids[account_id]))
        country = home_country

        if scenario_id == "account_takeover":
            transaction_type = "WITHDRAWAL"
            amount_usd *= float(self.rng.uniform(4.0, 10.0))
            device_id = str(self.rng.choice(self.entities.shared_device_ids))
            country = _different_country(home_country, self.rng)
        elif scenario_id == "high_transaction_velocity":
            transaction_type = "WITHDRAWAL"
            amount_usd *= float(self.rng.uniform(1.1, 1.8))
        elif scenario_id == "unusual_transaction_amount":
            transaction_type = str(self.rng.choice(["WITHDRAWAL", "TRANSFER"]))
            amount_usd *= float(self.rng.uniform(5.0, 14.0))
        elif scenario_id == "structuring":
            transaction_type = "WITHDRAWAL"
            amount_usd = min(9500.0, max(7500.0, amount_usd * 1.5))
        elif scenario_id == "mule_account_activity":
            transaction_type = "WITHDRAWAL"
            amount_usd *= float(self.rng.uniform(1.3, 3.2))
        elif scenario_id == "shared_suspicious_device":
            transaction_type = "WITHDRAWAL"
            amount_usd *= float(self.rng.uniform(2.0, 5.0))
            device_id = str(self.rng.choice(self.entities.shared_device_ids))
            country = _different_country(home_country, self.rng)

        source_wallet_id, destination_wallet_id = self._wallets(account_id, asset, transaction_type)
        price = self._market_price(asset)
        quantity = round(max(amount_usd / price, 0.00000001), 8)
        event_time = utc_now()
        source_time = event_time + timedelta(seconds=1)
        record = {
            "event_id": self.id_factory.next("customer_transaction_event"),
            "event_type": "customer_transaction",
            "schema_version": SCHEMA_VERSION,
            "source": SOURCE,
            "event_timestamp": event_time,
            "source_timestamp": source_time,
            "ingestion_timestamp": source_time + timedelta(seconds=1),
            "transaction_id": self.id_factory.next("transaction"),
            "account_id": account_id,
            "asset": asset,
            "crypto_quantity": quantity,
            "transaction_type": transaction_type,
            "source_wallet_id": source_wallet_id,
            "destination_wallet_id": destination_wallet_id,
            "device_id": device_id,
            "country": country,
            "market_price_usd": price,
            "transaction_amount_usd": round(quantity * price, 8),
            "transaction_status": "COMPLETED",
        }
        self._validate("customer-transaction", record)
        self.stats["transactions"] += 1
        if scenario_id is not None:
            self.stats["fraud_transactions"] += 1
        return record

    def _wallets(
        self,
        account_id: str,
        asset: str,
        transaction_type: str,
    ) -> tuple[str | None, str | None]:
        customer_wallet = str(self.entities.account_customer_wallet_ids[account_id][asset][0])
        external = self._external_wallet(asset)
        if transaction_type == "DEPOSIT":
            return None, customer_wallet
        return customer_wallet, external

    def _external_wallet(self, asset: str) -> str:
        wallets = self.entities.wallets[
            (self.entities.wallets["wallet_type"] == "EXTERNAL")
            & (self.entities.wallets["supported_assets"].apply(lambda values: asset in values))
        ]
        return str(self.rng.choice(wallets["wallet_id"].to_numpy()))

    def _normal_amount(self, account: dict[str, Any]) -> float:
        normal = float(account["normal_transaction_amount_usd"])
        return round(max(10.0, normal * float(self.rng.lognormal(mean=0.0, sigma=0.38))), 2)

    def _market_price(self, asset: str) -> float:
        base = MARKET_PRICES_USD[asset]
        return round(base * float(self.rng.normal(1.0, 0.002)), 2)

    def _validate(self, schema_name: str, record: dict[str, Any]) -> None:
        if self.validator is not None:
            self.validator.validate_record(schema_name, record)


def run(args: argparse.Namespace) -> int:
    generator = LiveCustomerGenerator(
        fraud_rate=args.fraud_rate,
        label_delay_seconds=args.label_delay,
        random_seed=args.random_seed,
        validate_schema=args.validate_schema,
        force_fraud_count=args.force_fraud_count,
    )
    try:
        publisher = EventHubPublisher()
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    transaction_hub = os.getenv("EVENT_HUB_TRANSACTIONS", "transaction-events")
    authentication_hub = os.getenv("EVENT_HUB_AUTHENTICATION", "authentication-events")
    fraud_label_hub = os.getenv("EVENT_HUB_FRAUD_LABELS", "fraud-labels")
    target_count = args.transaction_count
    try:
        while target_count == 0 or generator.stats["transactions"] < target_count:
            auths, transaction = generator.next_transaction_bundle()
            for event in auths:
                try:
                    publisher.publish(authentication_hub, event)
                except Exception as exc:
                    print(
                        f"event hub send failed for {authentication_hub}: {type(exc).__name__}",
                        file=sys.stderr,
                    )
                    continue
                emit("AUTH", event)
            try:
                publisher.publish(transaction_hub, transaction)
            except Exception as exc:
                print(
                    f"event hub send failed for {transaction_hub}: {type(exc).__name__}",
                    file=sys.stderr,
                )
            else:
                emit("TRANSACTION", transaction)
            for label in generator.emit_due_labels():
                try:
                    publisher.publish(fraud_label_hub, label)
                except Exception as exc:
                    print(
                        f"event hub send failed for {fraud_label_hub}: {type(exc).__name__}",
                        file=sys.stderr,
                    )
                    continue
                emit("FRAUD_LABEL", label)
            if args.interval > 0:
                time.sleep(args.interval)

        deadline = time.monotonic() + max(args.label_delay + 2.0, 2.0)
        while generator.pending_labels and time.monotonic() < deadline:
            for label in generator.emit_due_labels():
                try:
                    publisher.publish(fraud_label_hub, label)
                except Exception as exc:
                    print(
                        f"event hub send failed for {fraud_label_hub}: {type(exc).__name__}",
                        file=sys.stderr,
                    )
                    continue
                emit("FRAUD_LABEL", label)
            if generator.pending_labels:
                time.sleep(min(0.25, max(args.label_delay, 0.01)))

        print("SUMMARY: " + json.dumps(generator.stats, separators=(",", ":")), file=sys.stderr)
        return 0 if not generator.pending_labels else 1
    except KeyboardInterrupt:
        print("shutdown requested", file=sys.stderr)
        return 0
    finally:
        publisher.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Live synthetic customer-event generator.")
    parser.add_argument("--transaction-count", type=int, default=0)
    parser.add_argument("--interval", type=float, default=TRANSACTION_INTERVAL_SECONDS)
    parser.add_argument("--fraud-rate", type=float, default=DEFAULT_FRAUD_RATE)
    parser.add_argument("--label-delay", type=float, default=LABEL_DELAY_SECONDS)
    parser.add_argument("--random-seed", type=int, default=RANDOM_SEED)
    parser.add_argument("--force-fraud-count", type=int, default=0)
    parser.add_argument("--validate-schema", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(run(parse_args()))
