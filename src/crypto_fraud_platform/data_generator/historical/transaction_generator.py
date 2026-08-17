from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any

import numpy as np
import pandas as pd

from crypto_fraud_platform.data_generator.common.config import HistoricalGeneratorConfig
from crypto_fraud_platform.data_generator.common.identifiers import DeterministicIdFactory
from crypto_fraud_platform.data_generator.common.time_utils import parse_utc, random_datetimes
from crypto_fraud_platform.data_generator.fraud_scenarios.scenario_engine import ScenarioEngine
from crypto_fraud_platform.data_generator.historical.authentication_generator import (
    _different_country,
    _ip_address,
    make_authentication_event,
)
from crypto_fraud_platform.data_generator.historical.entity_generator import GeneratedEntities


@dataclass(frozen=True)
class MarketMatch:
    product_id: str
    candle_start_timestamp: datetime
    candle_end_timestamp: datetime
    market_price_usd: float
    freshness_seconds: float


@dataclass(frozen=True)
class ScenarioGenerationResult:
    authentication_events: list[dict[str, Any]]
    transactions: list[dict[str, Any]]
    audit_rows: list[dict[str, Any]]
    fraud_rows: list[dict[str, Any]]
    market_matches: list[dict[str, Any]]


class MarketLookupError(RuntimeError):
    pass


class MarketPriceLookup:
    def __init__(self, candles: pd.DataFrame, asset_product_mapping: dict[str, str]) -> None:
        if candles.empty:
            raise MarketLookupError("Market candle data is empty")
        self.asset_product_mapping = asset_product_mapping
        self.candles = candles.copy()
        for column in ["candle_start_timestamp", "candle_end_timestamp"]:
            self.candles[column] = pd.to_datetime(self.candles[column], utc=True)
        self.candles = self.candles.sort_values(["product_id", "candle_end_timestamp"]).reset_index(drop=True)

    def lookup(self, asset: str, event_timestamp: datetime) -> MarketMatch:
        product_id = self.asset_product_mapping[asset]
        event_time = pd.Timestamp(parse_utc(event_timestamp))
        product_candles = self.candles[self.candles["product_id"] == product_id]
        available = product_candles[product_candles["candle_end_timestamp"] <= event_time]
        if available.empty:
            raise MarketLookupError(
                f"No completed {product_id} candle is available at {event_time.isoformat()}"
            )
        row = available.iloc[-1]
        candle_end = parse_utc(row["candle_end_timestamp"])
        return MarketMatch(
            product_id=product_id,
            candle_start_timestamp=parse_utc(row["candle_start_timestamp"]),
            candle_end_timestamp=candle_end,
            market_price_usd=round(float(row["close_price_usd"]), 8),
            freshness_seconds=(parse_utc(event_timestamp) - candle_end).total_seconds(),
        )


def make_customer_transaction(
    *,
    id_factory: DeterministicIdFactory,
    market_lookup: MarketPriceLookup,
    event_timestamp: datetime,
    account_id: str,
    asset: str,
    desired_amount_usd: float,
    transaction_type: str,
    source_wallet_id: str | None,
    destination_wallet_id: str | None,
    device_id: str,
    country: str,
    transaction_status: str = "COMPLETED",
) -> tuple[dict[str, Any], dict[str, Any]]:
    event_time = parse_utc(event_timestamp)
    market_match = market_lookup.lookup(asset, event_time)
    crypto_quantity = round(max(float(desired_amount_usd), 0.01) / market_match.market_price_usd, 8)
    transaction_amount_usd = round(crypto_quantity * market_match.market_price_usd, 8)
    source_time = event_time + timedelta(seconds=1)
    transaction_id = id_factory.next("transaction")
    record = {
        "event_id": id_factory.next("customer_transaction_event"),
        "event_type": "customer_transaction",
        "schema_version": "1.0",
        "source": "historical_customer_generator",
        "event_timestamp": event_time,
        "source_timestamp": source_time,
        "ingestion_timestamp": source_time + timedelta(seconds=1),
        "transaction_id": transaction_id,
        "account_id": account_id,
        "asset": asset,
        "crypto_quantity": crypto_quantity,
        "transaction_type": transaction_type,
        "source_wallet_id": source_wallet_id,
        "destination_wallet_id": destination_wallet_id,
        "device_id": device_id,
        "country": country,
        "market_price_usd": market_match.market_price_usd,
        "transaction_amount_usd": transaction_amount_usd,
        "transaction_status": transaction_status,
    }
    match_record = {
        "transaction_id": transaction_id,
        "asset": asset,
        "product_id": market_match.product_id,
        "transaction_event_timestamp": event_time,
        "matched_market_candle_timestamp": market_match.candle_start_timestamp,
        "matched_market_candle_end_timestamp": market_match.candle_end_timestamp,
        "market_data_freshness_seconds": market_match.freshness_seconds,
        "market_price_usd": market_match.market_price_usd,
    }
    return record, match_record


def generate_scenario_activity(
    *,
    config: HistoricalGeneratorConfig,
    entities: GeneratedEntities,
    market_lookup: MarketPriceLookup,
    scenario_engine: ScenarioEngine,
    rng: np.random.Generator,
    id_factory: DeterministicIdFactory,
) -> ScenarioGenerationResult:
    target_fraud_count = max(1, round(config.scale.target_transaction_count * config.scale.target_fraud_rate))
    allocations = scenario_engine.allocate_fraud_counts(target_fraud_count)
    active_accounts = entities.accounts[entities.accounts["account_status"] == "ACTIVE"].to_dict(orient="records")
    account_cursor = 0

    auth_events: list[dict[str, Any]] = []
    transactions: list[dict[str, Any]] = []
    audit_rows: list[dict[str, Any]] = []
    fraud_rows: list[dict[str, Any]] = []
    market_matches: list[dict[str, Any]] = []

    def next_account() -> dict[str, Any]:
        nonlocal account_cursor
        account = active_accounts[account_cursor % len(active_accounts)]
        account_cursor += 1
        return account

    for scenario_id, fraud_count in allocations.items():
        if scenario_id == "account_takeover":
            for _ in range(fraud_count):
                account = next_account()
                result = _account_takeover_case(
                    config=config,
                    account=account,
                    entities=entities,
                    market_lookup=market_lookup,
                    rng=rng,
                    id_factory=id_factory,
                    scenario_engine=scenario_engine,
                )
                _extend_result(result, auth_events, transactions, audit_rows, fraud_rows, market_matches)
        elif scenario_id == "high_transaction_velocity":
            for _ in range(fraud_count):
                account = next_account()
                result = _high_velocity_case(
                    config=config,
                    account=account,
                    entities=entities,
                    market_lookup=market_lookup,
                    rng=rng,
                    id_factory=id_factory,
                    scenario_engine=scenario_engine,
                )
                _extend_result(result, auth_events, transactions, audit_rows, fraud_rows, market_matches)
        elif scenario_id == "unusual_transaction_amount":
            for _ in range(fraud_count):
                account = next_account()
                result = _unusual_amount_case(
                    config=config,
                    account=account,
                    entities=entities,
                    market_lookup=market_lookup,
                    rng=rng,
                    id_factory=id_factory,
                    scenario_engine=scenario_engine,
                )
                _extend_result(result, auth_events, transactions, audit_rows, fraud_rows, market_matches)
        elif scenario_id == "structuring":
            remaining = fraud_count
            while remaining > 0:
                sequence_count = min(remaining, int(rng.integers(4, 7)))
                account = next_account()
                result = _structuring_case(
                    config=config,
                    account=account,
                    transaction_count=sequence_count,
                    entities=entities,
                    market_lookup=market_lookup,
                    rng=rng,
                    id_factory=id_factory,
                    scenario_engine=scenario_engine,
                )
                _extend_result(result, auth_events, transactions, audit_rows, fraud_rows, market_matches)
                remaining -= sequence_count
        elif scenario_id == "mule_account_activity":
            result = _mule_activity_case(
                config=config,
                accounts=[next_account() for _ in range(fraud_count)],
                entities=entities,
                market_lookup=market_lookup,
                rng=rng,
                id_factory=id_factory,
                scenario_engine=scenario_engine,
            )
            _extend_result(result, auth_events, transactions, audit_rows, fraud_rows, market_matches)
        elif scenario_id == "shared_suspicious_device":
            for _ in range(fraud_count):
                account = next_account()
                related_accounts = [account, next_account(), next_account()]
                result = _shared_device_case(
                    config=config,
                    account=account,
                    related_accounts=related_accounts,
                    entities=entities,
                    market_lookup=market_lookup,
                    rng=rng,
                    id_factory=id_factory,
                    scenario_engine=scenario_engine,
                )
                _extend_result(result, auth_events, transactions, audit_rows, fraud_rows, market_matches)

    return ScenarioGenerationResult(auth_events, transactions, audit_rows, fraud_rows, market_matches)


def generate_normal_transactions(
    *,
    config: HistoricalGeneratorConfig,
    entities: GeneratedEntities,
    market_lookup: MarketPriceLookup,
    existing_transactions: list[dict[str, Any]],
    existing_market_matches: list[dict[str, Any]],
    rng: np.random.Generator,
    id_factory: DeterministicIdFactory,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    records = list(existing_transactions)
    matches = list(existing_market_matches)
    remaining = config.scale.target_transaction_count - len(records)
    if remaining < 0:
        raise ValueError("Scenario transactions exceed configured target transaction count")
    if remaining == 0:
        return _transactions_to_frame(records), _matches_to_frame(matches)

    active_accounts = entities.accounts[entities.accounts["account_status"] == "ACTIVE"]
    account_records = active_accounts.to_dict(orient="records")
    start = parse_utc(config.time_range.start) + timedelta(minutes=30)
    timestamps = random_datetimes(rng, remaining, start, config.time_range.end)

    for event_timestamp in timestamps:
        account = account_records[int(rng.integers(0, len(account_records)))]
        account_id = str(account["account_id"])
        asset = _asset_for_account(account, rng)
        transaction_type = str(rng.choice(["DEPOSIT", "WITHDRAWAL", "TRANSFER"], p=[0.34, 0.46, 0.20]))
        source_wallet_id, destination_wallet_id = _wallet_pair_for_transaction(
            account=account,
            asset=asset,
            transaction_type=transaction_type,
            entities=entities,
            rng=rng,
        )
        status = str(rng.choice(["COMPLETED", "PENDING", "FAILED"], p=[0.90, 0.05, 0.05]))
        if not config.generation.include_failed_transactions and status == "FAILED":
            status = "COMPLETED"
        desired_amount = _normal_amount_for_account(account, rng)
        record, match = make_customer_transaction(
            id_factory=id_factory,
            market_lookup=market_lookup,
            event_timestamp=event_timestamp,
            account_id=account_id,
            asset=asset,
            desired_amount_usd=desired_amount,
            transaction_type=transaction_type,
            source_wallet_id=source_wallet_id,
            destination_wallet_id=destination_wallet_id,
            device_id=str(rng.choice(entities.account_device_ids[account_id])),
            country=str(account["home_country"])
            if rng.random() > 0.025
            else _different_country(str(account["home_country"]), rng),
            transaction_status=status,
        )
        records.append(record)
        matches.append(match)

    return _transactions_to_frame(records), _matches_to_frame(matches)


def _account_takeover_case(
    *,
    config: HistoricalGeneratorConfig,
    account: dict[str, Any],
    entities: GeneratedEntities,
    market_lookup: MarketPriceLookup,
    rng: np.random.Generator,
    id_factory: DeterministicIdFactory,
    scenario_engine: ScenarioEngine,
) -> ScenarioGenerationResult:
    scenario_id = "account_takeover"
    event_time = _scenario_time(config, rng)
    account_id = str(account["account_id"])
    suspicious_device = str(rng.choice(entities.shared_device_ids))
    attack_country = _different_country(str(account["home_country"]), rng)
    auths: list[dict[str, Any]] = []
    failed_attempts = int(rng.integers(2, 6))
    first_failed = event_time - timedelta(minutes=failed_attempts + 4)
    for offset in range(failed_attempts):
        auths.append(
            make_authentication_event(
                id_factory=id_factory,
                event_timestamp=first_failed + timedelta(minutes=offset),
                account_id=account_id,
                device_id=suspicious_device,
                country=attack_country,
                ip_address=_ip_address(rng),
                login_success=False,
                mfa_success=False,
                password_reset_flag=False,
                failure_reason="MFA_FAILED" if offset % 2 else "INVALID_PASSWORD",
            )
        )
    if rng.random() < 0.65:
        auths.append(
            make_authentication_event(
                id_factory=id_factory,
                event_timestamp=event_time - timedelta(minutes=3),
                account_id=account_id,
                device_id=suspicious_device,
                country=attack_country,
                ip_address=_ip_address(rng),
                login_success=False,
                mfa_success=False,
                password_reset_flag=True,
                failure_reason="OTHER",
            )
        )
    auths.append(
        make_authentication_event(
            id_factory=id_factory,
            event_timestamp=event_time - timedelta(minutes=1),
            account_id=account_id,
            device_id=suspicious_device,
            country=attack_country,
            ip_address=_ip_address(rng),
            login_success=True,
            mfa_success=True,
            password_reset_flag=False,
            failure_reason=None,
        )
    )
    asset = _asset_for_account(account, rng)
    tx, match = make_customer_transaction(
        id_factory=id_factory,
        market_lookup=market_lookup,
        event_timestamp=event_time,
        account_id=account_id,
        asset=asset,
        desired_amount_usd=float(account["normal_transaction_amount_usd"]) * float(rng.uniform(4, 12)),
        transaction_type="WITHDRAWAL",
        source_wallet_id=_customer_wallet(account_id, asset, entities, rng),
        destination_wallet_id=_external_wallet(asset, entities, rng, prefer_new=True),
        device_id=suspicious_device,
        country=attack_country,
        transaction_status="COMPLETED",
    )
    return _single_fraud_result(
        scenario_engine=scenario_engine,
        scenario_id=scenario_id,
        transaction=tx,
        market_match=match,
        auth_events=auths,
        id_factory=id_factory,
    )


def _high_velocity_case(
    *,
    config: HistoricalGeneratorConfig,
    account: dict[str, Any],
    entities: GeneratedEntities,
    market_lookup: MarketPriceLookup,
    rng: np.random.Generator,
    id_factory: DeterministicIdFactory,
    scenario_engine: ScenarioEngine,
) -> ScenarioGenerationResult:
    scenario_id = "high_transaction_velocity"
    base_time = _scenario_time(config, rng)
    account_id = str(account["account_id"])
    asset = _asset_for_account(account, rng)
    device_id = str(rng.choice(entities.account_device_ids[account_id]))
    source_wallet_id = _customer_wallet(account_id, asset, entities, rng)
    destination_wallet_id = _external_wallet(asset, entities, rng)
    records: list[dict[str, Any]] = []
    matches: list[dict[str, Any]] = []
    audit_rows: list[dict[str, Any]] = []
    support_count = int(rng.integers(3, 6))
    execution_id = id_factory.next("scenario_execution")
    for index in range(support_count + 1):
        tx, match = make_customer_transaction(
            id_factory=id_factory,
            market_lookup=market_lookup,
            event_timestamp=base_time + timedelta(seconds=index * int(rng.integers(20, 55))),
            account_id=account_id,
            asset=asset,
            desired_amount_usd=_normal_amount_for_account(account, rng) * float(rng.uniform(0.7, 1.4)),
            transaction_type="WITHDRAWAL",
            source_wallet_id=source_wallet_id,
            destination_wallet_id=destination_wallet_id,
            device_id=device_id,
            country=str(account["home_country"]),
            transaction_status="COMPLETED",
        )
        is_final = index == support_count
        records.append(tx)
        matches.append(match)
        audit_rows.append(
            _audit_row(
                scenario_engine=scenario_engine,
                scenario_id=scenario_id,
                execution_id=execution_id,
                transaction=tx,
                label_expected=is_final,
            )
        )
    return ScenarioGenerationResult([], records, audit_rows, [_fraud_row(scenario_engine, scenario_id, records[-1])], matches)


def _unusual_amount_case(
    *,
    config: HistoricalGeneratorConfig,
    account: dict[str, Any],
    entities: GeneratedEntities,
    market_lookup: MarketPriceLookup,
    rng: np.random.Generator,
    id_factory: DeterministicIdFactory,
    scenario_engine: ScenarioEngine,
) -> ScenarioGenerationResult:
    scenario_id = "unusual_transaction_amount"
    account_id = str(account["account_id"])
    asset = _asset_for_account(account, rng)
    tx, match = make_customer_transaction(
        id_factory=id_factory,
        market_lookup=market_lookup,
        event_timestamp=_scenario_time(config, rng),
        account_id=account_id,
        asset=asset,
        desired_amount_usd=float(account["normal_transaction_amount_usd"]) * float(rng.uniform(5, 15)),
        transaction_type=str(rng.choice(["WITHDRAWAL", "TRANSFER"], p=[0.75, 0.25])),
        source_wallet_id=_customer_wallet(account_id, asset, entities, rng),
        destination_wallet_id=_external_wallet(asset, entities, rng, prefer_new=bool(rng.integers(0, 2))),
        device_id=str(rng.choice(entities.account_device_ids[account_id])),
        country=str(account["home_country"]),
        transaction_status="COMPLETED",
    )
    return _single_fraud_result(scenario_engine=scenario_engine, scenario_id=scenario_id, transaction=tx, market_match=match, auth_events=[], id_factory=id_factory)


def _structuring_case(
    *,
    config: HistoricalGeneratorConfig,
    account: dict[str, Any],
    transaction_count: int,
    entities: GeneratedEntities,
    market_lookup: MarketPriceLookup,
    rng: np.random.Generator,
    id_factory: DeterministicIdFactory,
    scenario_engine: ScenarioEngine,
) -> ScenarioGenerationResult:
    scenario_id = "structuring"
    account_id = str(account["account_id"])
    asset = _asset_for_account(account, rng)
    base_time = _scenario_time(config, rng)
    source_wallet_id = _customer_wallet(account_id, asset, entities, rng)
    destination_wallet_id = _external_wallet(asset, entities, rng)
    device_id = str(rng.choice(entities.account_device_ids[account_id]))
    execution_id = id_factory.next("scenario_execution")
    records: list[dict[str, Any]] = []
    matches: list[dict[str, Any]] = []
    audit_rows: list[dict[str, Any]] = []
    fraud_rows: list[dict[str, Any]] = []
    for index in range(transaction_count):
        tx, match = make_customer_transaction(
            id_factory=id_factory,
            market_lookup=market_lookup,
            event_timestamp=base_time + timedelta(minutes=index * int(rng.integers(1, 8))),
            account_id=account_id,
            asset=asset,
            desired_amount_usd=float(account["normal_transaction_amount_usd"]) * float(rng.uniform(0.7, 1.4)),
            transaction_type="WITHDRAWAL",
            source_wallet_id=source_wallet_id,
            destination_wallet_id=destination_wallet_id,
            device_id=device_id,
            country=str(account["home_country"]),
            transaction_status="COMPLETED",
        )
        records.append(tx)
        matches.append(match)
        audit_rows.append(_audit_row(scenario_engine=scenario_engine, scenario_id=scenario_id, execution_id=execution_id, transaction=tx, label_expected=True))
        fraud_rows.append(_fraud_row(scenario_engine, scenario_id, tx))
    return ScenarioGenerationResult([], records, audit_rows, fraud_rows, matches)


def _mule_activity_case(
    *,
    config: HistoricalGeneratorConfig,
    accounts: list[dict[str, Any]],
    entities: GeneratedEntities,
    market_lookup: MarketPriceLookup,
    rng: np.random.Generator,
    id_factory: DeterministicIdFactory,
    scenario_engine: ScenarioEngine,
) -> ScenarioGenerationResult:
    scenario_id = "mule_account_activity"
    asset = str(rng.choice(["BTC", "ETH"]))
    destination_wallet_id = str(rng.choice(entities.shared_external_wallet_ids))
    base_time = _scenario_time(config, rng)
    execution_id = id_factory.next("scenario_execution")
    records: list[dict[str, Any]] = []
    matches: list[dict[str, Any]] = []
    audit_rows: list[dict[str, Any]] = []
    fraud_rows: list[dict[str, Any]] = []
    for index, account in enumerate(accounts):
        account_id = str(account["account_id"])
        if asset not in entities.account_customer_wallet_ids[account_id]:
            asset = _asset_for_account(account, rng)
        tx, match = make_customer_transaction(
            id_factory=id_factory,
            market_lookup=market_lookup,
            event_timestamp=base_time + timedelta(minutes=index * int(rng.integers(7, 30))),
            account_id=account_id,
            asset=asset,
            desired_amount_usd=float(account["normal_transaction_amount_usd"]) * float(rng.uniform(1.2, 3.5)),
            transaction_type="WITHDRAWAL",
            source_wallet_id=_customer_wallet(account_id, asset, entities, rng),
            destination_wallet_id=destination_wallet_id,
            device_id=str(rng.choice(entities.account_device_ids[account_id])),
            country=str(account["home_country"]),
            transaction_status="COMPLETED",
        )
        records.append(tx)
        matches.append(match)
        audit_rows.append(_audit_row(scenario_engine=scenario_engine, scenario_id=scenario_id, execution_id=execution_id, transaction=tx, label_expected=True))
        fraud_rows.append(_fraud_row(scenario_engine, scenario_id, tx))
    return ScenarioGenerationResult([], records, audit_rows, fraud_rows, matches)


def _shared_device_case(
    *,
    config: HistoricalGeneratorConfig,
    account: dict[str, Any],
    related_accounts: list[dict[str, Any]],
    entities: GeneratedEntities,
    market_lookup: MarketPriceLookup,
    rng: np.random.Generator,
    id_factory: DeterministicIdFactory,
    scenario_engine: ScenarioEngine,
) -> ScenarioGenerationResult:
    scenario_id = "shared_suspicious_device"
    event_time = _scenario_time(config, rng)
    shared_device_id = str(rng.choice(entities.shared_device_ids))
    auths: list[dict[str, Any]] = []
    for index, related_account in enumerate(related_accounts):
        home_country = str(related_account["home_country"])
        auths.append(
            make_authentication_event(
                id_factory=id_factory,
                event_timestamp=event_time - timedelta(minutes=10 - index * 2),
                account_id=str(related_account["account_id"]),
                device_id=shared_device_id,
                country=_different_country(home_country, rng),
                ip_address=_ip_address(rng),
                login_success=index == 0,
                mfa_success=index == 0,
                password_reset_flag=False,
                failure_reason=None if index == 0 else "INVALID_PASSWORD",
            )
        )
    account_id = str(account["account_id"])
    asset = _asset_for_account(account, rng)
    tx, match = make_customer_transaction(
        id_factory=id_factory,
        market_lookup=market_lookup,
        event_timestamp=event_time,
        account_id=account_id,
        asset=asset,
        desired_amount_usd=float(account["normal_transaction_amount_usd"]) * float(rng.uniform(2, 6)),
        transaction_type="WITHDRAWAL",
        source_wallet_id=_customer_wallet(account_id, asset, entities, rng),
        destination_wallet_id=_external_wallet(asset, entities, rng),
        device_id=shared_device_id,
        country=auths[0]["country"],
        transaction_status="COMPLETED",
    )
    return _single_fraud_result(scenario_engine=scenario_engine, scenario_id=scenario_id, transaction=tx, market_match=match, auth_events=auths, id_factory=id_factory)


def _single_fraud_result(
    *,
    scenario_engine: ScenarioEngine,
    scenario_id: str,
    transaction: dict[str, Any],
    market_match: dict[str, Any],
    auth_events: list[dict[str, Any]],
    id_factory: DeterministicIdFactory,
) -> ScenarioGenerationResult:
    execution_id = id_factory.next("scenario_execution")
    return ScenarioGenerationResult(
        auth_events,
        [transaction],
        [
            _audit_row(
                scenario_engine=scenario_engine,
                scenario_id=scenario_id,
                execution_id=execution_id,
                transaction=transaction,
                label_expected=True,
            )
        ],
        [_fraud_row(scenario_engine, scenario_id, transaction)],
        [market_match],
    )


def _extend_result(
    result: ScenarioGenerationResult,
    auth_events: list[dict[str, Any]],
    transactions: list[dict[str, Any]],
    audit_rows: list[dict[str, Any]],
    fraud_rows: list[dict[str, Any]],
    market_matches: list[dict[str, Any]],
) -> None:
    auth_events.extend(result.authentication_events)
    transactions.extend(result.transactions)
    audit_rows.extend(result.audit_rows)
    fraud_rows.extend(result.fraud_rows)
    market_matches.extend(result.market_matches)


def _audit_row(
    *,
    scenario_engine: ScenarioEngine,
    scenario_id: str,
    execution_id: str,
    transaction: dict[str, Any],
    label_expected: bool,
) -> dict[str, Any]:
    scenario = scenario_engine.scenarios[scenario_id]
    return {
        "scenario_execution_id": execution_id,
        "scenario_id": scenario_id,
        "scenario_version": scenario.scenario_version,
        "fraud_type": scenario.fraud_type,
        "reason_codes": scenario.reason_codes,
        "transaction_id": transaction["transaction_id"],
        "account_id": transaction["account_id"],
        "event_timestamp": transaction["event_timestamp"],
        "label_expected": label_expected,
    }


def _fraud_row(
    scenario_engine: ScenarioEngine,
    scenario_id: str,
    transaction: dict[str, Any],
) -> dict[str, Any]:
    scenario = scenario_engine.scenarios[scenario_id]
    return {
        "transaction_id": transaction["transaction_id"],
        "transaction_timestamp": transaction["event_timestamp"],
        "scenario_id": scenario_id,
        "fraud_type": scenario.fraud_type,
        "reason_codes": scenario.reason_codes,
        "label_delay_hours": scenario.label_delay_hours,
    }


def _scenario_time(config: HistoricalGeneratorConfig, rng: np.random.Generator) -> datetime:
    start = parse_utc(config.time_range.start) + timedelta(hours=3)
    end = parse_utc(config.time_range.end) - timedelta(hours=4)
    return random_datetimes(rng, 1, start, end)[0]


def _transactions_to_frame(records: list[dict[str, Any]]) -> pd.DataFrame:
    frame = pd.DataFrame(records)
    if frame.empty:
        return frame
    for column in ["event_timestamp", "source_timestamp", "ingestion_timestamp"]:
        frame[column] = pd.to_datetime(frame[column], utc=True)
    return frame.sort_values("event_timestamp").reset_index(drop=True)


def _matches_to_frame(records: list[dict[str, Any]]) -> pd.DataFrame:
    frame = pd.DataFrame(records)
    if frame.empty:
        return frame
    for column in [
        "transaction_event_timestamp",
        "matched_market_candle_timestamp",
        "matched_market_candle_end_timestamp",
    ]:
        frame[column] = pd.to_datetime(frame[column], utc=True)
    return frame.sort_values("transaction_event_timestamp").reset_index(drop=True)


def _asset_for_account(account: dict[str, Any], rng: np.random.Generator) -> str:
    return str(rng.choice(list(account["preferred_assets"])))


def _customer_wallet(
    account_id: str,
    asset: str,
    entities: GeneratedEntities,
    rng: np.random.Generator,
) -> str:
    return str(rng.choice(entities.account_customer_wallet_ids[account_id][asset]))


def _external_wallet(
    asset: str,
    entities: GeneratedEntities,
    rng: np.random.Generator,
    *,
    prefer_new: bool = False,
) -> str:
    wallets = entities.wallets[
        (entities.wallets["wallet_type"] == "EXTERNAL")
        & (entities.wallets["supported_assets"].apply(lambda values: asset in values))
    ]
    if prefer_new:
        wallets = wallets[wallets["is_known_destination"] == False]  # noqa: E712
    if wallets.empty:
        wallets = entities.wallets[entities.wallets["wallet_type"] == "EXTERNAL"]
    return str(rng.choice(wallets["wallet_id"].to_numpy()))


def _wallet_pair_for_transaction(
    *,
    account: dict[str, Any],
    asset: str,
    transaction_type: str,
    entities: GeneratedEntities,
    rng: np.random.Generator,
) -> tuple[str | None, str | None]:
    account_id = str(account["account_id"])
    customer_wallet = _customer_wallet(account_id, asset, entities, rng)
    if transaction_type == "DEPOSIT":
        return None, customer_wallet
    if transaction_type == "WITHDRAWAL":
        return customer_wallet, _external_wallet(asset, entities, rng)
    return customer_wallet, _external_wallet(asset, entities, rng)


def _normal_amount_for_account(account: dict[str, Any], rng: np.random.Generator) -> float:
    normal_amount = float(account["normal_transaction_amount_usd"])
    return round(max(10.0, normal_amount * float(rng.lognormal(mean=0.0, sigma=0.38))), 2)
