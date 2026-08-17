from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any

import numpy as np
import pandas as pd

from crypto_fraud_platform.data_generator.common.config import HistoricalGeneratorConfig
from crypto_fraud_platform.data_generator.common.identifiers import DeterministicIdFactory
from crypto_fraud_platform.data_generator.common.time_utils import parse_utc, random_datetimes
from crypto_fraud_platform.data_generator.historical.entity_generator import GeneratedEntities


FAILURE_REASONS = ["INVALID_PASSWORD", "MFA_FAILED", "ACCOUNT_LOCKED", "DEVICE_BLOCKED", "OTHER"]
RESERVED_IP_BLOCKS = ["203.0.113", "198.51.100", "192.0.2"]


def make_authentication_event(
    *,
    id_factory: DeterministicIdFactory,
    event_timestamp: datetime,
    account_id: str,
    device_id: str,
    country: str,
    ip_address: str,
    login_success: bool,
    mfa_success: bool,
    password_reset_flag: bool,
    failure_reason: str | None,
) -> dict[str, Any]:
    event_time = parse_utc(event_timestamp)
    source_time = event_time + timedelta(seconds=1)
    return {
        "event_id": id_factory.next("authentication_event"),
        "event_type": "authentication_event",
        "schema_version": "1.0",
        "source": "authentication_generator",
        "event_timestamp": event_time,
        "source_timestamp": source_time,
        "ingestion_timestamp": source_time + timedelta(seconds=1),
        "login_id": id_factory.next("login"),
        "account_id": account_id,
        "device_id": device_id,
        "country": country,
        "ip_address": ip_address,
        "login_success": login_success,
        "mfa_success": mfa_success,
        "password_reset_flag": password_reset_flag,
        "failure_reason": failure_reason,
    }


def generate_normal_authentication_events(
    *,
    config: HistoricalGeneratorConfig,
    entities: GeneratedEntities,
    existing_events: list[dict[str, Any]],
    rng: np.random.Generator,
    id_factory: DeterministicIdFactory,
) -> pd.DataFrame:
    target_count = config.scale.target_authentication_event_count
    records = list(existing_events)
    remaining = max(0, target_count - len(records))
    if remaining == 0:
        return _events_to_frame(records)

    active_accounts = entities.accounts[entities.accounts["account_status"] == "ACTIVE"]
    account_records = active_accounts.to_dict(orient="records")
    timestamps = random_datetimes(rng, remaining, config.time_range.start, config.time_range.end)

    for event_timestamp in timestamps:
        account = account_records[int(rng.integers(0, len(account_records)))]
        account_id = str(account["account_id"])
        device_id = str(rng.choice(entities.account_device_ids[account_id]))
        login_success = bool(rng.choice([True, False], p=[0.955, 0.045]))
        mfa_success = True if login_success else bool(rng.choice([True, False], p=[0.25, 0.75]))
        failure_reason = None
        if not login_success:
            failure_reason = "MFA_FAILED" if not mfa_success else str(rng.choice(FAILURE_REASONS))
        country = str(account["home_country"])
        if rng.random() < 0.035:
            country = _different_country(country, rng)
        records.append(
            make_authentication_event(
                id_factory=id_factory,
                event_timestamp=event_timestamp,
                account_id=account_id,
                device_id=device_id,
                country=country,
                ip_address=_ip_address(rng),
                login_success=login_success,
                mfa_success=mfa_success,
                password_reset_flag=bool(rng.choice([False, True], p=[0.992, 0.008])),
                failure_reason=failure_reason,
            )
        )

    return _events_to_frame(records)


def _events_to_frame(records: list[dict[str, Any]]) -> pd.DataFrame:
    frame = pd.DataFrame(records)
    if frame.empty:
        return frame
    for column in ["event_timestamp", "source_timestamp", "ingestion_timestamp"]:
        frame[column] = pd.to_datetime(frame[column], utc=True)
    return frame.sort_values("event_timestamp").reset_index(drop=True)


def _ip_address(rng: np.random.Generator) -> str:
    block = str(rng.choice(RESERVED_IP_BLOCKS))
    return f"{block}.{int(rng.integers(1, 255))}"


def _different_country(current_country: str, rng: np.random.Generator) -> str:
    choices = [country for country in ["US", "CA", "GB", "DE", "FR", "NL", "SG", "BR", "AE"] if country != current_country]
    return str(rng.choice(choices))
