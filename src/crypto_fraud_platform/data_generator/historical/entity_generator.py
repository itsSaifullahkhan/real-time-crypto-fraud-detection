from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any

import numpy as np
import pandas as pd

from crypto_fraud_platform.data_generator.common.identifiers import DeterministicIdFactory
from crypto_fraud_platform.data_generator.common.time_utils import parse_utc


COUNTRIES = ["US", "CA", "GB", "DE", "FR", "NL", "SG", "AU", "JP", "BR", "IN", "AE"]
DEVICE_TYPES = ["MOBILE", "DESKTOP", "TABLET"]
OPERATING_SYSTEMS = ["ANDROID", "IOS", "WINDOWS", "MACOS", "LINUX", "OTHER"]
WALLET_RISK_LEVELS = ["LOW", "MEDIUM", "HIGH", "UNKNOWN"]


@dataclass(frozen=True)
class GeneratedEntities:
    accounts: pd.DataFrame
    devices: pd.DataFrame
    wallets: pd.DataFrame
    account_device_ids: dict[str, list[str]]
    account_customer_wallet_ids: dict[str, dict[str, list[str]]]
    external_wallet_ids: list[str]
    shared_device_ids: list[str]
    shared_external_wallet_ids: list[str]


def generate_entities(
    *,
    account_count: int,
    start_timestamp: datetime,
    end_timestamp: datetime,
    rng: np.random.Generator,
    id_factory: DeterministicIdFactory,
) -> GeneratedEntities:
    accounts = _generate_accounts(
        account_count=account_count,
        activity_start=parse_utc(start_timestamp),
        rng=rng,
        id_factory=id_factory,
    )
    devices, account_device_ids, shared_device_ids = _generate_devices(
        accounts=accounts,
        activity_start=parse_utc(start_timestamp),
        activity_end=parse_utc(end_timestamp),
        rng=rng,
        id_factory=id_factory,
    )
    (
        wallets,
        account_customer_wallet_ids,
        external_wallet_ids,
        shared_external_wallet_ids,
    ) = _generate_wallets(
        accounts=accounts,
        activity_start=parse_utc(start_timestamp),
        rng=rng,
        id_factory=id_factory,
    )
    return GeneratedEntities(
        accounts=accounts,
        devices=devices,
        wallets=wallets,
        account_device_ids=account_device_ids,
        account_customer_wallet_ids=account_customer_wallet_ids,
        external_wallet_ids=external_wallet_ids,
        shared_device_ids=shared_device_ids,
        shared_external_wallet_ids=shared_external_wallet_ids,
    )


def _generate_accounts(
    *,
    account_count: int,
    activity_start: datetime,
    rng: np.random.Generator,
    id_factory: DeterministicIdFactory,
) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for index in range(account_count):
        account_id = id_factory.next("account")
        created_days_before = int(rng.integers(20, 540))
        created_at = activity_start - timedelta(days=created_days_before, hours=int(rng.integers(0, 24)))
        updated_at = activity_start - timedelta(days=int(rng.integers(1, 20)))
        risk_tier = str(rng.choice(["LOW", "MEDIUM", "HIGH"], p=[0.72, 0.22, 0.06]))
        amount_multiplier = {"LOW": 1.0, "MEDIUM": 1.9, "HIGH": 4.5}[risk_tier]
        normal_amount = round(float(rng.lognormal(mean=5.2, sigma=0.55) * amount_multiplier), 2)
        normal_frequency = round(float(rng.gamma(shape=2.2, scale=0.9)), 2)
        preferred_assets = _preferred_assets(rng)
        records.append(
            {
                "account_id": account_id,
                "schema_version": "1.0",
                "created_at": created_at,
                "updated_at": max(updated_at, created_at),
                "home_country": str(rng.choice(COUNTRIES, p=_country_probabilities())),
                "kyc_level": str(rng.choice(["BASIC", "STANDARD", "ENHANCED"], p=[0.18, 0.62, 0.20])),
                "customer_risk_tier": risk_tier,
                "normal_transaction_amount_usd": normal_amount,
                "normal_transaction_frequency_per_day": normal_frequency,
                "preferred_assets": preferred_assets,
                "account_status": str(rng.choice(["ACTIVE", "SUSPENDED", "CLOSED"], p=[0.96, 0.03, 0.01])),
            }
        )
    return pd.DataFrame(records)


def _generate_devices(
    *,
    accounts: pd.DataFrame,
    activity_start: datetime,
    activity_end: datetime,
    rng: np.random.Generator,
    id_factory: DeterministicIdFactory,
) -> tuple[pd.DataFrame, dict[str, list[str]], list[str]]:
    records: list[dict[str, Any]] = []
    account_device_ids: dict[str, list[str]] = {}
    for account in accounts.to_dict(orient="records"):
        account_id = str(account["account_id"])
        device_count = int(rng.choice([1, 2, 3], p=[0.62, 0.32, 0.06]))
        account_device_ids[account_id] = []
        for device_index in range(device_count):
            first_seen = activity_start - timedelta(days=int(rng.integers(7, 360)))
            last_seen = activity_end - timedelta(hours=int(rng.integers(0, 96)))
            device_id = id_factory.next("device")
            account_device_ids[account_id].append(device_id)
            records.append(
                {
                    "device_id": device_id,
                    "schema_version": "1.0",
                    "first_seen_at": first_seen,
                    "last_seen_at": max(last_seen, first_seen),
                    "device_type": str(rng.choice(DEVICE_TYPES, p=[0.68, 0.27, 0.05])),
                    "operating_system": _operating_system_for_device(device_index, rng),
                    "is_trusted": bool(rng.choice([True, False], p=[0.86, 0.14])),
                    "device_country": str(account["home_country"]),
                    "primary_account_id": account_id,
                }
            )

    shared_device_ids: list[str] = []
    for shared_index in range(2):
        device_id = id_factory.next("shared_device")
        shared_device_ids.append(device_id)
        first_seen = activity_start - timedelta(days=int(rng.integers(1, 15)))
        records.append(
            {
                "device_id": device_id,
                "schema_version": "1.0",
                "first_seen_at": first_seen,
                "last_seen_at": activity_end,
                "device_type": "DESKTOP",
                "operating_system": str(rng.choice(["WINDOWS", "LINUX", "OTHER"])),
                "is_trusted": False,
                "device_country": str(rng.choice(["NL", "SG", "AE", "BR"])),
                "primary_account_id": None,
            }
        )

    return pd.DataFrame(records), account_device_ids, shared_device_ids


def _generate_wallets(
    *,
    accounts: pd.DataFrame,
    activity_start: datetime,
    rng: np.random.Generator,
    id_factory: DeterministicIdFactory,
) -> tuple[pd.DataFrame, dict[str, dict[str, list[str]]], list[str], list[str]]:
    records: list[dict[str, Any]] = []
    account_customer_wallet_ids: dict[str, dict[str, list[str]]] = {}
    for account in accounts.to_dict(orient="records"):
        account_id = str(account["account_id"])
        account_customer_wallet_ids[account_id] = {"BTC": [], "ETH": []}
        for asset in ["BTC", "ETH"]:
            wallet_id = id_factory.next("wallet")
            account_customer_wallet_ids[account_id][asset].append(wallet_id)
            records.append(
                {
                    "wallet_id": wallet_id,
                    "schema_version": "1.0",
                    "owner_account_id": account_id,
                    "wallet_type": "CUSTOMER",
                    "first_seen_at": activity_start - timedelta(days=int(rng.integers(14, 360))),
                    "risk_level": "LOW",
                    "is_known_destination": True,
                    "supported_assets": [asset],
                }
            )

    external_wallet_ids: list[str] = []
    external_count = max(140, len(accounts) + 20)
    for _ in range(external_count):
        wallet_id = id_factory.next("external_wallet")
        external_wallet_ids.append(wallet_id)
        records.append(
            {
                "wallet_id": wallet_id,
                "schema_version": "1.0",
                "owner_account_id": None,
                "wallet_type": "EXTERNAL",
                "first_seen_at": activity_start - timedelta(days=int(rng.integers(0, 240))),
                "risk_level": str(rng.choice(WALLET_RISK_LEVELS, p=[0.62, 0.18, 0.08, 0.12])),
                "is_known_destination": bool(rng.choice([True, False], p=[0.72, 0.28])),
                "supported_assets": _preferred_assets(rng),
            }
        )

    shared_external_wallet_ids: list[str] = []
    for _ in range(2):
        wallet_id = id_factory.next("shared_external_wallet")
        shared_external_wallet_ids.append(wallet_id)
        external_wallet_ids.append(wallet_id)
        records.append(
            {
                "wallet_id": wallet_id,
                "schema_version": "1.0",
                "owner_account_id": None,
                "wallet_type": "EXTERNAL",
                "first_seen_at": activity_start - timedelta(days=int(rng.integers(0, 10))),
                "risk_level": "HIGH",
                "is_known_destination": False,
                "supported_assets": ["BTC", "ETH"],
            }
        )

    return pd.DataFrame(records), account_customer_wallet_ids, external_wallet_ids, shared_external_wallet_ids


def _preferred_assets(rng: np.random.Generator) -> list[str]:
    choice = str(rng.choice(["BTC", "ETH", "BOTH"], p=[0.42, 0.38, 0.20]))
    if choice == "BTC":
        return ["BTC"]
    if choice == "ETH":
        return ["ETH"]
    return ["BTC", "ETH"]


def _country_probabilities() -> list[float]:
    return [0.48, 0.08, 0.08, 0.06, 0.04, 0.03, 0.04, 0.04, 0.04, 0.04, 0.04, 0.03]


def _operating_system_for_device(device_index: int, rng: np.random.Generator) -> str:
    if device_index == 0:
        return str(rng.choice(["IOS", "ANDROID"], p=[0.55, 0.45]))
    return str(rng.choice(OPERATING_SYSTEMS, p=[0.18, 0.16, 0.28, 0.18, 0.12, 0.08]))
