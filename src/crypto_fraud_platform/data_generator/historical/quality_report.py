from __future__ import annotations

from typing import Any

import pandas as pd

from crypto_fraud_platform.data_generator.common.config import HistoricalGeneratorConfig
from crypto_fraud_platform.data_generator.common.schema_validation import (
    SchemaValidator,
    TRANSACTION_LEAKAGE_FIELDS,
)


def build_quality_report(
    *,
    config: HistoricalGeneratorConfig,
    accounts: pd.DataFrame,
    devices: pd.DataFrame,
    wallets: pd.DataFrame,
    authentication_events: pd.DataFrame,
    customer_transactions: pd.DataFrame,
    fraud_labels: pd.DataFrame,
    market_candles: pd.DataFrame,
    scenario_audit: pd.DataFrame,
    market_matches: pd.DataFrame,
    market_download_report: dict[str, Any],
    schema_validator: SchemaValidator | None = None,
) -> dict[str, Any]:
    report = {
        "status": "PASS",
        "record_counts": {
            "accounts": int(len(accounts)),
            "devices": int(len(devices)),
            "wallets": int(len(wallets)),
            "authentication_events": int(len(authentication_events)),
            "customer_transactions": int(len(customer_transactions)),
            "fraud_labels": int(len(fraud_labels)),
            "market_candles": int(len(market_candles)),
            "scenario_assignments": int(len(scenario_audit)),
            "market_enrichment_matches": int(len(market_matches)),
        },
        "uniqueness": _uniqueness_checks(
            accounts,
            devices,
            wallets,
            authentication_events,
            customer_transactions,
            fraud_labels,
            market_candles,
        ),
        "relationships": _relationship_checks(
            accounts,
            devices,
            wallets,
            authentication_events,
            customer_transactions,
            fraud_labels,
        ),
        "timestamps": _timestamp_checks(
            config,
            accounts,
            devices,
            authentication_events,
            customer_transactions,
            fraud_labels,
            market_matches,
        ),
        "values": _value_checks(config, customer_transactions, market_candles),
        "leakage": _leakage_checks(customer_transactions),
        "fraud_distribution": _fraud_distribution(
            customer_transactions,
            fraud_labels,
            scenario_audit,
        ),
        "market_download": market_download_report,
        "schema_validation": {},
    }

    if schema_validator is not None and config.validation.validate_schema_samples:
        report["schema_validation"] = _schema_validation_checks(
            schema_validator,
            accounts,
            devices,
            wallets,
            authentication_events,
            customer_transactions,
            fraud_labels,
            market_candles,
        )

    failed_sections = _failed_paths(report)
    if failed_sections:
        report["status"] = "FAIL"
        report["failed_checks"] = failed_sections
    return report


def assert_quality_passed(report: dict[str, Any]) -> None:
    if report["status"] != "PASS":
        raise ValueError(f"Historical pilot quality checks failed: {report.get('failed_checks', [])}")


def _uniqueness_checks(
    accounts: pd.DataFrame,
    devices: pd.DataFrame,
    wallets: pd.DataFrame,
    authentication_events: pd.DataFrame,
    customer_transactions: pd.DataFrame,
    fraud_labels: pd.DataFrame,
    market_candles: pd.DataFrame,
) -> dict[str, Any]:
    return {
        "account_ids_unique": _is_unique(accounts, "account_id"),
        "device_ids_unique": _is_unique(devices, "device_id"),
        "wallet_ids_unique": _is_unique(wallets, "wallet_id"),
        "authentication_event_ids_unique": _is_unique(authentication_events, "event_id"),
        "login_ids_unique": _is_unique(authentication_events, "login_id"),
        "transaction_event_ids_unique": _is_unique(customer_transactions, "event_id"),
        "transaction_ids_unique": _is_unique(customer_transactions, "transaction_id"),
        "fraud_label_event_ids_unique": _is_unique(fraud_labels, "event_id"),
        "market_candle_natural_keys_unique": _is_unique(
            market_candles,
            ["product_id", "candle_start_timestamp"],
        ),
    }


def _relationship_checks(
    accounts: pd.DataFrame,
    devices: pd.DataFrame,
    wallets: pd.DataFrame,
    authentication_events: pd.DataFrame,
    customer_transactions: pd.DataFrame,
    fraud_labels: pd.DataFrame,
) -> dict[str, bool]:
    account_ids = set(accounts["account_id"])
    device_ids = set(devices["device_id"])
    wallet_ids = set(wallets["wallet_id"])
    transaction_ids = set(customer_transactions["transaction_id"])
    return {
        "device_primary_accounts_resolve": _nullable_subset(devices["primary_account_id"], account_ids),
        "wallet_owner_accounts_resolve": _nullable_subset(wallets["owner_account_id"], account_ids),
        "authentication_accounts_resolve": set(authentication_events["account_id"]).issubset(account_ids),
        "authentication_devices_resolve": set(authentication_events["device_id"]).issubset(device_ids),
        "transaction_accounts_resolve": set(customer_transactions["account_id"]).issubset(account_ids),
        "transaction_devices_resolve": set(customer_transactions["device_id"]).issubset(device_ids),
        "transaction_source_wallets_resolve": _nullable_subset(
            customer_transactions["source_wallet_id"],
            wallet_ids,
        ),
        "transaction_destination_wallets_resolve": _nullable_subset(
            customer_transactions["destination_wallet_id"],
            wallet_ids,
        ),
        "fraud_label_transactions_resolve": set(fraud_labels["transaction_id"]).issubset(transaction_ids),
    }


def _timestamp_checks(
    config: HistoricalGeneratorConfig,
    accounts: pd.DataFrame,
    devices: pd.DataFrame,
    authentication_events: pd.DataFrame,
    customer_transactions: pd.DataFrame,
    fraud_labels: pd.DataFrame,
    market_matches: pd.DataFrame,
) -> dict[str, Any]:
    label_join = fraud_labels.merge(
        customer_transactions[["transaction_id", "event_timestamp"]],
        on="transaction_id",
        how="left",
        suffixes=("_label", "_transaction"),
    )
    activity_start = pd.Timestamp(config.time_range.start)
    activity_end = pd.Timestamp(config.time_range.end)
    return {
        "account_updated_at_not_before_created_at": bool((accounts["updated_at"] >= accounts["created_at"]).all()),
        "device_last_seen_at_not_before_first_seen_at": bool((devices["last_seen_at"] >= devices["first_seen_at"]).all()),
        "authentication_metadata_ordered": _event_metadata_ordered(authentication_events),
        "transaction_metadata_ordered": _event_metadata_ordered(customer_transactions),
        "fraud_label_metadata_ordered": _event_metadata_ordered(fraud_labels),
        "labels_after_transactions": bool(
            (label_join["label_timestamp"] > label_join["event_timestamp_transaction"]).all()
        ),
        "transactions_inside_configured_period": bool(
            (
                (customer_transactions["event_timestamp"] >= activity_start)
                & (customer_transactions["event_timestamp"] <= activity_end)
            ).all()
        ),
        "authentications_inside_configured_period": bool(
            (
                (authentication_events["event_timestamp"] >= activity_start)
                & (authentication_events["event_timestamp"] <= activity_end)
            ).all()
        ),
        "market_candle_not_from_future": bool(
            (
                market_matches["matched_market_candle_end_timestamp"]
                <= market_matches["transaction_event_timestamp"]
            ).all()
        )
        if not market_matches.empty
        else False,
    }


def _value_checks(
    config: HistoricalGeneratorConfig,
    customer_transactions: pd.DataFrame,
    market_candles: pd.DataFrame,
) -> dict[str, Any]:
    expected_amount = (
        customer_transactions["crypto_quantity"].astype(float)
        * customer_transactions["market_price_usd"].astype(float)
    )
    max_amount_delta = (
        customer_transactions["transaction_amount_usd"].astype(float) - expected_amount
    ).abs().max()
    return {
        "transactions_use_supported_assets": set(customer_transactions["asset"]).issubset(set(config.assets)),
        "positive_crypto_quantities": bool((customer_transactions["crypto_quantity"] > 0).all()),
        "positive_market_prices": bool((customer_transactions["market_price_usd"] > 0).all()),
        "transaction_amount_calculation_within_tolerance": bool(max_amount_delta <= 0.0001),
        "transaction_amount_max_delta": float(max_amount_delta),
        "market_products_supported": set(market_candles["product_id"]).issubset({"BTC-USD", "ETH-USD"}),
        "market_prices_positive": bool(
            (
                (market_candles["open_price_usd"] > 0)
                & (market_candles["high_price_usd"] > 0)
                & (market_candles["low_price_usd"] > 0)
                & (market_candles["close_price_usd"] > 0)
            ).all()
        ),
    }


def _leakage_checks(customer_transactions: pd.DataFrame) -> dict[str, Any]:
    leakage_columns = sorted(TRANSACTION_LEAKAGE_FIELDS.intersection(customer_transactions.columns))
    return {
        "customer_transactions_have_no_leakage_columns": leakage_columns == [],
        "leakage_columns": leakage_columns,
    }


def _fraud_distribution(
    customer_transactions: pd.DataFrame,
    fraud_labels: pd.DataFrame,
    scenario_audit: pd.DataFrame,
) -> dict[str, Any]:
    confirmed = fraud_labels[fraud_labels["is_fraud"] == True]  # noqa: E712
    confirmed_count = int(len(confirmed))
    transaction_count = int(len(customer_transactions))
    fraud_rate = confirmed_count / transaction_count if transaction_count else 0.0
    completed_count = int((customer_transactions["transaction_status"] == "COMPLETED").sum())
    completed_fraud_rate = confirmed_count / completed_count if completed_count else 0.0
    if scenario_audit.empty:
        scenario_distribution: dict[str, int] = {}
    else:
        labeled_audit = scenario_audit[scenario_audit["label_expected"] == True]  # noqa: E712
        scenario_distribution = {
            str(key): int(value)
            for key, value in labeled_audit.groupby("scenario_id")["transaction_id"].count().items()
        }
    return {
        "confirmed_fraud_count": confirmed_count,
        "transaction_count": transaction_count,
        "completed_transaction_count": completed_count,
        "fraud_rate": fraud_rate,
        "completed_transaction_fraud_rate": completed_fraud_rate,
        "fraud_rate_within_approved_range": bool(0.005 <= fraud_rate <= 0.01),
        "scenario_distribution": scenario_distribution,
    }


def _schema_validation_checks(
    schema_validator: SchemaValidator,
    accounts: pd.DataFrame,
    devices: pd.DataFrame,
    wallets: pd.DataFrame,
    authentication_events: pd.DataFrame,
    customer_transactions: pd.DataFrame,
    fraud_labels: pd.DataFrame,
    market_candles: pd.DataFrame,
) -> dict[str, Any]:
    checks: dict[str, Any] = {}
    datasets = {
        "account": accounts,
        "device": devices,
        "wallet": wallets,
        "authentication-event": authentication_events,
        "customer-transaction": customer_transactions,
        "fraud-label": fraud_labels,
        "historical-market-candle": market_candles.drop(columns=["event_date"], errors="ignore"),
    }
    for schema_name, dataframe in datasets.items():
        try:
            sample_count = schema_validator.validate_dataframe(
                schema_name,
                dataframe.drop(columns=["event_date", "label_date"], errors="ignore"),
                sample_limit=50,
            )
            checks[schema_name] = {"passed": True, "sample_count": sample_count}
        except Exception as exc:  # pragma: no cover - surfaced in report for CLI users
            checks[schema_name] = {"passed": False, "error": str(exc)}
    return checks


def _event_metadata_ordered(events: pd.DataFrame) -> bool:
    return bool(
        (
            (events["event_timestamp"] <= events["source_timestamp"])
            & (events["source_timestamp"] <= events["ingestion_timestamp"])
        ).all()
    )


def _is_unique(dataframe: pd.DataFrame, columns: str | list[str]) -> bool:
    return bool(~dataframe.duplicated(columns).any())


def _nullable_subset(values: pd.Series, allowed_values: set[Any]) -> bool:
    non_null = {value for value in values if pd.notna(value)}
    return non_null.issubset(allowed_values)


def _failed_paths(value: Any, prefix: str = "") -> list[str]:
    failures: list[str] = []
    if isinstance(value, dict):
        for key, child in value.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            failures.extend(_failed_paths(child, path))
    elif isinstance(value, bool) and not value:
        failures.append(prefix)
    return failures
