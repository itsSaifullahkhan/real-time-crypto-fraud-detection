# Feature Catalog

## Purpose

This Phase 3 draft maps fraud scenarios to expected candidate features. These features are not implemented. All entries are:

```text
Design candidate - definition to be finalized in Phase 8.
```

The catalogue exists to keep Phase 3 fraud-scenario design aligned with later offline and online feature engineering.

## Feature Availability Rules

- Features must use only data available at or before the prediction timestamp.
- Future investigation outcomes must not be used as model input.
- Fraud scenario metadata must not be written into model-facing records.
- Offline and online definitions must later be made consistent in Phase 8 and streaming phases.

Forbidden model inputs:

- `is_fraud`
- `fraud_type`
- `scenario_id`
- `investigation_status`
- label status
- generator-internal flags

## Current-Event Features

| Feature | Status | Source | Scenario mapping | Notes |
| --- | --- | --- | --- | --- |
| `transaction_amount_usd` | Design candidate - definition to be finalized in Phase 8. | Current customer transaction | Unusual amount, volatility scenario | Raw current transaction amount. |
| `transaction_type` | Design candidate - definition to be finalized in Phase 8. | Current customer transaction | Unusual amount, account takeover, structuring | Uses approved enum values only. |
| `transaction_amount_adjusted_for_market_context` | Design candidate - definition to be finalized in Phase 8. | Current transaction plus prior market events | Optional volatility scenario | Must use market context available before prediction. |

## Account-Profile Features

| Feature | Status | Source | Scenario mapping | Notes |
| --- | --- | --- | --- | --- |
| `amount_vs_customer_average` | Design candidate - definition to be finalized in Phase 8. | Account profile and current transaction | Account takeover, unusual amount, volatility scenario | Compares current amount with `normal_transaction_amount_usd`. |
| `transaction_velocity_vs_normal` | Design candidate - definition to be finalized in Phase 8. | Account profile and prior transactions | High velocity, structuring | Compares recent count with `normal_transaction_frequency_per_day`. |
| `customer_risk_tier` | Design candidate - definition to be finalized in Phase 8. | Account master data | Unusual amount | Uses Phase 2 account enum. |
| `account_age_days` | Design candidate - definition to be finalized in Phase 8. | Account master data | Unusual amount | Derived from account `created_at` and prediction timestamp. |
| `is_new_country` | Design candidate - definition to be finalized in Phase 8. | Account profile and prior authentication events | Account takeover | Compares current country with home or recent normal country. |

## Transaction-Window Features

| Feature | Status | Source | Scenario mapping | Notes |
| --- | --- | --- | --- | --- |
| `transaction_count_last_1_minute` | Design candidate - definition to be finalized in Phase 8. | Prior customer transactions | High velocity | Event-time rolling count. |
| `transaction_count_last_5_minutes` | Design candidate - definition to be finalized in Phase 8. | Prior customer transactions | High velocity, structuring | Event-time rolling count. |
| `transaction_count_last_1_hour` | Design candidate - definition to be finalized in Phase 8. | Prior customer transactions | High velocity, structuring, mule-account activity | Event-time rolling count. |
| `withdrawal_sum_last_5_minutes` | Design candidate - definition to be finalized in Phase 8. | Prior customer transactions | High velocity, structuring | Event-time rolling sum. |
| `withdrawal_sum_last_1_hour` | Design candidate - definition to be finalized in Phase 8. | Prior customer transactions | High velocity, structuring | Event-time rolling sum. |
| `time_since_previous_transaction` | Design candidate - definition to be finalized in Phase 8. | Prior customer transactions | Account takeover, high velocity | Time gap before current transaction. |
| `unique_destination_wallets_last_1_hour` | Design candidate - definition to be finalized in Phase 8. | Prior customer transactions | Mule-account activity | Counts unique destination wallets before prediction. |

## Authentication-Window Features

| Feature | Status | Source | Scenario mapping | Notes |
| --- | --- | --- | --- | --- |
| `failed_logins_last_10_minutes` | Design candidate - definition to be finalized in Phase 8. | Prior authentication events | Account takeover, shared suspicious device | Event-time rolling count. |
| `mfa_failures_last_10_minutes` | Design candidate - definition to be finalized in Phase 8. | Prior authentication events | Account takeover | Event-time rolling count. |
| `password_reset_before_transaction` | Design candidate - definition to be finalized in Phase 8. | Prior authentication events | Account takeover | Uses authentication events before transaction. |
| `country_change_flag` | Design candidate - definition to be finalized in Phase 8. | Prior authentication events | Shared suspicious device | Derived from recent login country changes. |

## Device Features

| Feature | Status | Source | Scenario mapping | Notes |
| --- | --- | --- | --- | --- |
| `is_new_device` | Design candidate - definition to be finalized in Phase 8. | Device master data and prior authentication events | Account takeover, shared suspicious device | Checks whether device was previously observed for the account. |
| `device_change_flag` | Design candidate - definition to be finalized in Phase 8. | Prior authentication events | Shared suspicious device | Flags change from normal device usage. |
| `accounts_connected_to_device` | Design candidate - definition to be finalized in Phase 8. | Prior authentication events | Shared suspicious device | Counts distinct accounts tied to device before prediction. |
| `device_account_count_recent` | Design candidate - definition to be finalized in Phase 8. | Prior authentication events | Shared suspicious device | Recent distinct-account count for a device. |

## Wallet Features

| Feature | Status | Source | Scenario mapping | Notes |
| --- | --- | --- | --- | --- |
| `destination_wallet_is_new` | Design candidate - definition to be finalized in Phase 8. | Wallet master data and prior transactions | Account takeover, unusual amount, volatility scenario | New relative to account history. |
| `repeated_destination_wallet_count` | Design candidate - definition to be finalized in Phase 8. | Prior wallet activity and prior transactions | Structuring | Counts repeated use in a rolling window. |
| `accounts_connected_to_destination_wallet` | Design candidate - definition to be finalized in Phase 8. | Prior wallet activity and prior transactions | Mule-account activity | Distinct accounts connected before prediction. |
| `destination_wallet_transaction_count` | Design candidate - definition to be finalized in Phase 8. | Prior transactions | Mule-account activity | Rolling count for a destination wallet. |
| `destination_wallet_risk_score` | Design candidate - definition to be finalized in Phase 8. | Wallet master data and approved later logic | Mule-account activity | Placeholder candidate; final definition deferred. |
| `wallet_recent_fraud_count` | Design candidate - definition to be finalized in Phase 8. | Prior confirmed labels only | Mule-account activity | Must never use future labels. |

## Market-Context Features

Historical market-context features will be calculated from prior Coinbase one-minute candles. Live market-context features will be calculated from Coinbase WebSocket trade-level events and streaming state. Both paths must later use the same product mapping, event-time rules, five-minute window definition, timezone, null handling, freshness policy, and aggregation formulas.

| Feature | Status | Source | Scenario mapping | Notes |
| --- | --- | --- | --- | --- |
| `latest_market_price_usd` | Design candidate - definition to be finalized in Phase 8. | Prior historical candles or live market events | Transaction valuation, optional volatility scenario | Must use the latest completed market state at or before prediction. |
| `market_volatility_last_5_minutes` | Design candidate - definition to be finalized in Phase 8. | Prior historical candles or live market events | Optional volatility scenario | Uses Coinbase public market data only. |
| `price_change_last_5_minutes` | Design candidate - definition to be finalized in Phase 8. | Prior historical candles or live market events | Optional volatility scenario | Event-time price movement. |
| `market_trade_volume_last_5_minutes` | Design candidate - definition to be finalized in Phase 8. | Prior historical candles or live market events | Optional volatility scenario | Rolling market trade volume. |
| `market_data_freshness_seconds` | Design candidate - definition to be finalized in Phase 8. | Matched historical candle or live market state | All transaction enrichment | Difference between transaction timestamp and matched market-state timestamp. |
| `matched_market_candle_timestamp` | Design candidate - definition to be finalized in Phase 8. | Historical market candle enrichment | Offline transaction features | Audit field for point-in-time historical enrichment. |

## Scenario-to-Feature Mapping

| Scenario | Candidate features |
| --- | --- |
| Account takeover | `failed_logins_last_10_minutes`, `mfa_failures_last_10_minutes`, `is_new_device`, `is_new_country`, `password_reset_before_transaction`, `amount_vs_customer_average`, `destination_wallet_is_new`, `time_since_previous_transaction` |
| High transaction velocity | `transaction_count_last_1_minute`, `transaction_count_last_5_minutes`, `transaction_count_last_1_hour`, `withdrawal_sum_last_5_minutes`, `withdrawal_sum_last_1_hour`, `transaction_velocity_vs_normal`, `time_since_previous_transaction` |
| Unusual transaction amount | `transaction_amount_usd`, `amount_vs_customer_average`, `customer_risk_tier`, `transaction_type`, `account_age_days`, `destination_wallet_is_new` |
| Structuring | `transaction_count_last_5_minutes`, `withdrawal_sum_last_5_minutes`, `transaction_count_last_1_hour`, `withdrawal_sum_last_1_hour`, `repeated_destination_wallet_count`, `transaction_velocity_vs_normal` |
| Mule-account activity | `accounts_connected_to_destination_wallet`, `destination_wallet_transaction_count`, `destination_wallet_risk_score`, `unique_destination_wallets_last_1_hour`, `transaction_count_last_1_hour`, `wallet_recent_fraud_count` |
| Shared suspicious device | `accounts_connected_to_device`, `device_account_count_recent`, `is_new_device`, `failed_logins_last_10_minutes`, `country_change_flag`, `device_change_flag` |
| Optional high-volatility unusual withdrawal | `market_volatility_last_5_minutes`, `price_change_last_5_minutes`, `market_trade_volume_last_5_minutes`, `amount_vs_customer_average`, `destination_wallet_is_new`, `transaction_amount_adjusted_for_market_context` |

## Implementation Status

No feature engineering code, offline feature tables, online state stores, model inputs, or streaming feature definitions are implemented in Phase 3.

Status: Phase 3 draft mapping complete and awaiting review before Phase 8 feature definitions.
