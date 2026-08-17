# Fraud Scenario Catalogue

## 1. Purpose

This Phase 3 catalogue defines controlled fraud scenarios for the Real-Time Crypto Fraud Detection Platform. The scenarios will later guide historical and real-time Python data generators, but this phase only defines behaviour, event sequences, timing, labels, expected features, reason codes, and validation rules.

No datasets, generator classes, cloud resources, Databricks assets, model thresholds, or dashboards are implemented in Phase 3.

## 2. Fraud-Generation Principles

- Fraud scenarios must be realistic multi-event behaviour, not a transaction with `is_fraud = true`.
- Fraud labels remain separate `fraud_label` events.
- Customer transactions must stay compatible with `customer-transaction.schema.json` and must not contain `is_fraud`, `fraud_type`, `fraud_label`, or `confirmed_fraud`.
- Scenario rules are relative to each account's profile wherever possible.
- Scenario metadata is generation-control information only and must not become model-facing transaction data.
- Each fraud scenario must have normal control cases so the model does not learn a single deterministic rule.

## 3. Target Fraud-Rate Range

The approved project range is 0.5% to 1.0% of customer transactions. `config/fraud-rules.yaml` records this as:

```yaml
target_fraud_rate_range:
  min: 0.005
  max: 0.01
```

Final parameter values and scenario mix will be calibrated during data-generation testing. The catalogue does not define final fraud rates beyond the approved range.

## 4. Event and Label Separation

Correct conceptual order:

```text
normal or suspicious activity events
-> customer transaction event
-> fraud decision event
-> investigation delay
-> separate fraud-label event
```

Fraud decisions represent model output before investigation. Fraud labels represent delayed investigation outcomes after the prediction point.

## 5. Scenario Summary Table

| Scenario | Enabled | Entities | Event types | Expected feature signals | Reason codes | Fraud type | Difficulty | False-positive risk |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Account takeover | Yes | account, device, wallet | authentication, transaction, decision, label | failed logins, MFA failures, new device, new country, password reset, unusual amount, new destination, transaction recency | `NEW_DEVICE`, `NEW_COUNTRY`, `FAILED_LOGIN_ACTIVITY`, `UNUSUAL_AMOUNT`, `NEW_DESTINATION_WALLET` | `ACCOUNT_TAKEOVER` | `MEDIUM` | Medium |
| High transaction velocity | Yes | account, device, wallet | transaction, decision, label | 1-minute, 5-minute, and 1-hour counts; withdrawal sums; velocity versus normal | `HIGH_TRANSACTION_VELOCITY` | `HIGH_TRANSACTION_VELOCITY` | `EASY_TO_MEDIUM` | Medium |
| Unusual transaction amount | Yes | account, wallet | transaction, decision, label | amount, amount versus customer average, risk tier, transaction type, account age, new destination | `UNUSUAL_AMOUNT`, `NEW_DESTINATION_WALLET` | `UNUSUAL_TRANSACTION_AMOUNT` | `MEDIUM` | Medium |
| Structuring | Yes | account, wallet | transaction, decision, label | rolling counts, rolling sums, repeated destination usage, velocity versus normal | `HIGH_TRANSACTION_VELOCITY`, `UNUSUAL_AMOUNT` | `STRUCTURING` | `HARD` | High |
| Mule-account activity | Yes | accounts, devices, wallet | transaction, decision, label | destination-wallet connectivity, transaction count, wallet risk, recent label history | `NEW_DESTINATION_WALLET` | `MULE_ACCOUNT_ACTIVITY` | `HARD` | High |
| Shared suspicious device | Yes | accounts, device | authentication, transaction, decision, label | account count per device, new device, failed logins, country/device changes | `SUSPICIOUS_SHARED_DEVICE`, `NEW_DEVICE`, `FAILED_LOGIN_ACTIVITY` | `SHARED_SUSPICIOUS_DEVICE` | `HARD` | High |
| High-volatility unusual withdrawal | No | account, wallet, market | market, transaction, decision, label | market volatility, price change, trade volume, unusual amount, new destination | `HIGH_MARKET_VOLATILITY`, `UNUSUAL_AMOUNT`, `NEW_DESTINATION_WALLET` | `HIGH_VOLATILITY_UNUSUAL_WITHDRAWAL` | `HARD` | High |

## 6. Detailed Account-Takeover Design

Purpose: represent an attacker gaining access to an existing account and performing a suspicious withdrawal.

Required sequence:

1. Multiple failed login attempts.
2. Login attempt from a new or untrusted device.
3. Country different from the account home country or recent normal activity.
4. Optional password reset.
5. Successful authentication or bypass simulation.
6. Large withdrawal relative to `normal_transaction_amount_usd`.
7. New external destination wallet.
8. Delayed fraud investigation label.

Expected features:

- `failed_logins_last_10_minutes`
- `mfa_failures_last_10_minutes`
- `is_new_device`
- `is_new_country`
- `password_reset_before_transaction`
- `amount_vs_customer_average`
- `destination_wallet_is_new`
- `time_since_previous_transaction`

The suspicious amount is configured as a multiplier of the account's normal amount. The scenario allows controlled variation: not every case needs password reset, the device signal can be new or untrusted, and country comparison may use home country or recent activity.

## 7. Detailed High-Velocity Design

Purpose: represent an account performing an unusually large number of transactions or withdrawals in a short period.

Required sequence:

1. Normal historical transactions establish `normal_transaction_frequency_per_day`.
2. Several valid transactions occur within a short event-time window.
3. The burst count significantly exceeds the account profile.
4. The final transaction is scored using rolling count and sum features.
5. A delayed fraud label identifies the final transaction in the burst.

Recommended label policy: label only the final transaction in the burst. This keeps the target definition consistent and avoids inflating fraud counts by labelling every transaction in a burst.

Expected features:

- `transaction_count_last_1_minute`
- `transaction_count_last_5_minutes`
- `transaction_count_last_1_hour`
- `withdrawal_sum_last_5_minutes`
- `withdrawal_sum_last_1_hour`
- `transaction_velocity_vs_normal`
- `time_since_previous_transaction`

## 8. Detailed Unusual-Amount Design

Purpose: represent a transaction significantly larger than an account's normal behavioural amount.

The suspicious amount is calculated as:

```text
current transaction amount compared with normal_transaction_amount_usd
```

The design supports different customer profiles and must not use one fixed global amount.

Expected features:

- `transaction_amount_usd`
- `amount_vs_customer_average`
- `customer_risk_tier`
- `transaction_type`
- `account_age_days`
- `destination_wallet_is_new`

Legitimate high-value transactions are expected false-positive controls, especially for high-value customers or long-standing customers moving funds to known wallets.

## 9. Detailed Structuring Design

Purpose: represent a larger suspicious amount divided into several smaller transactions to avoid simple amount-based detection.

Required sequence:

1. One account has established normal amount and frequency profiles.
2. Several transactions occur close together.
3. Individual amounts may appear normal.
4. The combined rolling amount is unusually high.
5. The same destination wallet may be reused.
6. Delayed labels identify all completed transactions in the structured sequence.

The planned combined fraud amount is generator-control metadata only. It must not appear inside customer-transaction events.

Expected features:

- `transaction_count_last_5_minutes`
- `withdrawal_sum_last_5_minutes`
- `transaction_count_last_1_hour`
- `withdrawal_sum_last_1_hour`
- `repeated_destination_wallet_count`
- `transaction_velocity_vs_normal`

Individual transactions may look legitimate while the rolling sequence is suspicious.

## 10. Detailed Mule-Account Design

Purpose: represent multiple apparently unrelated accounts sending funds to the same external destination wallet.

Required sequence:

1. Multiple unrelated accounts exist.
2. Separate devices are used where possible.
3. One external destination wallet receives funds from those accounts.
4. Transactions occur within a defined event-time horizon.
5. Destination wallet connectivity becomes suspicious over time.
6. Delayed labels apply to transactions after the second distinct account connects to the wallet.

Expected features:

- `accounts_connected_to_destination_wallet`
- `destination_wallet_transaction_count`
- `destination_wallet_risk_score`
- `unique_destination_wallets_last_1_hour`
- `transaction_count_last_1_hour`
- `wallet_recent_fraud_count`

`wallet_recent_fraud_count` may use only labels confirmed before the prediction timestamp. A wallet-specific reason code is not present in the Phase 2 decision schema; adding one later requires an approved schema-version change.

## 11. Detailed Shared-Device Design

Purpose: represent one device being used across several unrelated customer accounts.

Required sequence:

1. One `device_id` appears in authentication attempts for multiple accounts.
2. Accounts are not normally connected.
3. Login timing or geography is suspicious.
4. One or more subsequent transactions occur.
5. Delayed labels identify documented suspicious transactions.

This remains compatible with the Phase 2 device schema: a device master record may have one `primary_account_id`, while authentication events can link the device to multiple accounts.

Expected features:

- `accounts_connected_to_device`
- `device_account_count_recent`
- `is_new_device`
- `failed_logins_last_10_minutes`
- `country_change_flag`
- `device_change_flag`

Shared family or corporate devices are important legitimate false-positive controls.

## 12. Optional Volatility Scenario

Purpose: represent an unusual withdrawal occurring during unusually high BTC or ETH market volatility.

Status: disabled by default.

Required sequence:

1. Prior market events establish elevated volatility, price change, or trade volume.
2. A withdrawal is unusual relative to the account profile.
3. A new destination wallet may appear.
4. Delayed investigation label arrives after prediction.

Expected features:

- `market_volatility_last_5_minutes`
- `price_change_last_5_minutes`
- `market_trade_volume_last_5_minutes`
- `amount_vs_customer_average`
- `destination_wallet_is_new`
- `transaction_amount_adjusted_for_market_context`

Volatility alone must never make a transaction fraudulent.

## 13. Normal Control Cases

Every scenario must include non-fraud controls:

- Legitimate travel causing a new country.
- Customer purchasing and enrolling a new phone.
- Genuine large transaction from a high-value customer.
- Payroll, merchant, or operational activity causing higher velocity.
- Corporate wallet shared by authorized accounts.
- Family or corporate device used by multiple approved users.
- High market volatility with otherwise normal transaction behaviour.

Normal controls should later receive delayed cleared labels:

```text
investigation_status = CLEARED
is_fraud = false
fraud_type = null
```

## 14. Event-Time Rules

Each scenario config defines:

- Prerequisite history period.
- Event-time ordering.
- Spacing between suspicious events.
- Maximum scenario duration.
- Investigation-label delay range.
- Whether late or out-of-order variants will be added later.
- Whether the scenario can overlap with another scenario.

These are simulation parameters for portfolio data, not production fraud rules.

## 15. Delayed-Label Rules

Labels use:

```text
label_source = SIMULATED_INVESTIGATION
```

Confirmed fraud uses:

```text
investigation_status = CONFIRMED_FRAUD
is_fraud = true
```

Labels are emitted as separate `fraud_label` events. Label timestamps must occur after the related transaction event timestamp, and `event_timestamp` must equal `label_timestamp` for the label event.

## 16. Feature Expectations

Features are design candidates only. Definitions will be finalized in Phase 8.

Feature sources must be one of:

- Current transaction.
- Historical account profile.
- Prior authentication events.
- Prior transaction events.
- Prior wallet activity.
- Prior market events.
- Prior confirmed labels.

Only events with timestamps earlier than or equal to the prediction point may be used.

## 17. Target-Leakage Prevention

Do not use these as model inputs:

- `is_fraud`
- `fraud_type`
- `scenario_id`
- `investigation_status`
- Label status.
- Generator-internal flags.

Generator-internal scenario metadata may later be stored only in a separate audit output for testing generation quality. It must never appear in model-facing transaction records.

## 18. Scenario Overlap Policy

Default policy: avoid applying more than one enabled fraud scenario to the same transaction.

The optional market-context scenario is disabled by default and may overlap only after explicit approval in a later phase. Account-level overlap across distant time periods can be introduced later if it does not make labels ambiguous.

## 19. Scenario Difficulty

- `EASY_TO_MEDIUM`: high transaction velocity.
- `MEDIUM`: account takeover, unusual transaction amount.
- `HARD`: structuring, mule-account activity, shared suspicious device, optional high-volatility unusual withdrawal.

Difficulty reflects how much multi-event context a model needs, not implementation status.

## 20. False-Positive Risks

Key false-positive risks:

- New country may be legitimate travel.
- New device may be a phone replacement.
- Large amount may be normal for high-value customers.
- High velocity may be payroll, merchant, or operational activity.
- Shared wallet may be a corporate or exchange destination.
- Shared device may be family or corporate usage.
- Market volatility may coincide with legitimate portfolio management.

## 21. Validation Requirements

Phase 3 validation tests must confirm:

- All scenario YAML files exist and parse.
- Required keys are present.
- Required six scenarios are enabled.
- Optional volatility scenario is disabled.
- Fraud types match the fraud-label schema.
- Reason codes match the fraud-decision schema.
- Label rules, timing rules, feature signals, and control cases exist.
- Scenario configs do not define final model decision cutoffs.
- Model-facing features do not include leakage fields.
- Referenced paths in the registry exist.

## 22. Phase 3 Approval Status

Status: Phase 3 fraud scenario catalogue implemented and awaiting review.
