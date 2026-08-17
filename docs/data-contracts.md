# Data Contracts

## 1. Purpose

This Phase 2 document defines the approved data contracts for the Real-Time Crypto Fraud Detection Platform. These contracts describe record structure, field meanings, validation rules, keys, relationships, and compatibility rules for later historical generators, live generators, Coinbase collectors, Parquet outputs, Databricks Bronze and Silver processing, Azure Event Hubs, Spark Structured Streaming, feature engineering, model scoring, and delayed fraud-label processing.

This phase defines structure and validation only. It does not create datasets, Event Hubs, Databricks objects, Coinbase connections, fraud scenarios, features, models, or dashboards.

Machine-readable JSON Schemas are stored in `config/schemas/` using JSON Schema Draft 2020-12.

## 2. Contract Design Principles

- Use one explicit contract per entity or event type.
- Keep static master-data records separate from live event records.
- Reuse common definitions where practical, without hiding business meaning behind excessive abstraction.
- Keep fraud labels separate from customer transactions to prevent target leakage.
- Preserve one structure for historical and live records wherever the business meaning is the same.
- Treat cross-record and cross-field checks that JSON Schema cannot safely enforce as application-level validation rules.
- Use deterministic, auditable identifiers and event timestamps to support replay, deduplication, and model audit.

## 3. Schema-Version Policy

The initial schema version is `1.0`.

Schema versions use semantic versioning:

- `1.0`: initial approved contract.
- `1.x`: backward-compatible optional additions.
- `2.0`: breaking changes, including renamed fields, removed fields, changed meanings, changed requiredness, narrowed enum values, or changed identifier semantics.

Each record contains `schema_version`. Phase 2 schemas restrict this value to `1.0`. Later schema changes must keep older consumers in mind and must update both the JSON Schema file and this document.

## 4. Naming Conventions

All field names use `snake_case`. Entity names are singular and event names are descriptive.

Preferred examples:

- `account_id`
- `transaction_id`
- `event_timestamp`
- `market_price_usd`
- `processing_latency_ms`

Avoid ambiguous names such as `id`, `time`, `amount`, `type`, or `status` unless the surrounding meaning is completely clear. For this reason the contracts use names such as `transaction_type`, `transaction_status`, `investigation_status`, and `account_status`.

## 5. Identifier Rules

Platform identifiers use UUID strings with JSON Schema `format: uuid`.

This applies to:

- `event_id`
- `account_id`
- `device_id`
- `wallet_id`
- `login_id`
- `transaction_id`

Coinbase `trade_id` and `sequence_number` are source-native values. `trade_id` remains a non-empty string, and `sequence_number` is a non-negative integer.

## 6. Timestamp Definitions

All timestamps are UTC ISO 8601 strings using JSON Schema `format: date-time`. Timestamps must include timezone information and normally end with `Z`; `+00:00` is also accepted by the schema.

Common timestamp meanings:

- `event_timestamp`: when the business event actually occurred, such as a withdrawal, login attempt, Coinbase trade, label assignment, or model prediction.
- `source_timestamp`: when the source system created, recorded, or published the event.
- `ingestion_timestamp`: when this platform received the event.

Conceptual ordering, where applicable:

```text
event_timestamp <= source_timestamp <= ingestion_timestamp
```

This ordering is an application-level validation rule. JSON Schema validates timestamp shape, not cross-field time ordering.

Special event timestamp rules:

- Market events: `event_timestamp = trade_timestamp`.
- Fraud-label events: `event_timestamp = label_timestamp`.
- Fraud decisions: `event_timestamp = prediction_timestamp`.

## 7. Common Event Fields

Common event fields apply only to live or historical event records. They are not added to account, device, or wallet master records.

| Field name | JSON type | Target Delta type | Required | Nullable | Allowed values | Description | Example |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `event_id` | string | STRING | Required | Non-null | UUID | Globally unique event identifier and main event deduplication key. | `66666666-6666-4666-8666-666666666666` |
| `event_type` | string | STRING | Required | Non-null | `authentication_event`, `customer_transaction`, `market_event`, `fraud_label`, `fraud_decision` | Contract-specific event type. | `customer_transaction` |
| `schema_version` | string | STRING | Required | Non-null | `1.0` | Semantic schema version. | `1.0` |
| `source` | string | STRING | Required | Non-null | See source enums below | System that produced the event. | `historical_customer_generator` |
| `event_timestamp` | string | TIMESTAMP | Required | Non-null | UTC date-time | Business event time. | `2026-01-15T12:00:00Z` |
| `source_timestamp` | string | TIMESTAMP | Required | Non-null | UTC date-time | Source-system record or publish time. | `2026-01-15T12:00:01Z` |
| `ingestion_timestamp` | string | TIMESTAMP | Required | Non-null | UTC date-time | Platform receipt time. | `2026-01-15T12:00:02Z` |

## 8. Account Contract

Schema file: `config/schemas/account.schema.json`

Account records are master data, not live events. The primary key is `account_id`.

| Field name | JSON type | Target Delta type | Required | Nullable | Allowed values | Description | Example |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `account_id` | string | STRING | Required | Non-null | UUID | Account primary key. | `11111111-1111-4111-8111-111111111111` |
| `schema_version` | string | STRING | Required | Non-null | `1.0` | Contract version. | `1.0` |
| `created_at` | string | TIMESTAMP | Required | Non-null | UTC date-time | Account creation time. | `2026-01-01T00:00:00Z` |
| `updated_at` | string | TIMESTAMP | Required | Non-null | UTC date-time | Most recent profile update time. | `2026-01-10T00:00:00Z` |
| `home_country` | string | STRING | Required | Non-null | Two uppercase letters | Customer home country. | `US` |
| `kyc_level` | string | STRING | Required | Non-null | `BASIC`, `STANDARD`, `ENHANCED` | Know-your-customer verification level. | `STANDARD` |
| `customer_risk_tier` | string | STRING | Required | Non-null | `LOW`, `MEDIUM`, `HIGH` | Baseline customer risk segment. | `LOW` |
| `normal_transaction_amount_usd` | number | DECIMAL(20, 8) | Required | Non-null | `>= 0` | Typical transaction amount in USD. | `250.0` |
| `normal_transaction_frequency_per_day` | number | DOUBLE | Required | Non-null | `>= 0` | Typical daily transaction frequency. | `2.0` |
| `preferred_assets` | array | ARRAY<STRING> | Required | Non-null | Unique `BTC`, `ETH` values | Assets normally used by the customer. | `["BTC", "ETH"]` |
| `account_status` | string | STRING | Required | Non-null | `ACTIVE`, `SUSPENDED`, `CLOSED` | Current account lifecycle status. | `ACTIVE` |

Application-level rule: `updated_at` must not be earlier than `created_at`.

## 9. Device Contract

Schema file: `config/schemas/device.schema.json`

Device records are master data. The primary key is `device_id`.

| Field name | JSON type | Target Delta type | Required | Nullable | Allowed values | Description | Example |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `device_id` | string | STRING | Required | Non-null | UUID | Device primary key. | `22222222-2222-4222-8222-222222222222` |
| `schema_version` | string | STRING | Required | Non-null | `1.0` | Contract version. | `1.0` |
| `first_seen_at` | string | TIMESTAMP | Required | Non-null | UTC date-time | First time the platform observed the device. | `2026-01-01T00:05:00Z` |
| `last_seen_at` | string | TIMESTAMP | Required | Non-null | UTC date-time | Most recent time the platform observed the device. | `2026-01-15T09:00:00Z` |
| `device_type` | string | STRING | Required | Non-null | `MOBILE`, `DESKTOP`, `TABLET` | Device category. | `MOBILE` |
| `operating_system` | string | STRING | Required | Non-null | `ANDROID`, `IOS`, `WINDOWS`, `MACOS`, `LINUX`, `OTHER` | Device operating system. | `IOS` |
| `is_trusted` | boolean | BOOLEAN | Required | Non-null | `true`, `false` | Whether the device is trusted for its primary account. | `true` |
| `device_country` | string | STRING | Required | Non-null | Two uppercase letters | Country associated with the device profile. | `US` |
| `primary_account_id` | string or null | STRING | Required | Nullable | UUID or null | Main account associated with the device when known. | `11111111-1111-4111-8111-111111111111` |

Application-level rules:

- `last_seen_at` must not be earlier than `first_seen_at`.
- `primary_account_id` may be null for unknown or suspicious devices.
- A device may appear in authentication events for more than one account. This supports the future `SHARED_SUSPICIOUS_DEVICE` scenario.

Foreign key when non-null: `primary_account_id -> account.account_id`.

## 10. Wallet Contract

Schema file: `config/schemas/wallet.schema.json`

Wallet records are master data. The primary key is `wallet_id`.

| Field name | JSON type | Target Delta type | Required | Nullable | Allowed values | Description | Example |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `wallet_id` | string | STRING | Required | Non-null | UUID | Wallet primary key. | `33333333-3333-4333-8333-333333333333` |
| `schema_version` | string | STRING | Required | Non-null | `1.0` | Contract version. | `1.0` |
| `owner_account_id` | string or null | STRING | Required | Nullable | UUID or null | Owning customer account when known. | `11111111-1111-4111-8111-111111111111` |
| `wallet_type` | string | STRING | Required | Non-null | `CUSTOMER`, `EXTERNAL` | Whether the wallet belongs to a customer account or an external destination. | `CUSTOMER` |
| `first_seen_at` | string | TIMESTAMP | Required | Non-null | UTC date-time | First time the platform observed the wallet. | `2026-01-01T00:10:00Z` |
| `risk_level` | string | STRING | Required | Non-null | `LOW`, `MEDIUM`, `HIGH`, `UNKNOWN` | Wallet risk classification. | `LOW` |
| `is_known_destination` | boolean | BOOLEAN | Required | Non-null | `true`, `false` | Whether this wallet has been seen as a destination before. | `true` |
| `supported_assets` | array | ARRAY<STRING> | Required | Non-null | Unique `BTC`, `ETH` values | Assets supported by the wallet. | `["BTC", "ETH"]` |

Application-level rules:

- Customer wallets normally have an `owner_account_id`.
- External destination wallets may have `owner_account_id = null`.

Foreign key when non-null: `owner_account_id -> account.account_id`.

## 11. Authentication-Event Contract

Schema file: `config/schemas/authentication-event.schema.json`

Authentication events include all common event fields. The primary key is `event_id`; the business event identifier is `login_id`.

| Field name | JSON type | Target Delta type | Required | Nullable | Allowed values | Description | Example |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `event_id` | string | STRING | Required | Non-null | UUID | Event primary key. | `44444444-4444-4444-8444-444444444444` |
| `event_type` | string | STRING | Required | Non-null | `authentication_event` | Event contract discriminator. | `authentication_event` |
| `schema_version` | string | STRING | Required | Non-null | `1.0` | Contract version. | `1.0` |
| `source` | string | STRING | Required | Non-null | `authentication_generator` | Approved authentication event producer. | `authentication_generator` |
| `event_timestamp` | string | TIMESTAMP | Required | Non-null | UTC date-time | Login attempt time. | `2026-01-15T11:58:00Z` |
| `source_timestamp` | string | TIMESTAMP | Required | Non-null | UTC date-time | Generator record time. | `2026-01-15T11:58:01Z` |
| `ingestion_timestamp` | string | TIMESTAMP | Required | Non-null | UTC date-time | Platform receipt time. | `2026-01-15T11:58:02Z` |
| `login_id` | string | STRING | Required | Non-null | UUID | Business identifier for the login attempt. | `55555555-5555-4555-8555-555555555555` |
| `account_id` | string | STRING | Required | Non-null | UUID | Account used in the login attempt. | `11111111-1111-4111-8111-111111111111` |
| `device_id` | string | STRING | Required | Non-null | UUID | Device used in the login attempt. | `22222222-2222-4222-8222-222222222222` |
| `country` | string | STRING | Required | Non-null | Two uppercase letters | Login country. | `US` |
| `ip_address` | string | STRING | Required | Non-null | IPv4 or IPv6 | Source IP address. | `203.0.113.10` |
| `login_success` | boolean | BOOLEAN | Required | Non-null | `true`, `false` | Whether login completed successfully. | `true` |
| `mfa_success` | boolean | BOOLEAN | Required | Non-null | `true`, `false` | Whether MFA completed successfully. | `true` |
| `password_reset_flag` | boolean | BOOLEAN | Required | Non-null | `true`, `false` | Whether a password reset was involved. | `false` |
| `failure_reason` | string or null | STRING | Required | Nullable | `INVALID_PASSWORD`, `MFA_FAILED`, `ACCOUNT_LOCKED`, `DEVICE_BLOCKED`, `OTHER`, null | Reason for failed authentication when available. | null |

Conditional business rules:

- If `login_success = true`, `failure_reason` should normally be null.
- If `mfa_success = false`, the event may still represent a failed authentication.

Foreign keys:

- `account_id -> account.account_id`
- `device_id -> device.device_id`

## 12. Customer-Transaction Contract

Schema file: `config/schemas/customer-transaction.schema.json`

Customer transactions include all common event fields. The primary key is `event_id`; the business identifier is `transaction_id`.

Fraud labels must not be stored in this contract. The following fields are forbidden: `is_fraud`, `fraud_type`, `fraud_label`, `confirmed_fraud`.

| Field name | JSON type | Target Delta type | Required | Nullable | Allowed values | Description | Example |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `event_id` | string | STRING | Required | Non-null | UUID | Event primary key. | `66666666-6666-4666-8666-666666666666` |
| `event_type` | string | STRING | Required | Non-null | `customer_transaction` | Event contract discriminator. | `customer_transaction` |
| `schema_version` | string | STRING | Required | Non-null | `1.0` | Contract version. | `1.0` |
| `source` | string | STRING | Required | Non-null | `historical_customer_generator`, `realtime_customer_generator` | Approved transaction producer. | `historical_customer_generator` |
| `event_timestamp` | string | TIMESTAMP | Required | Non-null | UTC date-time | Transaction business time. | `2026-01-15T12:00:00Z` |
| `source_timestamp` | string | TIMESTAMP | Required | Non-null | UTC date-time | Generator record time. | `2026-01-15T12:00:01Z` |
| `ingestion_timestamp` | string | TIMESTAMP | Required | Non-null | UTC date-time | Platform receipt time. | `2026-01-15T12:00:02Z` |
| `transaction_id` | string | STRING | Required | Non-null | UUID | Business identifier for the transaction. | `77777777-7777-4777-8777-777777777777` |
| `account_id` | string | STRING | Required | Non-null | UUID | Account that initiated the transaction. | `11111111-1111-4111-8111-111111111111` |
| `asset` | string | STRING | Required | Non-null | `BTC`, `ETH` | Crypto asset transacted. | `BTC` |
| `crypto_quantity` | number | DECIMAL(20, 8) | Required | Non-null | `> 0` | Crypto quantity. | `0.005` |
| `transaction_type` | string | STRING | Required | Non-null | `DEPOSIT`, `WITHDRAWAL`, `TRANSFER` | Transaction direction or movement type. | `WITHDRAWAL` |
| `source_wallet_id` | string or null | STRING | Required | Nullable | UUID or null | Source wallet when applicable. | `33333333-3333-4333-8333-333333333333` |
| `destination_wallet_id` | string or null | STRING | Required | Nullable | UUID or null | Destination wallet when applicable. | `88888888-8888-4888-8888-888888888888` |
| `device_id` | string | STRING | Required | Non-null | UUID | Device used for the transaction. | `22222222-2222-4222-8222-222222222222` |
| `country` | string | STRING | Required | Non-null | Two uppercase letters | Transaction country. | `US` |
| `market_price_usd` | number | DECIMAL(20, 8) | Required | Non-null | `> 0` | Market price used for valuation. | `40000.0` |
| `transaction_amount_usd` | number | DECIMAL(20, 8) | Required | Non-null | `> 0` | USD value of the transaction. | `200.0` |
| `transaction_status` | string | STRING | Required | Non-null | `PENDING`, `COMPLETED`, `FAILED` | Processing state of the transaction. | `COMPLETED` |

JSON Schema uses JSON `number` for quantities and money. Databricks tables should later use fixed-precision decimal types such as `DECIMAL(20, 8)` for these fields.

Conditional rules:

- `DEPOSIT` must have a destination customer wallet.
- `WITHDRAWAL` must have a source customer wallet and a destination wallet.
- `TRANSFER` must have both source and destination wallets.
- Source and destination wallets must not be identical.
- `transaction_amount_usd` should equal `crypto_quantity * market_price_usd` with a small rounding tolerance. Recommended tolerance: absolute difference less than or equal to `0.01` USD.
- Failed transactions remain in the dataset. Later feature logic must define whether failed transactions count in behavioural windows.

Foreign keys:

- `account_id -> account.account_id`
- `device_id -> device.device_id`
- `source_wallet_id -> wallet.wallet_id`
- `destination_wallet_id -> wallet.wallet_id`

## 13. Market-Event Contract

Schema file: `config/schemas/market-event.schema.json`

Market events include all common event fields. The primary key is `event_id`; the source business identifier is `product_id + trade_id`.

Market events are real Coinbase public market enrichment data. They are not customer transactions and do not contain customer fraud labels.

| Field name | JSON type | Target Delta type | Required | Nullable | Allowed values | Description | Example |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `event_id` | string | STRING | Required | Non-null | UUID | Event primary key. | `99999999-9999-4999-8999-999999999999` |
| `event_type` | string | STRING | Required | Non-null | `market_event` | Event contract discriminator. | `market_event` |
| `schema_version` | string | STRING | Required | Non-null | `1.0` | Contract version. | `1.0` |
| `source` | string | STRING | Required | Non-null | `coinbase_rest_api`, `coinbase_websocket` | Coinbase public market source. | `coinbase_websocket` |
| `event_timestamp` | string | TIMESTAMP | Required | Non-null | UTC date-time | Coinbase trade-level timestamp. | `2026-01-15T12:00:00Z` |
| `source_timestamp` | string | TIMESTAMP | Required | Non-null | UTC date-time | Coinbase source or message timestamp. | `2026-01-15T12:00:00Z` |
| `ingestion_timestamp` | string | TIMESTAMP | Required | Non-null | UTC date-time | Platform receipt time. | `2026-01-15T12:00:01Z` |
| `product_id` | string | STRING | Required | Non-null | `BTC-USD`, `ETH-USD` | Coinbase product identifier. | `BTC-USD` |
| `trade_id` | string | STRING | Required | Non-null | Non-empty string | Coinbase source-native trade identifier. | `coinbase-trade-1001` |
| `price_usd` | number | DECIMAL(20, 8) | Required | Non-null | `> 0` | Trade price in USD. | `40000.0` |
| `size` | number | DECIMAL(20, 8) | Required | Non-null | `> 0` | Coinbase trade size. | `0.25` |
| `side` | string | STRING | Required | Non-null | `BUY`, `SELL` | Coinbase trade side. | `BUY` |
| `trade_timestamp` | string | TIMESTAMP | Required | Non-null | UTC date-time | Coinbase trade time. | `2026-01-15T12:00:00Z` |
| `message_timestamp` | string | TIMESTAMP | Required | Non-null | UTC date-time | Coinbase message time. | `2026-01-15T12:00:00Z` |
| `sequence_number` | integer | BIGINT | Required | Non-null | `>= 0` | Coinbase source-native sequence value. | `1001` |

Application-level rules:

- `trade_timestamp` must equal `event_timestamp`.
- `message_timestamp` must not be earlier than `trade_timestamp` under normal conditions.
- Coinbase WebSocket sequence gaps must be detected by the collector in a later phase and handled as application-level quality events or quarantine candidates.

## 14. Fraud-Label Contract

Schema file: `config/schemas/fraud-label.schema.json`

Fraud labels include all common event fields. The primary key is `event_id`; the foreign key is `transaction_id -> customer_transaction.transaction_id`.

Fraud labels arrive after model prediction and represent delayed investigation outcomes. They are the supervised learning label source and must not be embedded in the customer-transaction contract.

| Field name | JSON type | Target Delta type | Required | Nullable | Allowed values | Description | Example |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `event_id` | string | STRING | Required | Non-null | UUID | Event primary key. | `aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa` |
| `event_type` | string | STRING | Required | Non-null | `fraud_label` | Event contract discriminator. | `fraud_label` |
| `schema_version` | string | STRING | Required | Non-null | `1.0` | Contract version. | `1.0` |
| `source` | string | STRING | Required | Non-null | `simulated_investigation` | Source that produced the label event. | `simulated_investigation` |
| `event_timestamp` | string | TIMESTAMP | Required | Non-null | UTC date-time | Label event time. | `2026-01-16T12:00:00Z` |
| `source_timestamp` | string | TIMESTAMP | Required | Non-null | UTC date-time | Investigation record time. | `2026-01-16T12:00:01Z` |
| `ingestion_timestamp` | string | TIMESTAMP | Required | Non-null | UTC date-time | Platform receipt time. | `2026-01-16T12:00:02Z` |
| `transaction_id` | string | STRING | Required | Non-null | UUID | Related customer transaction business identifier. | `77777777-7777-4777-8777-777777777777` |
| `is_fraud` | boolean | BOOLEAN | Required | Non-null | `true`, `false` | Final fraud outcome flag. | `true` |
| `fraud_type` | string or null | STRING | Required | Nullable | Fraud scenario enum or null | Fraud category when fraud is confirmed. | `ACCOUNT_TAKEOVER` |
| `label_timestamp` | string | TIMESTAMP | Required | Non-null | UTC date-time | Time the delayed label was assigned. | `2026-01-16T12:00:00Z` |
| `label_source` | string | STRING | Required | Non-null | `SIMULATED_INVESTIGATION` | Investigation label source. | `SIMULATED_INVESTIGATION` |
| `investigation_status` | string | STRING | Required | Non-null | `CONFIRMED_FRAUD`, `CLEARED`, `INCONCLUSIVE` | Investigation outcome state. | `CONFIRMED_FRAUD` |

Conditional rules:

- If `is_fraud = true`, `fraud_type` must not be null.
- If `is_fraud = false`, `fraud_type` should be null.
- If `investigation_status = CONFIRMED_FRAUD`, `is_fraud` must be true.
- If `investigation_status = CLEARED`, `is_fraud` must be false.
- `label_timestamp` must occur after the related transaction's `event_timestamp`.
- `event_timestamp` must equal `label_timestamp`.

## 14A. Historical Market-Candle Contract

Schema file: `config/schemas/historical-market-candle.schema.json`

Historical market candles are raw source-specific Coinbase Exchange REST candle records. They are real public market data only. They are not customer transactions, fraud data, withdrawal data, account data, or confirmed fraud labels.

This contract is separate from `market-event.schema.json`, which remains the future live WebSocket trade-level contract. Historical candles do not provide trade IDs, buy/sell side, message sequence numbers, or individual trade records, so those fields must not be fabricated.

| Field name | JSON type | Target Delta type | Required | Nullable | Allowed values | Description | Example |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `schema_version` | string | STRING | Required | Non-null | `1.0` | Initial contract version. | `1.0` |
| `source` | string | STRING | Required | Non-null | `coinbase_exchange_rest_api` | Source name for historical Coinbase candles. | `coinbase_exchange_rest_api` |
| `product_id` | string | STRING | Required | Non-null | `BTC-USD`, `ETH-USD` | Coinbase product identifier. | `BTC-USD` |
| `candle_start_timestamp` | string | TIMESTAMP | Required | Non-null | UTC date-time | Start of one-minute interval. | `2026-01-01T00:00:00Z` |
| `candle_end_timestamp` | string | TIMESTAMP | Required | Non-null | UTC date-time | End of one-minute interval. | `2026-01-01T00:01:00Z` |
| `granularity_seconds` | integer | BIGINT | Required | Non-null | `60` | Candle interval length. | `60` |
| `open_price_usd` | number | DECIMAL(20, 8) | Required | Non-null | `> 0` | Opening price. | `42500.12` |
| `high_price_usd` | number | DECIMAL(20, 8) | Required | Non-null | `> 0` | Highest price. | `42510.5` |
| `low_price_usd` | number | DECIMAL(20, 8) | Required | Non-null | `> 0` | Lowest price. | `42490.25` |
| `close_price_usd` | number | DECIMAL(20, 8) | Required | Non-null | `> 0` | Closing price. | `42505.75` |
| `volume` | number | DECIMAL(20, 8) | Required | Non-null | `>= 0` | Coinbase base-asset trading volume. | `12.3456789` |
| `retrieved_at` | string | TIMESTAMP | Required | Non-null | UTC date-time | Time our collector received the record. | `2026-07-01T00:00:00Z` |

Natural business key:

```text
product_id + candle_start_timestamp
```

Application-level rules:

- `candle_end_timestamp` must equal `candle_start_timestamp + 60 seconds`.
- `high_price_usd` must not be lower than open, close, or low.
- `low_price_usd` must not be higher than open, close, or high.
- Duplicate candles must be removed by `product_id + candle_start_timestamp`.
- Missing Coinbase intervals must be preserved and flagged; fake candles must not be created silently.
- Future customer-transaction enrichment must never use future candles.

## 15. Fraud-Decision Contract

Schema file: `config/schemas/fraud-decision.schema.json`

Fraud decisions include all common event fields. The primary key is `event_id`; the foreign key is `transaction_id -> customer_transaction.transaction_id`.

Fraud decisions occur before final fraud labels. They represent model scoring output, not investigation truth.

| Field name | JSON type | Target Delta type | Required | Nullable | Allowed values | Description | Example |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `event_id` | string | STRING | Required | Non-null | UUID | Event primary key. | `bbbbbbbb-bbbb-4bbb-8bbb-bbbbbbbbbbbb` |
| `event_type` | string | STRING | Required | Non-null | `fraud_decision` | Event contract discriminator. | `fraud_decision` |
| `schema_version` | string | STRING | Required | Non-null | `1.0` | Contract version. | `1.0` |
| `source` | string | STRING | Required | Non-null | `realtime_scoring_pipeline` | Real-time scoring source. | `realtime_scoring_pipeline` |
| `event_timestamp` | string | TIMESTAMP | Required | Non-null | UTC date-time | Prediction business time. | `2026-01-15T12:00:03Z` |
| `source_timestamp` | string | TIMESTAMP | Required | Non-null | UTC date-time | Scoring source time. | `2026-01-15T12:00:03Z` |
| `ingestion_timestamp` | string | TIMESTAMP | Required | Non-null | UTC date-time | Platform receipt time. | `2026-01-15T12:00:04Z` |
| `transaction_id` | string | STRING | Required | Non-null | UUID | Related customer transaction business identifier. | `77777777-7777-4777-8777-777777777777` |
| `risk_score` | number | DOUBLE | Required | Non-null | `0 <= value <= 1` | Model fraud risk score. | `0.82` |
| `decision` | string | STRING | Required | Non-null | `ALLOW`, `REVIEW`, `BLOCK` | Scoring decision. | `REVIEW` |
| `reason_codes` | array | ARRAY<STRING> | Required | Non-null | Unique reason-code enum values | Reasons contributing to the decision. | `["NEW_DESTINATION_WALLET"]` |
| `model_name` | string | STRING | Required | Non-null | Non-empty string | Model name used for scoring. | `xgboost_fraud_model` |
| `model_version` | string | STRING | Required | Non-null | Non-empty string | Model version used for scoring. | `1` |
| `prediction_timestamp` | string | TIMESTAMP | Required | Non-null | UTC date-time | Time the prediction was produced. | `2026-01-15T12:00:03Z` |
| `processing_latency_ms` | number | DOUBLE | Required | Non-null | `>= 0` | End-to-end scoring latency in milliseconds. | `42.5` |
| `threshold_policy_version` | string | STRING | Required | Non-null | Non-empty string | Identifier for the threshold policy applied by the scoring service. | `policy-1.0` |

Rules:

- Do not define arbitrary numeric decision thresholds in this contract.
- Thresholds will be selected after model validation in Phase 9.
- `event_timestamp` must equal `prediction_timestamp`.

## 16. Primary and Foreign Keys

| Contract | Primary key | Business identifier | Foreign keys |
| --- | --- | --- | --- |
| `account` | `account_id` | `account_id` | None |
| `device` | `device_id` | `device_id` | `primary_account_id -> account.account_id` when non-null |
| `wallet` | `wallet_id` | `wallet_id` | `owner_account_id -> account.account_id` when non-null |
| `authentication_event` | `event_id` | `login_id` | `account_id -> account.account_id`; `device_id -> device.device_id` |
| `customer_transaction` | `event_id` | `transaction_id` | `account_id -> account.account_id`; `device_id -> device.device_id`; `source_wallet_id -> wallet.wallet_id`; `destination_wallet_id -> wallet.wallet_id` |
| `market_event` | `event_id` | `product_id + trade_id` | None |
| `historical_market_candle` | `product_id + candle_start_timestamp` | `product_id + candle_start_timestamp` | None |
| `fraud_label` | `event_id` | `transaction_id` | `transaction_id -> customer_transaction.transaction_id` |
| `fraud_decision` | `event_id` | `transaction_id` | `transaction_id -> customer_transaction.transaction_id` |

Relationship summary:

```text
account
|-- devices through authentication-event usage
|-- customer wallets
|-- authentication events
`-- customer transactions

customer transaction
|-- account
|-- device
|-- source wallet
|-- destination wallet
|-- market enrichment
|-- fraud decision
`-- delayed fraud label
```

Clarifications:

- A transaction has one account.
- A transaction uses one device in the current scope.
- A device may appear across multiple accounts in suspicious scenarios.
- A customer wallet has an owner account.
- An external wallet may not have an owner account.
- A fraud decision occurs before the final fraud label.
- Market events are joined by asset and event time, not by customer identity.

## 17. Allowed Enums

Event types:

- `authentication_event`
- `customer_transaction`
- `market_event`
- `fraud_label`
- `fraud_decision`

Event sources:

- `historical_customer_generator`
- `realtime_customer_generator`
- `authentication_generator`
- `coinbase_rest_api`
- `coinbase_websocket`
- `simulated_investigation`
- `realtime_scoring_pipeline`

Assets:

- `BTC`
- `ETH`

Coinbase products:

- `BTC-USD`
- `ETH-USD`

KYC levels:

- `BASIC`
- `STANDARD`
- `ENHANCED`

Customer risk tiers, wallet risk levels, and account states:

- `customer_risk_tier`: `LOW`, `MEDIUM`, `HIGH`
- `risk_level`: `LOW`, `MEDIUM`, `HIGH`, `UNKNOWN`
- `account_status`: `ACTIVE`, `SUSPENDED`, `CLOSED`

Device values:

- `device_type`: `MOBILE`, `DESKTOP`, `TABLET`
- `operating_system`: `ANDROID`, `IOS`, `WINDOWS`, `MACOS`, `LINUX`, `OTHER`

Authentication failure reasons:

- `INVALID_PASSWORD`
- `MFA_FAILED`
- `ACCOUNT_LOCKED`
- `DEVICE_BLOCKED`
- `OTHER`
- null

Transaction values:

- `transaction_type`: `DEPOSIT`, `WITHDRAWAL`, `TRANSFER`
- `transaction_status`: `PENDING`, `COMPLETED`, `FAILED`

Market sides:

- `BUY`
- `SELL`

Fraud types:

- `ACCOUNT_TAKEOVER`
- `HIGH_TRANSACTION_VELOCITY`
- `UNUSUAL_TRANSACTION_AMOUNT`
- `STRUCTURING`
- `MULE_ACCOUNT_ACTIVITY`
- `SHARED_SUSPICIOUS_DEVICE`
- `HIGH_VOLATILITY_UNUSUAL_WITHDRAWAL`
- null

Fraud-label values:

- `label_source`: `SIMULATED_INVESTIGATION`
- `investigation_status`: `CONFIRMED_FRAUD`, `CLEARED`, `INCONCLUSIVE`

Fraud-decision values:

- `decision`: `ALLOW`, `REVIEW`, `BLOCK`
- `reason_codes`: `NEW_DEVICE`, `NEW_COUNTRY`, `HIGH_TRANSACTION_VELOCITY`, `UNUSUAL_AMOUNT`, `NEW_DESTINATION_WALLET`, `FAILED_LOGIN_ACTIVITY`, `HIGH_MARKET_VOLATILITY`, `SUSPICIOUS_SHARED_DEVICE`

## 18. Cross-Field Validation Rules

These rules are either partially enforced by JSON Schema or documented for application-level validation:

- `event_timestamp <= source_timestamp <= ingestion_timestamp` where applicable.
- Account `updated_at >= created_at`.
- Device `last_seen_at >= first_seen_at`.
- Transaction `transaction_amount_usd` approximately equals `crypto_quantity * market_price_usd`, with a recommended USD tolerance of `0.01`.
- Transaction source and destination wallets must not be identical.
- Transaction wallet IDs must match the semantics of `transaction_type`.
- Fraud-label `label_timestamp` must occur after the related transaction's `event_timestamp`.
- Fraud-label `event_timestamp = label_timestamp`.
- Fraud-decision `event_timestamp = prediction_timestamp`.
- Market-event `event_timestamp = trade_timestamp`.
- Market-event `message_timestamp` must not normally be earlier than `trade_timestamp`.
- Market WebSocket sequence gaps must be detected by the collector in a later phase.

## 19. Invalid-Record Handling

Future ingestion should quarantine records when they contain:

- Missing required identifiers.
- Malformed JSON.
- Invalid timestamps.
- Unsupported assets.
- Negative or zero transaction quantities.
- Negative or zero market prices.
- Invalid enum values.
- Unknown schema versions.
- Impossible wallet relationships.
- Broken foreign-key references where reference data is expected.

Bronze tables, quarantine tables, and quarantine workflows are not implemented in Phase 2. These rules are documented for later ingestion and streaming phases.

## 20. Deduplication Rules

Use `event_id` as the main event deduplication key for event contracts.

Use business identifiers as secondary checks:

- `transaction_id`
- `login_id`
- `product_id + trade_id`

Master data should use its primary key for upsert or latest-record logic:

- `account_id`
- `device_id`
- `wallet_id`

## 21. Compatibility Rules

Backward-compatible changes may be introduced in `1.x` versions only when they are optional and do not change existing field meaning.

Breaking changes require a `2.0` schema and a migration plan. Breaking changes include:

- Removing a field.
- Renaming a field.
- Changing a field's meaning.
- Changing an optional field to required.
- Changing nullability in a way that rejects previously valid records.
- Narrowing enums in a way that rejects previously valid records.
- Changing primary-key or foreign-key semantics.

All schema changes must update JSON Schema files, human-readable documentation, fixtures, and validation tests together.

## 22. Historical and Live Consistency

Historical generators and live generators must emit records that conform to the same contracts for the same business concepts. Historical data may be written to Parquet first, while live records may later pass through Event Hubs and Spark Structured Streaming, but their field names, identifiers, timestamps, enum values, and label separation rules must stay consistent.

Historical customer transactions must not contain fraud labels. Live customer transactions must not contain fraud labels. Delayed labels must arrive through the fraud-label contract after the related transaction and model decision.

## 23. Valid Sample Records

One small valid JSON record per contract is stored under `tests/fixtures/schemas/valid/`:

- `common-event.json`
- `account.json`
- `device.json`
- `wallet.json`
- `authentication-event.json`
- `customer-transaction.json`
- `market-event.json`
- `fraud-label.json`
- `fraud-decision.json`

Example customer transaction:

```json
{
  "event_id": "66666666-6666-4666-8666-666666666666",
  "event_type": "customer_transaction",
  "schema_version": "1.0",
  "source": "historical_customer_generator",
  "event_timestamp": "2026-01-15T12:00:00Z",
  "source_timestamp": "2026-01-15T12:00:01Z",
  "ingestion_timestamp": "2026-01-15T12:00:02Z",
  "transaction_id": "77777777-7777-4777-8777-777777777777",
  "account_id": "11111111-1111-4111-8111-111111111111",
  "asset": "BTC",
  "crypto_quantity": 0.005,
  "transaction_type": "WITHDRAWAL",
  "source_wallet_id": "33333333-3333-4333-8333-333333333333",
  "destination_wallet_id": "88888888-8888-4888-8888-888888888888",
  "device_id": "22222222-2222-4222-8222-222222222222",
  "country": "US",
  "market_price_usd": 40000.0,
  "transaction_amount_usd": 200.0,
  "transaction_status": "COMPLETED"
}
```

## 24. Invalid Sample Explanations

Representative invalid records are stored under `tests/fixtures/schemas/invalid/`:

| Fixture | Expected failure |
| --- | --- |
| `customer-transaction-missing-transaction-id.json` | Missing required `transaction_id`. |
| `customer-transaction-forbidden-is-fraud.json` | Customer transaction contains forbidden `is_fraud`. |
| `customer-transaction-unsupported-asset.json` | Customer transaction uses unsupported asset. |
| `customer-transaction-negative-quantity.json` | Customer transaction has non-positive `crypto_quantity`. |
| `fraud-decision-risk-score-greater-than-one.json` | Fraud decision has `risk_score > 1`. |
| `fraud-label-fraud-null-type.json` | Fraud label has `is_fraud = true` with null `fraud_type`. |
| `market-event-invalid-product-id.json` | Market event uses unsupported Coinbase product. |
| `authentication-event-missing-common-metadata.json` | Event record is missing required common metadata. |

## 25. Phase 2 Approval Status

Status: Phase 2 data contracts and schemas implemented and awaiting review.
