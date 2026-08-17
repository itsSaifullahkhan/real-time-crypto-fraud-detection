# Historical Data Generator

## 1. Purpose

Phase 5A implements a small deterministic local pilot for the historical data generator. The pilot verifies that the approved Phase 2 contracts, Phase 3 fraud scenarios, and Phase 4 market-data design can work together before the project scales to portfolio-size data.

Local pilot data is for generator validation. The approved full-scale dataset will later be written to ADLS using the same generator logic after cloud storage setup.

## 2. Local Pilot Versus Future ADLS Run

The pilot writes only to `data/generated/historical/pilot/`. It does not upload data, create cloud resources, create Databricks tables, or write Bronze/Silver outputs.

The future production-sized flow is:

```text
Same approved generator
-> full historical dataset
-> ADLS raw/historical folders
-> Databricks Bronze tables
```

## 3. Generator Modules

The Phase 5A generator lives under `src/crypto_fraud_platform/data_generator/`.

| Module | Purpose |
| --- | --- |
| `common/config.py` | Loads non-secret YAML configuration. |
| `common/identifiers.py` | Creates deterministic UUID identifiers. |
| `common/parquet_io.py` | Writes local PyArrow-backed Parquet outputs. |
| `common/schema_validation.py` | Validates records against JSON Schema Draft 2020-12. |
| `common/time_utils.py` | Centralizes UTC timestamp parsing and formatting. |
| `historical/market_candle_downloader.py` | Downloads and normalizes real Coinbase historical candles. |
| `historical/entity_generator.py` | Generates accounts, devices, and wallets. |
| `historical/authentication_generator.py` | Generates normal and suspicious authentication events. |
| `historical/transaction_generator.py` | Generates customer transactions and market-price enrichment. |
| `historical/label_generator.py` | Emits delayed fraud labels separately from transactions. |
| `historical/orchestrator.py` | Runs the local pilot end to end. |
| `historical/quality_report.py` | Validates output quality and leakage rules. |
| `fraud_scenarios/scenario_engine.py` | Reads the approved Phase 3 scenario registry and scenario YAML files. |

## 4. Coinbase Market Downloader

The downloader uses the public Coinbase Exchange candle endpoint:

```text
https://api.exchange.coinbase.com/products/{product_id}/candles
```

It downloads only `BTC-USD` and `ETH-USD` for the configured pilot period. It uses one-minute granularity, request windows of no more than 300 candles, bounded retries, exponential backoff, conservative throttling, and a local checkpoint file at `_metadata/market_download_checkpoint.json`.

If the endpoint or internet access is unavailable, the generator does not fabricate candles or random prices. Unit tests use mocked candle responses, but mocked data must not be presented as real Coinbase market data.

## 5. Entity Generation

The entity generator creates deterministic synthetic accounts, devices, and wallets that conform to the approved schemas. It does not include real personal information.

Accounts include country, KYC, risk tier, normal amount, normal frequency, preferred assets, and status. Devices include trusted normal devices plus untrusted shared devices needed for suspicious shared-device scenarios. Wallets include customer wallets owned by accounts, external wallets with null owners, and shared external wallets for mule-account activity.

## 6. Authentication Generation

Authentication events include common event metadata and resolve to valid account and device identifiers. Normal events are mostly successful, trusted-device logins from expected countries. Scenario-controlled events include failed logins, MFA failures, password-reset attempts, new or untrusted devices, country changes, and shared-device usage.

## 7. Transaction Generation

Customer transactions are generated as deposits, withdrawals, and transfers with BTC and ETH only. Completed, pending, and failed statuses are represented in the local pilot. All transaction records conform to `customer-transaction.schema.json`.

Transaction records do not contain `is_fraud`, `fraud_type`, `scenario_id`, `investigation_status`, or label status fields.

## 8. Market-Price Enrichment

For every generated customer transaction:

1. `BTC` maps to `BTC-USD`.
2. `ETH` maps to `ETH-USD`.
3. The generator selects the most recent completed candle where `candle_end_timestamp <= transaction event_timestamp`.
4. The selected candle close price becomes `market_price_usd`.
5. `transaction_amount_usd` is calculated as `crypto_quantity * market_price_usd`.
6. Market match metadata is written to `_audit/market_enrichment_matches.parquet`, not to model-facing transaction records.

The pilot uses JSON Schema `number` fields and PyArrow-backed Parquet. Later Delta tables should use fixed-precision decimal types where appropriate.

## 9. Fraud Scenario Injection

The scenario engine reads:

```text
config/fraud-rules.yaml
config/fraud-scenarios/*.yaml
```

The six enabled Phase 3 scenarios are represented:

| Scenario | Fraud type |
| --- | --- |
| Account takeover | `ACCOUNT_TAKEOVER` |
| High transaction velocity | `HIGH_TRANSACTION_VELOCITY` |
| Unusual transaction amount | `UNUSUAL_TRANSACTION_AMOUNT` |
| Structuring | `STRUCTURING` |
| Mule-account activity | `MULE_ACCOUNT_ACTIVITY` |
| Shared suspicious device | `SHARED_SUSPICIOUS_DEVICE` |

The optional high-volatility unusual-withdrawal scenario remains disabled by default.

## 10. Delayed Labels

Fraud labels are written as separate `fraud_label` events after the transaction timestamp. Confirmed fraud labels use `SIMULATED_INVESTIGATION`, `CONFIRMED_FRAUD`, `is_fraud = true`, and the approved fraud type. Normal completed control transactions receive delayed `CLEARED` labels with `is_fraud = false` and `fraud_type = null`.

Future labels are never used to create transaction records or model-facing features.

## 11. Output Folders

The local pilot writes:

```text
data/generated/historical/pilot/
├── market_candles/
├── accounts/
├── devices/
├── wallets/
├── authentication_events/
├── customer_transactions/
├── fraud_labels/
├── _audit/
└── _metadata/
```

The `_audit` folder contains generator-only metadata and must not be used as model input.

## 12. Parquet Design

Entity datasets are written as compact Parquet files. Event datasets are partitioned by event date, and labels are partitioned by label date. The market candle dataset is partitioned by `product_id` and `event_date`.

The manifest is written only after successful generation and quality validation.

## 13. Deterministic Random Seed

The default pilot seed is `20260729`. The same seed and configuration should produce the same synthetic entity and event structure, subject to successful retrieval of the same Coinbase candle input.

## 14. Leakage Prevention

Model-facing customer transactions must not contain:

```text
is_fraud
fraud_type
scenario_id
investigation_status
label_status
```

Scenario metadata is stored only in `_audit/scenario_assignments.parquet`.

## 15. Quality Checks

The quality report validates:

| Category | Checks |
| --- | --- |
| Uniqueness | Primary event and entity identifiers, transaction IDs, login IDs, and market natural keys. |
| Relationships | Account, device, wallet, and transaction foreign keys. |
| Timestamps | UTC ordering, label delays, activity-window bounds, and no future market candle usage. |
| Values | BTC/ETH only, positive quantities/prices, amount calculation, valid products. |
| Fraud distribution | Confirmed fraud count, fraud rate, and scenario distribution. |
| Leakage | Absence of model-facing label and scenario fields in transactions. |

## 16. How To Run The Pilot

Run from the repository root:

```bash
python scripts/generate_historical_pilot.py --config config/historical-generator.yaml
```

Useful flags:

```text
--overwrite
--skip-market-download
--validate-only
--output-root
```

`--skip-market-download` requires valid previously downloaded Coinbase candle Parquet files. It does not create fake market data.

## 17. How To Validate Outputs

Run:

```bash
python -m pytest tests/data_quality/test_historical_pilot_output.py -q -p no:cacheprovider
```

The test suite skips when no successful local pilot manifest exists.

## 18. Known Limitations

The pilot is intentionally small and local. It does not calibrate final fraud rates, model thresholds, production feature definitions, cloud storage paths, Databricks ingestion, Event Hubs, MLflow, dashboards, or full-scale generation. Coinbase availability can prevent the actual pilot output from being produced in an offline environment.

## 19. Future Cloud-Output Mode

Later phases will add full historical scale, ADLS raw output, Databricks Bronze ingestion, Silver transformations, feature generation, model training, and live-processing consistency. The Phase 5A logic is structured so those modes can reuse the same contract-aware generation path.

## 20. Phase 5 Status

Status: Local pilot implementation ready for validation and review.
