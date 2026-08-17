# Schema Definitions

This directory contains JSON Schema contracts for the Real-Time Crypto Fraud Detection Platform.

## Why JSON Schema

JSON Schema gives the project a machine-readable contract that can be used by Python generators, Coinbase market collectors, Parquet writers, Databricks Bronze ingestion, Silver transformations, Azure Event Hubs payload validation, Spark Structured Streaming, offline feature engineering, real-time feature processing, model scoring, and delayed fraud-label processing.

The schemas define structure and validation rules only. They do not create datasets, cloud resources, Databricks tables, Event Hubs, Coinbase connections, features, models, or fraud scenarios.

## Standard and Versioning

All schemas use JSON Schema Draft 2020-12.

The initial schema version is `1.0`. Future versions should follow semantic versioning:

- `1.0`: initial approved contract.
- `1.x`: backward-compatible optional additions.
- `2.0`: breaking changes.

Schema changes must be backward compatible unless the major version changes. Any schema change must update the JSON Schema file, `docs/data-contracts.md`, sample fixtures, and schema tests together.

## Contracts

- `common-event.schema.json`
- `account.schema.json`
- `device.schema.json`
- `wallet.schema.json`
- `authentication-event.schema.json`
- `customer-transaction.schema.json`
- `market-event.schema.json`
- `historical-market-candle.schema.json`
- `fraud-label.schema.json`
- `fraud-decision.schema.json`

`market-event.schema.json` is the live trade-level market event contract. `historical-market-candle.schema.json` is a separate raw Coinbase historical candle contract because historical candles do not contain trade IDs, buy/sell side, message timestamps, or sequence numbers.

## Validation

Check JSON syntax with:

```bash
python -m json.tool config/schemas/common-event.schema.json
```

Repeat for each schema file, or run the schema test suite:

```bash
pytest tests/data_quality/test_json_schemas.py -v
```

The tests validate that every schema is a valid Draft 2020-12 schema, valid fixtures pass, invalid fixtures fail, and important business constraints such as transaction label separation and risk-score bounds are enforced.

## Fixtures and Tests

One small valid sample record per contract is stored in:

```text
tests/fixtures/schemas/valid/
```

Representative invalid records are stored in:

```text
tests/fixtures/schemas/invalid/
```

Schema validation tests are stored in:

```text
tests/data_quality/test_json_schemas.py
```
