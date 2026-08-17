# Real-Time Crypto Fraud Detection Platform

A portfolio-grade fraud detection platform for suspicious cryptocurrency transaction monitoring. The system combines historical lakehouse processing, point-in-time feature engineering, MLflow model management, Databricks Real-Time Mode scoring, durable Event Hubs ingestion, delayed fraud feedback, controlled retraining, and Gold analytics.

## Problem

Crypto transaction monitoring needs fresh market context and customer behavior signals at scoring time. A platform must score transactions quickly, preserve raw events for auditability, and later join delayed fraud outcomes back to predictions without leaking labels into the live feature pipeline.

Customer, account, device, transaction, authentication, and fraud-label examples in this project are synthetically generated for a portfolio demonstration. Coinbase market data comes from the public Coinbase WebSocket/API feed. This project does not use or imply access to Coinbase private customer transaction data.

## Architecture

Historical / offline path:

Historical market data plus generated customer, authentication, transaction, and delayed-label data flow into ADLS-backed lakehouse storage. Databricks builds Bronze and Silver Delta tables, point-in-time offline features, a chronological training dataset, and MLflow-tracked Logistic Regression and XGBoost models. The selected XGBoost candidate is registered in Unity Catalog Model Registry.

Live scoring path:

Coinbase public WebSocket events publish to `market-events`. The synthetic live customer generator publishes `transaction-events`, `authentication-events`, and delayed `fraud-labels`. Azure Event Hubs transports those streams to Databricks Real-Time Mode, where the Phase 13 notebook builds the verified 55 live features, loads `crypto_fraud.models.fraud_detection_model@candidate`, scores fraud probability, applies threshold `0.80`, and publishes `fraud-decisions`.

Parallel durable path:

Event Hubs are also consumed independently with consumer group `bronze-storage`. Structured Streaming stores all five live streams into Unity Catalog Delta Bronze tables backed by ADLS. Delayed labels are joined to decisions only after scoring, feedback metrics are produced, a controlled retraining dataset is built, and Gold tables support dashboard analytics.

## Technologies

- Microsoft Azure
- Azure Event Hubs
- Azure Databricks
- Databricks Real-Time Mode
- Apache Spark / Structured Streaming
- Delta Lake
- Unity Catalog
- ADLS Gen2
- MLflow
- XGBoost
- Python
- Coinbase public WebSocket API

## Medallion / Data Layers

- `landing`: generated and collected source files before lakehouse refinement.
- `bronze`: raw or near-raw durable Delta records with source metadata.
- `silver`: validated, typed, deduplicated analytical entities and events.
- `features`: offline training features, training datasets, and feedback retraining datasets.
- `monitoring`: delayed-label prediction feedback and model outcome metrics.
- `gold`: dashboard-ready transaction decisions, KPIs, time series, and model monitoring summaries.

## ML Workflow

The project uses point-in-time-safe offline feature engineering and a chronological split. Logistic Regression provides a baseline, while XGBoost is the registered candidate model. Class imbalance is handled during training with fixed model configuration and tracked parameters. MLflow records experiments, artifacts, metrics, and Unity Catalog model versions.

Delayed fraud labels are joined to live decisions after scoring. Phase 16 created a retraining-ready dataset and ran one controlled XGBoost retraining experiment, registering model version `2` for comparison. The project intentionally does not automatically promote retrained models or move the `candidate` alias.

## Current Evaluation Limitation

This is a portfolio engineering pipeline, not a production fraud model. The labeled fraud sample is limited and synthetic. In the final live feedback sample, threshold `0.80` produced no positive alerts: TP = 0, FP = 0, TN = 714, FN = 91. The retrained model was registered separately but was not promoted. Production use would require more representative fraud data, calibration, monitoring, governance, and human review processes.

## Project Structure

- `config/`: feature definitions and project configuration.
- `databricks/`: phase notebooks for lakehouse, feature, streaming, feedback, and Gold analytics workflows.
- `docs/`: architecture, project status, data contracts, scenarios, and design notes.
- `reports/`: machine-readable phase summaries retained for validation evidence.
- `scripts/`: lightweight local utilities, including final smoke validation.
- `sql/`: dashboard-ready Databricks SQL queries.
- `src/crypto_fraud_platform/`: data generators, Event Hubs publisher, and Coinbase WebSocket producer.
- `tests/`: local unit and contract tests.

## Setup

Prerequisites:

- Azure subscription with the existing Event Hubs namespace and five hubs.
- Azure Databricks workspace with the existing Unity Catalog objects and cluster.
- Python environment with dependencies from `requirements.txt` and `requirements-dev.txt`.
- Databricks CLI configured with profile `crypto-fraud-dev`.
- Local `.env` file, which is intentionally ignored by Git.
- Databricks secret scope `crypto-fraud-secrets` with key `eventhubs-databricks-connection`.

Example local `.env` placeholders:

```powershell
EVENT_HUB_CONNECTION_STRING=<YOUR_CONNECTION_STRING>
EVENT_HUB_MARKET=market-events
EVENT_HUB_TRANSACTIONS=transaction-events
EVENT_HUB_AUTHENTICATION=authentication-events
EVENT_HUB_FRAUD_LABELS=fraud-labels
EVENT_HUB_FRAUD_DECISIONS=fraud-decisions
```

Never commit real credentials. If real credentials were ever exposed while developing or demonstrating the project, rotate them before publishing or sharing the repository.

## Demo

Use [DEMO_RUNBOOK.md](DEMO_RUNBOOK.md) for the exact pre-demo checklist, startup order, observability commands, dashboard SQL, and safe shutdown order.
