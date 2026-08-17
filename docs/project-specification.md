# Real-Time Crypto Fraud Detection Platform

## 1. Project Title

Real-Time Crypto Fraud Detection Platform

## 2. Executive Summary

This project will design and later implement an industry-style fraud-detection platform for cryptocurrency transaction risk decisions. It will combine real Coinbase public market data with simulated customer activity to support historical model training, live streaming feature calculation, and real-time fraud decisions.

## 3. Business Problem

Fraud-risk systems need fresh and consistent behavioural features during live predictions. Delayed or inconsistent feature values can reduce model quality and increase false positives, fraud losses, and manual-review volume.

## 4. Technical Problem

The platform must process customer, authentication, wallet, and market events with low latency. It must calculate stateful online features using definitions consistent with historical offline features, score transactions using an approved model, produce risk decisions, preserve events for replay and audit, and monitor streaming and model performance.

## 5. Proposed Solution

Build a hybrid real-time fraud-detection platform where:

- Coinbase public APIs provide real BTC-USD and ETH-USD market data.
- Python generates simulated customer accounts, devices, wallets, authentication events, transactions, controlled fraud scenarios, and delayed fraud labels.
- Historical data is stored in Parquet and processed through Bronze and Silver Delta layers before splitting into offline feature tables and Gold reporting tables.
- Gold tables are business-ready reporting and dashboard outputs. Model training uses point-in-time-correct offline feature tables derived from Silver data, not Gold tables.
- Logistic Regression is used as the baseline model.
- XGBoost is used as the main fraud model.
- MLflow manages experiments and model versions.
- Azure Event Hubs transports live events.
- Databricks Real-Time Mode and Spark Structured Streaming calculate stateful online features and perform live scoring.
- Fraud decisions are ALLOW, REVIEW, or BLOCK.
- Unity Catalog provides governance across data, features, and models.
- Databricks SQL dashboards present fraud, risk, model, and platform metrics.

## 6. Project Objectives

- Define a manageable portfolio-grade architecture for real-time fraud detection.
- Preserve consistency between offline historical features and online streaming features.
- Support replayable, auditable event processing.
- Train a Logistic Regression baseline and an XGBoost main model in a future phase.
- Register approved models with MLflow in a future phase.
- Produce low-latency fraud decisions in a future real-time pipeline.
- Monitor platform health, model quality, fraud outcomes, and review volume.

## 7. Intended Users

- Fraud analytics teams reviewing risk and fraud trends.
- Data engineers responsible for ingestion, streaming, and Delta Lake tables.
- Machine learning engineers responsible for model training, registration, and scoring.
- Data analysts using Databricks SQL dashboards.
- Portfolio reviewers evaluating the project architecture and implementation quality.

## 8. In-Scope Items

- BTC-USD and ETH-USD only.
- 10,000-25,000 generated accounts.
- 300,000-750,000 historical transactions.
- 100,000-300,000 authentication events.
- Approximately 0.5%-1% controlled fraud rate.
- Approximately six fraud scenarios.
- 20-30 features.
- Logistic Regression baseline.
- XGBoost main model.
- One real-time scoring pipeline.
- MLflow model registration.
- Bronze, Silver, Features, and Gold layers.
- Four dashboard areas.

## 9. Out-of-Scope Items

- LLMs.
- GenAI.
- Graph neural networks.
- Multiple blockchain networks.
- Hundreds of features.
- Multiple complex fraud models.
- Unnecessary multi-cloud infrastructure.
- Production handling of real customer funds.
- Private Coinbase customer information.

## 10. Data-Source Summary

Coinbase public APIs will provide only real market information such as:

- BTC-USD and ETH-USD prices.
- Market trades.
- Trade size.
- Buy or sell side.
- Event timestamps.
- Volume.
- Price changes.
- Volatility.

Coinbase public feeds are not customer fraud data.

Python will generate:

- Accounts.
- Devices.
- Wallets.
- Authentication activity.
- Customer deposits.
- Customer withdrawals.
- Crypto transfers.
- Normal behavioural profiles.
- Controlled fraud scenarios.
- Delayed fraud investigation labels.

Generated customer transactions will use real market prices and timestamps.

## 11. Historical Data Flow

```text
Historical Coinbase market data
+
Generated historical customer transactions
+
Generated authentication and device events
+
Delayed fraud labels
→ Parquet files
→ Lakeflow ingestion
→ Bronze Delta tables
→ Silver Delta tables
→ Offline feature tables
→ Logistic Regression and XGBoost training
→ MLflow Model Registry
```

Historical data must not be routed through Event Hubs.

Gold tables are business-ready reporting and dashboard outputs. Model training uses point-in-time-correct offline feature tables derived from Silver data, not Gold tables.

## 12. Live Data Flow

```text
Coinbase WebSocket
→ Python collector
→ Event Hub: market-events

Python live transaction generator
→ Event Hub: transaction-events

Python authentication generator
→ Event Hub: authentication-events

Python delayed-label generator
→ Event Hub: fraud-labels

Event Hubs
→ Databricks Real-Time Mode
→ Validation and deduplication
→ Market-state enrichment
→ Stateful online features
→ Approved MLflow model
→ Fraud probability
→ ALLOW / REVIEW / BLOCK
→ Event Hub: fraud-decisions
```

## 13. Success Metrics

Platform metrics:

- Processing latency.
- P50, P95, and P99 latency.
- Throughput.
- Consumer lag.
- Failed events.
- Duplicate-event handling.
- Late-event handling.
- Checkpoint recovery.
- Offline and online feature consistency.

Model metrics:

- Precision.
- Recall.
- F1-score.
- PR-AUC.
- False-positive rate.
- Fraud amount detected.
- Fraud amount missed.
- Manual-review volume.

Accuracy will not be used as the main model metric because fraud detection is expected to be highly imbalanced.

## 14. Major Assumptions

- Coinbase public market data access remains available for BTC-USD and ETH-USD.
- Generated customer data is acceptable for a student portfolio project.
- Delayed labels can be simulated realistically enough to evaluate workflows.
- Azure Student account limits will constrain infrastructure choices.
- The architecture should remain intact if a cloud component temporarily needs a local replacement.

## 15. Risks and Constraints

- Azure Student account limitations may restrict Event Hubs, Databricks, or storage usage.
- Databricks Real-Time Mode access may not be available.
- Cloud-cost limits may require smaller workloads or temporary local substitutes.
- Customer data will be generated, not real.
- Fraud labels will be delayed and imbalanced.
- Class imbalance will require precision, recall, PR-AUC, and false-positive analysis instead of accuracy-focused evaluation.
- The project must remain industry-style but manageable for a student project.
- If one cloud component requires a temporary local replacement, the overall architecture should still be preserved.

## 16. Final Deliverables

- Complete project documentation and architecture notes.
- Data contracts and schemas.
- Historical generated data design and sample outputs.
- Delta Lake Bronze, Silver, Feature, and Gold processing assets.
- Offline feature catalog.
- Logistic Regression baseline model.
- XGBoost main fraud model.
- MLflow experiment and model registry workflow.
- Real-time streaming scoring pipeline.
- Fraud decision output stream.
- Monitoring and Databricks SQL dashboard assets.
- Final portfolio README and limitations documentation.

## 17. Implementation Phases

1. Phase 1 — Project specification and repository setup
2. Phase 2 — Data contracts and schemas
3. Phase 3 — Fraud scenario catalogue
4. Phase 4 — Historical market data design
5. Phase 5 — Historical customer-data generator
6. Phase 6 — Lakehouse setup
7. Phase 7 — Historical ingestion and transformations
8. Phase 8 — Offline features
9. Phase 9 — Model training and MLflow
10. Phase 10 — Coinbase WebSocket collector
11. Phase 11 — Live customer generators
12. Phase 12 — Azure Event Hubs
13. Phase 13 — Real-Time Mode pipeline
14. Phase 14 — Live storage pipeline
15. Phase 15 — Feature-consistency testing
16. Phase 16 — Feedback and retraining
17. Phase 17 — Gold tables and dashboards
18. Phase 18 — Testing and documentation

## 18. Approval Status

This document is a Phase 1 draft and should be reviewed before Phase 2 begins.

Status: Draft complete — awaiting final approval before Phase 2.
