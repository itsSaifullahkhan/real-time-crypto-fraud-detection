# Project Status

| Phase | Purpose | Status | Main output |
|---|---|---:|---|
| 1 | Project foundation | PASS | Repository structure, Python package layout, initial docs |
| 2 | Event contracts | PASS | JSON schemas and event contract documentation |
| 3 | Fraud scenarios | PASS | Synthetic fraud scenario catalogue and generation rules |
| 4 | Historical market data design | PASS | Coinbase historical market-data collection design |
| 5 | Historical customer data | PASS | Generated customers, accounts, devices, wallets, transactions, authentication, and labels |
| 6 | Landing and lakehouse setup | PASS | Landing data prepared for Databricks lakehouse ingestion |
| 7 | Bronze ingestion | PASS | Historical Bronze Delta tables in Unity Catalog |
| 8 | Silver transformations and offline features | PASS | Silver tables and `crypto_fraud.features.transaction_features_offline` |
| 9A | Baseline model training | PASS | Logistic Regression baseline and chronological training split |
| 9B | XGBoost model selection | PASS | Registered `crypto_fraud.models.fraud_detection_model@candidate` |
| 10 | Live synthetic generators | PASS | Live transaction, authentication, and delayed fraud-label generator |
| 11 | Coinbase live market producer | PASS | Coinbase public WebSocket market producer |
| 12 | Event Hubs setup | PASS | Five Event Hubs and required consumer groups |
| 13 | Real-Time scoring | PASS | Real-Time Mode scoring, 55 live features, fraud decisions published |
| 14 | Live durable Bronze storage | PASS | Live Bronze Delta tables with `bronze-storage` consumer group |
| 15 | Offline/online feature consistency | PASS | 55-feature consistency validation, 0 mismatches, no label leakage |
| 16 | Delayed feedback and controlled retraining | PASS | Feedback table, metrics, retraining dataset, MLflow run, registered model version 2 |
| 17 | Gold analytics and dashboard layer | PASS | Gold transaction, KPI, time-series, model-monitoring tables, dashboard SQL |
| 18 | Final validation and documentation | PASS | Final smoke script, README, demo runbook, architecture doc, project status doc |

## Current Model State

- Live model URI: `models:/crypto_fraud.models.fraud_detection_model@candidate`
- Candidate alias version after Phase 17: `1`
- Phase 16 retrained comparison version: `2`
- Live threshold: `0.80`
- Automatic promotion: not performed

## Current Feedback Result

The final labeled live sample is intentionally reported as-is:

- labeled decisions: 805
- actual fraud: 91
- actual normal: 714
- TP: 0
- FP: 0
- TN: 714
- FN: 91
- precision: 0.0
- recall: 0.0
- F1: 0.0

This is not hidden or reframed as production accuracy. The project demonstrates the complete engineering workflow and would need richer fraud data, calibration, and governance before production use.
