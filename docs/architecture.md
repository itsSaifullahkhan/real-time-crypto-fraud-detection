# Architecture

This project implements an end-to-end crypto fraud detection platform with separate offline training, live scoring, live durable storage, feedback, retraining, and Gold monitoring paths.

## Architecture Diagram

```mermaid
flowchart LR
  subgraph Offline["OFFLINE PATH"]
    A1["Coinbase historical market data"] --> A2["ADLS / landing"]
    A3["Generated customers, auth, transactions, labels"] --> A2
    A2 --> A4["Bronze Delta"]
    A4 --> A5["Silver Delta"]
    A5 --> A6["Offline 55-feature table"]
    A6 --> A7["Chronological training dataset"]
    A7 --> A8["Logistic Regression baseline"]
    A7 --> A9["XGBoost candidate"]
    A8 --> A10["MLflow tracking"]
    A9 --> A10
    A10 --> A11["Unity Catalog Model Registry"]
  end

  subgraph LiveScoring["LIVE SCORING PATH"]
    L0["Lakeflow Job: crypto-fraud-live-demo"] --> B1["Bounded Coinbase producer"]
    L0 --> B3["Bounded synthetic customer generator"]
    L0 --> B7["Databricks Real-Time Mode scoring"]
    L0 --> D1["Alert monitor"]
    B1 --> B2["market-events"]
    B3 --> B4["transaction-events"]
    B3 --> B5["authentication-events"]
    B2 --> B6["Azure Event Hubs"]
    B4 --> B6
    B5 --> B6
    B6 --> B7
    B7 --> B8["Point-in-time live 55 features"]
    A11 --> B9["XGBoost candidate alias"]
    B8 --> B9
    B9 --> B10["Fraud probability"]
    B10 --> B11["Threshold 0.80"]
    B11 --> B12["fraud-decisions"]
    B12 --> D1
    D1 --> D2["High-risk notification adapter"]
  end

  subgraph LiveStorage["LIVE STORAGE + FEEDBACK PATH"]
    L0 --> C6["bronze-storage consumer"]
    C1["market-events"] --> C6["bronze-storage consumer"]
    C2["transaction-events"] --> C6
    C3["authentication-events"] --> C6
    C4["fraud-labels"] --> C6
    C5["fraud-decisions"] --> C6
    C6 --> C7["Live Bronze Delta / ADLS"]
    L0 --> C8["Delayed-label feedback join"]
    C7 --> C8["Delayed-label feedback join"]
    C8 --> C9["Monitoring metrics"]
    C8 --> C10["Retraining-ready dataset"]
    C10 --> C11["Controlled retraining run"]
    L0 --> C12["Gold analytics"]
    C9 --> C12["Gold analytics"]
    C7 --> C12
  end

  B3 -. delayed labels only .-> C4
  B12 --> C5
```

## One-Click Demo Orchestration

The `crypto-fraud-live-demo` Lakeflow Job is the portfolio demo entry point. A manual `Run Now` starts managed job compute with `spark.databricks.streaming.realTimeMode.enabled=true`, runs bounded producers, runs Real-Time scoring against `crypto_fraud.models.fraud_detection_model@candidate`, monitors `fraud-decisions` for alertable model decisions, persists live Event Hubs data to Bronze, refreshes feedback without retraining, refreshes Gold analytics, and finishes with a final summary task.

The alert path is event-driven from `fraud-decisions`. Alerts require `predicted_fraud = true` and `fraud_probability >= 0.80`; delayed `fraud-labels` are not used to create prediction alerts. A separate test-alert mechanism validates the external webhook channel without inserting a test event into `fraud-decisions` or affecting Bronze, Gold, or model metrics.

The demo uses bounded components and preserves existing checkpoints. Bronze storage uses the `bronze-storage` consumer group and existing Delta tables. Feedback refresh runs with retraining disabled for ordinary demos; controlled retraining remains a separate workflow.

## Key Boundaries

- Fraud labels are never used by the live scoring feature pipeline.
- Fraud labels enter only after scoring, through feedback, monitoring, and retraining workflows.
- Event Hubs are transport and buffering; Delta tables in Unity Catalog provide durable storage.
- The live scoring consumer group is `stream-processing`.
- The durable Bronze storage consumer group is `bronze-storage`.
- Phase 16 registered retrained model version `2`, but the active live model remains `crypto_fraud.models.fraud_detection_model@candidate`.
- Threshold `0.80` is preserved in scoring, feedback, Gold analytics, and dashboard SQL.

## Data Layers

- Landing: generated historical files and collected source inputs.
- Bronze: durable raw or near-raw records with ingestion and Kafka/Event Hubs metadata.
- Silver: cleaned, typed, deduplicated entities and events.
- Features: point-in-time offline features, training datasets, and retraining-ready feedback datasets.
- Monitoring: delayed-label feedback tables and model outcome metrics.
- Gold: business-friendly transaction decisions, KPI summary, activity time series, and model monitoring summaries.

## Final Gold Outputs

- `crypto_fraud.gold.fraud_transaction_decisions`
- `crypto_fraud.gold.fraud_kpi_summary`
- `crypto_fraud.gold.fraud_activity_timeseries`
- `crypto_fraud.gold.model_monitoring_summary`

Dashboard-ready SQL is available in `sql/fraud_monitoring_dashboard.sql`.
