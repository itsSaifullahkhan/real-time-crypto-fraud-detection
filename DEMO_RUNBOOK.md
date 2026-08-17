# Demo Runbook

This runbook demonstrates the completed Real-Time Crypto Fraud Detection Platform without changing model aliases, thresholds, Azure infrastructure, or historical data.

Do not paste or print connection strings, SAS keys, Databricks tokens, or secret values during the demo.

## Pre-Demo Check

1. Confirm the Azure Event Hubs namespace exists.

2. Confirm the five required hubs exist:
   - `market-events`
   - `transaction-events`
   - `authentication-events`
   - `fraud-labels`
   - `fraud-decisions`

3. Confirm consumer groups exist:
   - `stream-processing`
   - `bronze-storage`

4. Confirm Databricks cluster exists:
   - name: `crypto-fraud-dev-compute`
   - cluster id: `0803-061312-78fw66xn`

5. Confirm Real-Time Mode is enabled on the Databricks cluster.

6. Confirm local `.env` exists and contains Event Hubs settings. Do not print its values.

7. Confirm Databricks secret exists:
   - scope: `crypto-fraud-secrets`
   - key: `eventhubs-databricks-connection`

8. Confirm the Databricks CLI profile is configured:

```powershell
databricks --profile crypto-fraud-dev current-user me
```

## ONE-CLICK ORCHESTRATED DEMO

Recommended instructor demo path:

1. Open Databricks.
2. Open Jobs & Pipelines.
3. Open `crypto-fraud-live-demo`.
4. Click `Run Now`.
5. Watch the Lakeflow task DAG.
6. Show Event Hubs activity for `market-events`, `transaction-events`, `authentication-events`, `fraud-labels`, and `fraud-decisions`.
7. Show Bronze tables under `crypto_fraud.bronze`.
8. Show fraud decisions in `crypto_fraud.bronze.live_fraud_decisions`.
9. Show the alert monitor result. A real alert is sent only for `predicted_fraud = true` and `fraud_probability >= 0.80`. The separate `send_test_alert` parameter validates notification delivery without writing test data to `fraud-decisions`.
10. Show Gold/dashboard queries from `sql/fraud_monitoring_dashboard.sql`.
11. Confirm the job status is `SUCCESS`.

Default job parameters:

| Parameter | Default | Purpose |
| --- | ---: | --- |
| `demo_duration_seconds` | `300` | Bounded producer emission window. |
| `target_transactions` | `25` | Target synthetic live transactions for the demo. |
| `fraud_rate` | `0.10` | Controlled synthetic fraud-label rate for delayed feedback only. |
| `refresh_feedback` | `true` | Refresh feedback tables after the live window. |
| `refresh_gold` | `true` | Refresh Gold analytics after feedback refresh. |
| `enable_alerting` | `true` | Monitor `fraud-decisions` for high-risk model decisions. |
| `send_test_alert` | `true` | Send a clearly labeled test notification when a webhook secret is configured. |

The job uses managed Databricks job compute with Real-Time Mode enabled, so a separate manual cluster start is not required. The live producers, Real-Time scoring, and alert monitor run concurrently during the live window. Bronze storage drains the retained Event Hubs messages into existing Bronze Delta tables after the scoring window to avoid single-node Real-Time scheduler slot contention while preserving checkpoints and the `bronze-storage` consumer group.

The ordinary demo run refreshes feedback with `run_retraining=false`; it does not retrain XGBoost, register another model version, move the `candidate` alias, or change threshold `0.80`.

External notification setup is optional for the orchestration job. To enable real webhook/test-alert delivery, create a Databricks secret:

- scope: `crypto-fraud-secrets`
- key: `fraud-alert-webhook-url`
- value: HTTPS webhook URL for Teams, an email bridge, or another approved notification endpoint

If that secret is not present, the alert monitor still verifies `fraud-decisions` consumption and exits successfully with `BLOCKED_MANUAL` in its alert-channel summary. No secret value should be printed during the demo.

## Live Demo Start Order

Run commands from the repository root.

### A. Start Databricks Compute

```powershell
databricks --profile crypto-fraud-dev clusters start 0803-061312-78fw66xn
databricks --profile crypto-fraud-dev clusters get 0803-061312-78fw66xn
```

Wait until the cluster state is `RUNNING`.

### B. Start Phase 13 Real-Time Scoring

Import the current notebook:

```powershell
databricks --profile crypto-fraud-dev workspace import /Users/akanaskhan1506@gmail.com/06_realtime_fraud_scoring --file databricks\06_realtime_fraud_scoring.py --format SOURCE --language PYTHON --overwrite
```

Submit one demo run:

```powershell
New-Item -ItemType Directory -Force .tmp | Out-Null
@'
{
  "run_name": "phase13_realtime_fraud_scoring_demo",
  "tasks": [
    {
      "task_key": "phase13_realtime_fraud_scoring",
      "existing_cluster_id": "0803-061312-78fw66xn",
      "notebook_task": {
        "notebook_path": "/Users/akanaskhan1506@gmail.com/06_realtime_fraud_scoring",
        "source": "WORKSPACE"
      },
      "timeout_seconds": 1800
    }
  ]
}
'@ | Set-Content -LiteralPath .tmp\phase13_demo_submit.json

databricks --profile crypto-fraud-dev jobs submit --json '@.tmp\phase13_demo_submit.json'
```

Save the returned `run_id`; it is needed for status checks and shutdown.

### C. Start Coinbase Market Producer

```powershell
New-Item -ItemType Directory -Force .tmp | Out-Null
$market = Start-Process -FilePath python -ArgumentList @(
  'src\crypto_fraud_platform\websocket_collector\coinbase_market_producer.py',
  '--validate-schema'
) -PassThru -RedirectStandardOutput .tmp\demo_market.out.log -RedirectStandardError .tmp\demo_market.err.log
$market.Id | Set-Content -LiteralPath .tmp\demo_market.pid
```

### D. Start Live Customer Generator

This emits synthetic live transactions, authentication events, and delayed fraud labels. Labels remain separate from the scoring feature pipeline.

```powershell
$customer = Start-Process -FilePath python -ArgumentList @(
  'src\crypto_fraud_platform\live_generators\live_customer_generator.py',
  '--interval', '1.0',
  '--fraud-rate', '0.10',
  '--label-delay', '10',
  '--validate-schema'
) -PassThru -RedirectStandardOutput .tmp\demo_customer.out.log -RedirectStandardError .tmp\demo_customer.err.log
$customer.Id | Set-Content -LiteralPath .tmp\demo_customer.pid
```

### E. Confirm Events Flow Through Event Hubs

Check local producer logs:

```powershell
Get-Content -LiteralPath .tmp\demo_market.out.log -Tail 20
Get-Content -LiteralPath .tmp\demo_customer.out.log -Tail 20
```

Check the Phase 13 Databricks run:

```powershell
databricks --profile crypto-fraud-dev jobs get-run <PHASE13_RUN_ID>
```

### F. Confirm Fraud Decisions Are Generated

After Phase 13 completes, retrieve the task output. Use the task run id from `jobs get-run`.

```powershell
databricks --profile crypto-fraud-dev jobs get-run-output <PHASE13_TASK_RUN_ID>
```

Expected demo indicators:

- market events consumed > 0
- transaction events consumed > 0
- authentication events consumed > 0
- live model feature count = 55
- fraud probabilities generated
- threshold = 0.80
- fraud decisions published

### G. Run Phase 14 Durable Bronze Storage If Needed

Use this when the demo needs to refresh durable Bronze from retained Event Hubs messages.

```powershell
databricks --profile crypto-fraud-dev workspace import /Users/akanaskhan1506@gmail.com/07_live_bronze_storage --file databricks\07_live_bronze_storage.py --format SOURCE --language PYTHON --overwrite

@'
{
  "run_name": "phase14_live_bronze_storage_demo",
  "tasks": [
    {
      "task_key": "phase14_live_bronze_storage",
      "existing_cluster_id": "0803-061312-78fw66xn",
      "notebook_task": {
        "notebook_path": "/Users/akanaskhan1506@gmail.com/07_live_bronze_storage",
        "source": "WORKSPACE"
      },
      "timeout_seconds": 1800
    }
  ]
}
'@ | Set-Content -LiteralPath .tmp\phase14_demo_submit.json

databricks --profile crypto-fraud-dev jobs submit --json '@.tmp\phase14_demo_submit.json'
```

### H. Query Gold Tables

Gold tables from Phase 17:

```sql
SELECT COUNT(*) FROM crypto_fraud.gold.fraud_transaction_decisions;
SELECT * FROM crypto_fraud.gold.fraud_kpi_summary;
SELECT * FROM crypto_fraud.gold.fraud_activity_timeseries ORDER BY time_bucket_utc;
SELECT * FROM crypto_fraud.gold.model_monitoring_summary;
```

### I. Use Dashboard SQL

Open `sql/fraud_monitoring_dashboard.sql` in Databricks SQL or a notebook SQL cell. It includes queries for:

- KPI cards
- fraud activity over time
- asset breakdown
- country breakdown
- confusion matrix
- model monitoring metrics
- recent high-risk transactions

## Demo Observability Commands

Verify live producers are emitting:

```powershell
Get-Process -Id (Get-Content .tmp\demo_market.pid) -ErrorAction SilentlyContinue
Get-Process -Id (Get-Content .tmp\demo_customer.pid) -ErrorAction SilentlyContinue
Get-Content -LiteralPath .tmp\demo_market.out.log -Tail 20
Get-Content -LiteralPath .tmp\demo_customer.out.log -Tail 20
```

Verify Event Hub processing is active:

```powershell
databricks --profile crypto-fraud-dev jobs get-run <PHASE13_RUN_ID>
```

Verify fraud decisions exist:

```sql
SELECT COUNT(*) AS fraud_decision_rows
FROM crypto_fraud.bronze.live_fraud_decisions;
```

Verify Bronze live tables contain rows:

```sql
SELECT 'market' AS stream_name, COUNT(*) AS rows FROM crypto_fraud.bronze.live_market_events
UNION ALL
SELECT 'transactions', COUNT(*) FROM crypto_fraud.bronze.live_customer_transactions
UNION ALL
SELECT 'authentication', COUNT(*) FROM crypto_fraud.bronze.live_authentication_events
UNION ALL
SELECT 'fraud_labels', COUNT(*) FROM crypto_fraud.bronze.live_fraud_labels
UNION ALL
SELECT 'fraud_decisions', COUNT(*) FROM crypto_fraud.bronze.live_fraud_decisions;
```

Verify Gold tables contain rows:

```sql
SELECT 'transaction_decisions' AS table_name, COUNT(*) AS rows FROM crypto_fraud.gold.fraud_transaction_decisions
UNION ALL
SELECT 'kpi_summary', COUNT(*) FROM crypto_fraud.gold.fraud_kpi_summary
UNION ALL
SELECT 'time_series', COUNT(*) FROM crypto_fraud.gold.fraud_activity_timeseries
UNION ALL
SELECT 'model_monitoring', COUNT(*) FROM crypto_fraud.gold.model_monitoring_summary;
```

Verify model alias/version from the Gold monitoring table:

```sql
SELECT
  model_role,
  model_name,
  model_version,
  active_candidate_version,
  phase16_retrained_version,
  threshold,
  monitoring_note
FROM crypto_fraud.gold.model_monitoring_summary;
```

Verify MLflow experiment/model information in the Databricks UI:

- experiment: `/Users/akanaskhan1506@gmail.com/crypto-fraud-phase16-retraining`
- model: `crypto_fraud.models.fraud_detection_model`
- active live alias: `candidate`
- retrained comparison version: `2`

## Demo Stop Order

1. Stop local customer generator:

```powershell
if (Test-Path .tmp\demo_customer.pid) {
  Stop-Process -Id ([int](Get-Content .tmp\demo_customer.pid)) -ErrorAction SilentlyContinue
}
```

2. Stop Coinbase producer:

```powershell
if (Test-Path .tmp\demo_market.pid) {
  Stop-Process -Id ([int](Get-Content .tmp\demo_market.pid)) -ErrorAction SilentlyContinue
}
```

3. Stop Databricks streaming query/job if it is still active:

```powershell
databricks --profile crypto-fraud-dev jobs cancel-run <PHASE13_RUN_ID>
```

4. Confirm no active streaming verification job remains:

```powershell
databricks --profile crypto-fraud-dev api get '/api/2.1/jobs/runs/list?active_only=true&limit=20'
```

5. Terminate Databricks compute when finished:

```powershell
databricks --profile crypto-fraud-dev clusters delete 0803-061312-78fw66xn
```

Terminating the cluster after the demo avoids unnecessary Azure compute cost. Do not delete Event Hubs, storage, Unity Catalog tables, checkpoints, or model versions as part of the demo shutdown.
