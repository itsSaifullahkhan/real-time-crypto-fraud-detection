-- Phase 17 fraud monitoring dashboard queries
-- Use these in Databricks SQL against the existing Unity Catalog Gold tables.

USE CATALOG crypto_fraud;
USE SCHEMA gold;

-- 1. KPI cards: transactions, actual fraud, fraud rate, total value.
SELECT
  total_transactions,
  total_scored_transactions,
  total_labeled_transactions,
  actual_fraud_count,
  actual_normal_count,
  predicted_fraud_count,
  fraud_rate,
  total_transaction_value_usd,
  actual_fraudulent_transaction_value_usd
FROM fraud_kpi_summary;

-- 2. Fraud activity over time.
SELECT
  time_bucket_utc,
  transaction_count,
  transaction_volume_usd,
  scored_transaction_count,
  actual_fraud_count,
  predicted_fraud_count,
  average_fraud_probability,
  max_fraud_probability,
  fraudulent_value_usd
FROM fraud_activity_timeseries
ORDER BY time_bucket_utc;

-- 3. Transactions and fraud by asset.
SELECT
  asset,
  transaction_count,
  transaction_value_usd,
  actual_fraud_count,
  predicted_fraud_count,
  average_fraud_probability,
  max_fraud_probability
FROM v_fraud_activity_by_asset
ORDER BY transaction_count DESC;

-- 4. Transactions and fraud by country.
SELECT
  country,
  transaction_count,
  transaction_value_usd,
  actual_fraud_count,
  predicted_fraud_count,
  average_fraud_probability,
  max_fraud_probability
FROM v_fraud_activity_by_country
ORDER BY transaction_count DESC;

-- 5. Confusion matrix / outcome summary.
SELECT
  outcome,
  transaction_count
FROM v_confusion_matrix
ORDER BY outcome;

-- 6. Model monitoring metrics.
SELECT
  model_role,
  model_name,
  model_version,
  active_candidate_version,
  phase16_retrained_version,
  threshold,
  labeled_sample_count,
  true_positives,
  false_positives,
  true_negatives,
  false_negatives,
  precision,
  recall,
  f1,
  pr_auc,
  roc_auc,
  fraud_prevalence,
  evaluation_timestamp,
  monitoring_note
FROM model_monitoring_summary
ORDER BY model_role;

-- 7. Recent high-risk transactions.
SELECT
  transaction_id,
  transaction_timestamp,
  account_id,
  asset,
  transaction_type,
  country,
  transaction_amount_usd,
  fraud_probability,
  threshold,
  predicted_fraud,
  actual_fraud,
  outcome,
  model_name,
  model_version
FROM v_recent_high_risk_transactions
ORDER BY fraud_probability DESC, transaction_timestamp DESC
LIMIT 50;

-- Optional: fraud probability distribution bands.
SELECT
  fraud_probability_band,
  transaction_count,
  actual_fraud_count,
  predicted_fraud_count
FROM v_fraud_probability_bands
ORDER BY fraud_probability_band;
