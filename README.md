# 🛡️ Real-Time Crypto Fraud Detection Platform

<p align="left">
  <img src="https://img.shields.io/badge/DATABRICKS-FF3621?style=for-the-badge&logo=databricks&logoColor=white" />
  <img src="https://img.shields.io/badge/APACHE%20SPARK-E25A1C?style=for-the-badge&logo=apachespark&logoColor=white" />
  <img src="https://img.shields.io/badge/MLFLOW-0194E2?style=for-the-badge&logo=mlflow&logoColor=white" />
  <img src="https://img.shields.io/badge/PYTHON-3776AB?style=for-the-badge&logo=python&logoColor=white" />
</p>

A production-oriented **Data Engineering + Machine Learning platform** for detecting suspicious cryptocurrency transactions using Databricks, behavioural feature engineering, MLflow experiment tracking, and a real-time fraud scoring architecture.

---

# 🎯 Problem Statement

Crypto platforms process large volumes of transactions 24/7, making manual fraud monitoring difficult.

Fraud detection must identify suspicious activity while handling:

* High transaction volumes
* Highly imbalanced fraud data
* Rapid transaction behaviour
* Authentication and device anomalies
* Risky destination wallets
* Changing crypto market conditions

The goal is to transform raw transaction data into **ML-ready behavioural features** and generate reliable fraud-risk predictions.

---

# 💡 Solution

The platform combines:

* **Databricks Lakehouse** for scalable data processing
* **Delta Lake** for reliable transaction storage
* **Unity Catalog** for governed ML datasets
* **PySpark** for data transformation and feature engineering
* **Scikit-learn** for fraud classification
* **MLflow** for experiment and model tracking
* **Structured Streaming** for future real-time fraud scoring

The current implementation includes a complete baseline ML workflow using **55 engineered features**, DummyClassifier benchmarking, Logistic Regression, model evaluation, MLflow logging, and model reload validation.

---

# 🏗️ Architecture Overview

<p align="center">
  <img src="architecture (3).png" width="100%" alt="Real-Time Crypto Fraud Detection Architecture">
</p>

<p align="center">
  <i>End-to-end fraud detection architecture combining data engineering, machine learning, MLOps, and real-time transaction scoring.</i>
</p>

---

# 🧩 Core Architecture

| Layer               | Technology           | Purpose                             |
| ------------------- | -------------------- | ----------------------------------- |
| Data Processing     | Databricks / Spark   | Distributed transaction processing  |
| Storage             | Delta Lake           | Reliable lakehouse storage          |
| Governance          | Unity Catalog        | Governed ML datasets                |
| Feature Engineering | PySpark              | Behavioural and fraud-risk features |
| Machine Learning    | Scikit-learn         | Fraud classification                |
| MLOps               | MLflow               | Experiments, metrics, and models    |
| Streaming           | Structured Streaming | Real-time transaction processing    |
| Analytics           | Gold Layer           | Fraud predictions and analytics     |

---

# 🧠 Machine Learning

Current baseline implementation:

```text
databricks/04_logistic_regression_baseline.py
```

Training dataset:

```text
crypto_fraud.features.transaction_training_dataset
```

The pipeline performs:

* Data validation
* 55-feature selection
* Train / validation / test split
* Feature preprocessing
* DummyClassifier benchmark
* Logistic Regression training
* Model evaluation
* MLflow experiment logging
* Model artifact validation
* Evaluation reports and plots

---

# 📊 Model Evaluation

Fraud detection is an imbalanced classification problem, so the project evaluates models using more than accuracy.

| Metric           | Purpose                          |
| ---------------- | -------------------------------- |
| Precision        | Measures quality of fraud alerts |
| Recall           | Measures detected fraud cases    |
| F1 Score         | Balances precision and recall    |
| PR-AUC           | Evaluates minority fraud class   |
| ROC-AUC          | Measures classification quality  |
| Confusion Matrix | Shows prediction errors          |

---

# ⚡ Real-Time Fraud Scoring

The next stage of the platform extends the trained model toward streaming inference.

Real-time scoring is designed to combine:

* Transaction information
* Customer behaviour
* Transaction velocity
* Authentication risk
* Device activity
* Wallet risk
* Market context

The model will generate outputs such as:

```text
transaction_id
risk_score
decision
reason_codes
model_version
prediction_timestamp
```

---

# 💼 Business Value

### ⚡ Faster Detection

Identify suspicious transactions closer to transaction time.

### 💰 Reduced Fraud Loss

Provide earlier opportunities to investigate or stop high-risk activity.

### 🎯 Better Prioritization

Use fraud-risk scores to focus analysts on the highest-risk transactions.

### 📉 Reduced Manual Review

Automate transaction screening instead of reviewing every transaction manually.

### 📊 Better Fraud Intelligence

Analyze transaction patterns, suspicious wallets, authentication anomalies, and changing fraud behaviour.

---

# 🛠️ Tech Stack

<p align="left">
  <img src="https://img.shields.io/badge/Databricks-FF3621?style=flat-square&logo=databricks&logoColor=white" />
  <img src="https://img.shields.io/badge/Apache%20Spark-E25A1C?style=flat-square&logo=apachespark&logoColor=white" />
  <img src="https://img.shields.io/badge/Delta%20Lake-00ADD8?style=flat-square" />
  <img src="https://img.shields.io/badge/Unity%20Catalog-5B5FC7?style=flat-square" />
  <img src="https://img.shields.io/badge/Scikit--learn-F7931E?style=flat-square&logo=scikitlearn&logoColor=white" />
  <img src="https://img.shields.io/badge/MLflow-0194E2?style=flat-square&logo=mlflow&logoColor=white" />
  <img src="https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=python&logoColor=white" />
</p>

---

# 👨‍💻 Author

## Saifullah Khan

**Data Engineering • FinTech • Machine Learning**

[GitHub](https://github.com/itsSaifullahkhan)

---

<p align="center">
  <b>Building scalable data and machine learning systems for financial fraud detection.</b>
</p>
