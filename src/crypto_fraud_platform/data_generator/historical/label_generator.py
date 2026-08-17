from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any

import numpy as np
import pandas as pd

from crypto_fraud_platform.data_generator.common.config import HistoricalGeneratorConfig
from crypto_fraud_platform.data_generator.common.identifiers import DeterministicIdFactory
from crypto_fraud_platform.data_generator.common.time_utils import parse_utc


def make_fraud_label(
    *,
    id_factory: DeterministicIdFactory,
    transaction_id: str,
    label_timestamp: datetime,
    is_fraud: bool,
    fraud_type: str | None,
    investigation_status: str,
) -> dict[str, Any]:
    label_time = parse_utc(label_timestamp)
    source_time = label_time + timedelta(seconds=1)
    return {
        "event_id": id_factory.next("fraud_label_event"),
        "event_type": "fraud_label",
        "schema_version": "1.0",
        "source": "simulated_investigation",
        "event_timestamp": label_time,
        "source_timestamp": source_time,
        "ingestion_timestamp": source_time + timedelta(seconds=1),
        "transaction_id": transaction_id,
        "is_fraud": is_fraud,
        "fraud_type": fraud_type,
        "label_timestamp": label_time,
        "label_source": "SIMULATED_INVESTIGATION",
        "investigation_status": investigation_status,
    }


def generate_fraud_labels(
    *,
    config: HistoricalGeneratorConfig,
    transactions: pd.DataFrame,
    scenario_fraud_rows: list[dict[str, Any]],
    rng: np.random.Generator,
    id_factory: DeterministicIdFactory,
) -> pd.DataFrame:
    fraud_by_transaction = {row["transaction_id"]: row for row in scenario_fraud_rows}
    records: list[dict[str, Any]] = []

    for transaction in transactions.to_dict(orient="records"):
        transaction_id = str(transaction["transaction_id"])
        transaction_time = parse_utc(transaction["event_timestamp"])
        if transaction_id in fraud_by_transaction:
            fraud_row = fraud_by_transaction[transaction_id]
            delay_hours = _delay_hours(fraud_row["label_delay_hours"], rng)
            records.append(
                make_fraud_label(
                    id_factory=id_factory,
                    transaction_id=transaction_id,
                    label_timestamp=transaction_time + timedelta(hours=delay_hours),
                    is_fraud=True,
                    fraud_type=str(fraud_row["fraud_type"]),
                    investigation_status="CONFIRMED_FRAUD",
                )
            )
        elif (
            config.generation.label_completed_transactions
            and transaction["transaction_status"] == "COMPLETED"
        ):
            records.append(
                make_fraud_label(
                    id_factory=id_factory,
                    transaction_id=transaction_id,
                    label_timestamp=transaction_time + timedelta(hours=float(rng.uniform(24, 168))),
                    is_fraud=False,
                    fraud_type=None,
                    investigation_status="CLEARED",
                )
            )

    return _labels_to_frame(records)


def _delay_hours(delay_range: dict[str, int | float], rng: np.random.Generator) -> float:
    return float(rng.uniform(float(delay_range["min"]), float(delay_range["max"])))


def _labels_to_frame(records: list[dict[str, Any]]) -> pd.DataFrame:
    frame = pd.DataFrame(records)
    if frame.empty:
        return frame
    for column in ["event_timestamp", "source_timestamp", "ingestion_timestamp", "label_timestamp"]:
        frame[column] = pd.to_datetime(frame[column], utc=True)
    return frame.sort_values("label_timestamp").reset_index(drop=True)
