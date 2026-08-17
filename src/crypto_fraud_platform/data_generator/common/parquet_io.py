from __future__ import annotations

import json
import shutil
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path
from typing import Any

import pandas as pd


TIMESTAMP_COLUMNS = {
    "created_at",
    "updated_at",
    "first_seen_at",
    "last_seen_at",
    "event_timestamp",
    "source_timestamp",
    "ingestion_timestamp",
    "trade_timestamp",
    "message_timestamp",
    "label_timestamp",
    "prediction_timestamp",
    "candle_start_timestamp",
    "candle_end_timestamp",
    "retrieved_at",
    "matched_market_candle_timestamp",
    "generation_started_at",
    "generation_completed_at",
}

DECIMAL_8_COLUMNS = {
    "normal_transaction_amount_usd",
    "crypto_quantity",
    "market_price_usd",
    "transaction_amount_usd",
    "open_price_usd",
    "high_price_usd",
    "low_price_usd",
    "close_price_usd",
    "price_usd",
    "volume",
}
DECIMAL_8_QUANT = Decimal("0.00000001")


def prepare_for_parquet(dataframe: pd.DataFrame) -> pd.DataFrame:
    prepared = dataframe.copy()
    for column in TIMESTAMP_COLUMNS.intersection(prepared.columns):
        prepared[column] = pd.to_datetime(prepared[column], utc=True)
    for column in DECIMAL_8_COLUMNS.intersection(prepared.columns):
        prepared[column] = prepared[column].map(_decimal_8)
    return prepared


def write_parquet_file(
    dataframe: pd.DataFrame,
    path: Path,
    *,
    overwrite: bool = False,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not overwrite:
        raise FileExistsError(f"Parquet file already exists: {path}")
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    if tmp_path.exists():
        tmp_path.unlink()
    prepare_for_parquet(dataframe).to_parquet(tmp_path, engine="pyarrow", index=False)
    if path.exists():
        path.unlink()
    tmp_path.replace(path)


def write_partitioned_parquet(
    dataframe: pd.DataFrame,
    root_path: Path,
    *,
    partition_columns: list[str],
    overwrite: bool = False,
) -> None:
    if root_path.exists():
        if not overwrite:
            raise FileExistsError(f"Parquet dataset already exists: {root_path}")
        shutil.rmtree(root_path)
    tmp_path = root_path.with_name(root_path.name + ".__tmp__")
    if tmp_path.exists():
        shutil.rmtree(tmp_path)
    tmp_path.parent.mkdir(parents=True, exist_ok=True)
    prepare_for_parquet(dataframe).to_parquet(
        tmp_path,
        engine="pyarrow",
        index=False,
        partition_cols=partition_columns,
    )
    tmp_path.rename(root_path)


def read_parquet_dataset(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path, engine="pyarrow")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, sort_keys=True, default=str)
        file.write("\n")
    if path.exists():
        path.unlink()
    tmp_path.replace(path)


def read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as file:
        return json.load(file)


def _decimal_8(value: Any) -> Decimal | None:
    if pd.isna(value):
        return None
    return Decimal(str(value)).quantize(DECIMAL_8_QUANT, rounding=ROUND_HALF_UP)
