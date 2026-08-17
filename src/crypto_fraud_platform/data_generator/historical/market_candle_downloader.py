from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd
import requests

from crypto_fraud_platform.data_generator.common.parquet_io import read_json, write_json
from crypto_fraud_platform.data_generator.common.time_utils import format_utc, parse_utc, utc_now


COINBASE_EXCHANGE_URL = "https://api.exchange.coinbase.com/products/{product_id}/candles"
RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}


@dataclass(frozen=True)
class RequestWindow:
    product_id: str
    start: datetime
    end_exclusive: datetime

    @property
    def key(self) -> str:
        return f"{self.product_id}|{format_utc(self.start)}|{format_utc(self.end_exclusive)}"


class CoinbaseResponseError(RuntimeError):
    pass


def is_retryable_status(status_code: int) -> bool:
    return status_code in RETRYABLE_STATUS_CODES


def generate_request_windows(
    product_id: str,
    start: datetime,
    end: datetime,
    *,
    granularity_seconds: int = 60,
    max_candles_per_request: int = 300,
) -> list[RequestWindow]:
    start_utc = parse_utc(start)
    end_exclusive = parse_utc(end) + timedelta(seconds=1)
    window_seconds = granularity_seconds * max_candles_per_request

    windows: list[RequestWindow] = []
    current = start_utc
    while current < end_exclusive:
        next_end = min(current + timedelta(seconds=window_seconds), end_exclusive)
        windows.append(RequestWindow(product_id=product_id, start=current, end_exclusive=next_end))
        current = next_end
    return windows


def normalize_coinbase_candles(
    payload: Any,
    *,
    product_id: str,
    retrieved_at: datetime,
    start: datetime,
    end_exclusive: datetime,
    granularity_seconds: int = 60,
) -> list[dict[str, Any]]:
    if not isinstance(payload, list):
        raise CoinbaseResponseError("Coinbase candle response must be a list")

    records: list[dict[str, Any]] = []
    start_utc = parse_utc(start)
    end_utc = parse_utc(end_exclusive)
    for item in payload:
        if not isinstance(item, list) or len(item) != 6:
            raise CoinbaseResponseError(f"Malformed Coinbase candle row: {item!r}")
        try:
            unix_time, low, high, open_price, close, volume = item
            candle_start = datetime.fromtimestamp(int(unix_time), tz=start_utc.tzinfo)
            candle_start = parse_utc(candle_start)
            low_value = float(low)
            high_value = float(high)
            open_value = float(open_price)
            close_value = float(close)
            volume_value = float(volume)
        except (TypeError, ValueError) as exc:
            raise CoinbaseResponseError(f"Malformed Coinbase candle values: {item!r}") from exc

        if not (start_utc <= candle_start < end_utc):
            continue
        candle_end = candle_start + timedelta(seconds=granularity_seconds)
        records.append(
            {
                "schema_version": "1.0",
                "source": "coinbase_exchange_rest_api",
                "product_id": product_id,
                "candle_start_timestamp": candle_start,
                "candle_end_timestamp": candle_end,
                "granularity_seconds": granularity_seconds,
                "open_price_usd": open_value,
                "high_price_usd": high_value,
                "low_price_usd": low_value,
                "close_price_usd": close_value,
                "volume": volume_value,
                "retrieved_at": parse_utc(retrieved_at),
            }
        )

    records.sort(key=lambda record: record["candle_start_timestamp"])
    deduped: dict[tuple[str, datetime], dict[str, Any]] = {}
    for record in records:
        deduped[(record["product_id"], record["candle_start_timestamp"])] = record
    return list(deduped.values())


class MarketCandleDownloader:
    def __init__(
        self,
        *,
        session: requests.Session | None = None,
        base_url: str = COINBASE_EXCHANGE_URL,
        timeout_seconds: int = 20,
        max_retries: int = 3,
        backoff_seconds: float = 0.5,
        throttle_seconds: float = 0.12,
    ) -> None:
        self.session = session or requests.Session()
        self.base_url = base_url
        self.timeout_seconds = timeout_seconds
        self.max_retries = max_retries
        self.backoff_seconds = backoff_seconds
        self.throttle_seconds = throttle_seconds

    def download(
        self,
        *,
        products: list[str],
        start: datetime,
        end: datetime,
        checkpoint_path: Path,
        granularity_seconds: int = 60,
        max_candles_per_request: int = 300,
    ) -> tuple[pd.DataFrame, dict[str, Any]]:
        checkpoint = self._load_checkpoint(checkpoint_path)
        all_records: list[dict[str, Any]] = []
        report = {
            "products": {},
            "total_missing_intervals": 0,
            "total_candles": 0,
            "checkpoint_path": str(checkpoint_path),
        }

        for product_id in products:
            windows = generate_request_windows(
                product_id,
                start,
                end,
                granularity_seconds=granularity_seconds,
                max_candles_per_request=max_candles_per_request,
            )
            product_records: list[dict[str, Any]] = []
            for window in windows:
                if window.key in checkpoint["completed_windows"]:
                    cached = checkpoint["completed_windows"][window.key].get("records", [])
                    product_records.extend(cached)
                    continue
                payload, retrieved_at = self._request_window(window, granularity_seconds)
                normalized = normalize_coinbase_candles(
                    payload,
                    product_id=product_id,
                    retrieved_at=retrieved_at,
                    start=window.start,
                    end_exclusive=window.end_exclusive,
                    granularity_seconds=granularity_seconds,
                )
                checkpoint["completed_windows"][window.key] = {
                    "product_id": product_id,
                    "start": format_utc(window.start),
                    "end_exclusive": format_utc(window.end_exclusive),
                    "retrieved_at": format_utc(retrieved_at),
                    "record_count": len(normalized),
                    "records": [_serialize_checkpoint_record(record) for record in normalized],
                }
                write_json(checkpoint_path, checkpoint)
                product_records.extend(normalized)
                time.sleep(self.throttle_seconds)

            product_frame = _records_to_frame(product_records)
            missing = calculate_missing_intervals(
                product_frame,
                product_id=product_id,
                start=start,
                end=end,
                granularity_seconds=granularity_seconds,
            )
            report["products"][product_id] = {
                "request_window_count": len(windows),
                "candle_count": int(len(product_frame)),
                "missing_interval_count": int(len(missing)),
                "missing_interval_samples": [format_utc(value) for value in missing[:20]],
            }
            report["total_missing_intervals"] += int(len(missing))
            report["total_candles"] += int(len(product_frame))
            all_records.extend(product_frame.to_dict(orient="records"))

        full_frame = _records_to_frame(all_records)
        full_frame = full_frame.sort_values(["product_id", "candle_start_timestamp"]).reset_index(
            drop=True
        )
        return full_frame, report

    def _request_window(
        self,
        window: RequestWindow,
        granularity_seconds: int,
    ) -> tuple[Any, datetime]:
        url = self.base_url.format(product_id=window.product_id)
        params = {
            "start": format_utc(window.start),
            "end": format_utc(window.end_exclusive),
            "granularity": granularity_seconds,
        }
        last_error: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                response = self.session.get(url, params=params, timeout=self.timeout_seconds)
                if response.status_code == 200:
                    return response.json(), utc_now()
                if not is_retryable_status(response.status_code):
                    raise CoinbaseResponseError(
                        f"Coinbase returned non-retryable status {response.status_code}"
                    )
                last_error = CoinbaseResponseError(
                    f"Coinbase returned retryable status {response.status_code}"
                )
            except (requests.RequestException, CoinbaseResponseError) as exc:
                last_error = exc
                if isinstance(exc, CoinbaseResponseError) and "non-retryable" in str(exc):
                    raise
            if attempt < self.max_retries:
                time.sleep(min(self.backoff_seconds * (2**attempt), 8.0))
        raise CoinbaseResponseError(f"Failed to download {window.key}: {last_error}") from last_error

    @staticmethod
    def _load_checkpoint(path: Path) -> dict[str, Any]:
        if not path.exists():
            return {"completed_windows": {}}
        checkpoint = read_json(path)
        checkpoint.setdefault("completed_windows", {})
        return checkpoint


def calculate_missing_intervals(
    candles: pd.DataFrame,
    *,
    product_id: str,
    start: datetime,
    end: datetime,
    granularity_seconds: int = 60,
) -> list[datetime]:
    start_utc = parse_utc(start)
    end_exclusive = parse_utc(end) + timedelta(seconds=1)
    expected = set()
    current = start_utc
    while current < end_exclusive:
        expected.add(current)
        current += timedelta(seconds=granularity_seconds)

    if candles.empty:
        actual: set[datetime] = set()
    else:
        product_candles = candles[candles["product_id"] == product_id]
        actual = {parse_utc(value) for value in product_candles["candle_start_timestamp"]}
    return sorted(expected - actual)


def _records_to_frame(records: list[dict[str, Any]]) -> pd.DataFrame:
    frame = pd.DataFrame(records)
    if frame.empty:
        return frame
    for column in ["candle_start_timestamp", "candle_end_timestamp", "retrieved_at"]:
        frame[column] = pd.to_datetime(frame[column], utc=True)
    frame = frame.drop_duplicates(["product_id", "candle_start_timestamp"], keep="last")
    return frame


def _serialize_checkpoint_record(record: dict[str, Any]) -> dict[str, Any]:
    serialized = record.copy()
    for column in ["candle_start_timestamp", "candle_end_timestamp", "retrieved_at"]:
        serialized[column] = format_utc(serialized[column])
    return serialized
