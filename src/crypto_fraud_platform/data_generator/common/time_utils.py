from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Iterable

import numpy as np
import pandas as pd


def parse_utc(value: str | datetime | pd.Timestamp) -> datetime:
    if isinstance(value, pd.Timestamp):
        value = value.to_pydatetime()
    if isinstance(value, datetime):
        if value.tzinfo is None:
            return value.replace(tzinfo=UTC)
        return value.astimezone(UTC)
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def format_utc(value: str | datetime | pd.Timestamp) -> str:
    parsed = parse_utc(value)
    return parsed.isoformat(timespec="seconds").replace("+00:00", "Z")


def utc_now() -> datetime:
    return datetime.now(tz=UTC).replace(microsecond=0)


def add_seconds(value: datetime, seconds: int) -> datetime:
    return parse_utc(value) + timedelta(seconds=seconds)


def random_datetimes(
    rng: np.random.Generator,
    count: int,
    start: datetime,
    end: datetime,
) -> list[datetime]:
    start_ts = int(parse_utc(start).timestamp())
    end_ts = int(parse_utc(end).timestamp())
    values = rng.integers(start_ts, end_ts + 1, size=count)
    return [datetime.fromtimestamp(int(value), tz=UTC) for value in values]


def minute_floor(value: datetime) -> datetime:
    parsed = parse_utc(value)
    return parsed.replace(second=0, microsecond=0)


def date_string(value: str | datetime | pd.Timestamp) -> str:
    return parse_utc(value).date().isoformat()


def to_utc_datetime_series(values: Iterable[object]) -> pd.Series:
    return pd.to_datetime(list(values), utc=True)
