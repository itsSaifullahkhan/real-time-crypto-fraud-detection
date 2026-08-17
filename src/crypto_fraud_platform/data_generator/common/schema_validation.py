from __future__ import annotations

import json
from datetime import date, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from jsonschema import Draft202012Validator, FormatChecker
from referencing import Registry, Resource
from referencing.jsonschema import DRAFT202012

from crypto_fraud_platform.data_generator.common.time_utils import format_utc


SCHEMA_FILES = {
    "common-event": "common-event.schema.json",
    "account": "account.schema.json",
    "device": "device.schema.json",
    "wallet": "wallet.schema.json",
    "authentication-event": "authentication-event.schema.json",
    "customer-transaction": "customer-transaction.schema.json",
    "market-event": "market-event.schema.json",
    "historical-market-candle": "historical-market-candle.schema.json",
    "fraud-label": "fraud-label.schema.json",
    "fraud-decision": "fraud-decision.schema.json",
}

TRANSACTION_LEAKAGE_FIELDS = {
    "is_fraud",
    "fraud_type",
    "scenario_id",
    "investigation_status",
    "label_status",
}


def load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as file:
        return json.load(file)


class SchemaValidator:
    def __init__(self, project_root: Path) -> None:
        self.schema_dir = project_root / "config" / "schemas"
        self.schemas = {
            schema_name: load_json(self.schema_dir / schema_file)
            for schema_name, schema_file in SCHEMA_FILES.items()
        }
        resources = [
            (
                schema["$id"],
                Resource.from_contents(schema, default_specification=DRAFT202012),
            )
            for schema in self.schemas.values()
        ]
        self.registry = Registry().with_resources(resources)
        self.format_checker = FormatChecker()

    def validator_for(self, schema_name: str) -> Draft202012Validator:
        return Draft202012Validator(
            self.schemas[schema_name],
            registry=self.registry,
            format_checker=self.format_checker,
        )

    def validate_record(self, schema_name: str, record: dict[str, Any]) -> None:
        self.validator_for(schema_name).validate(normalize_for_jsonschema(record))

    def validate_dataframe(
        self,
        schema_name: str,
        dataframe: pd.DataFrame,
        *,
        sample_limit: int | None = None,
    ) -> int:
        records = dataframe.to_dict(orient="records")
        if sample_limit is not None:
            records = records[:sample_limit]
        validator = self.validator_for(schema_name)
        for record in records:
            validator.validate(normalize_for_jsonschema(record))
        return len(records)


def normalize_for_jsonschema(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: normalize_for_jsonschema(child) for key, child in value.items()}
    if isinstance(value, list):
        return [normalize_for_jsonschema(item) for item in value]
    if isinstance(value, (tuple, np.ndarray)):
        return [normalize_for_jsonschema(item) for item in value]
    if isinstance(value, (pd.Timestamp, datetime)):
        return format_utc(value)
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, Decimal):
        return float(value)
    if isinstance(value, np.generic):
        return value.item()
    if not isinstance(value, (list, dict, str)):
        try:
            if pd.isna(value):
                return None
        except (TypeError, ValueError):
            pass
    return value


def assert_no_transaction_leakage(dataframe: pd.DataFrame) -> None:
    leakage = TRANSACTION_LEAKAGE_FIELDS.intersection(dataframe.columns)
    if leakage:
        raise ValueError(f"Customer transactions contain leakage fields: {sorted(leakage)}")
