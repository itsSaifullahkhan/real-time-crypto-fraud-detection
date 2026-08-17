from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from crypto_fraud_platform.data_generator.common.time_utils import parse_utc


@dataclass(frozen=True)
class TimeRange:
    start: Any
    end: Any
    timezone: str


@dataclass(frozen=True)
class ScaleConfig:
    account_count: int
    target_transaction_count: int
    target_authentication_event_count: int
    target_fraud_rate: float


@dataclass(frozen=True)
class OutputConfig:
    root_path: Path
    format: str
    overwrite_existing: bool


@dataclass(frozen=True)
class ValidationConfig:
    validate_schema_samples: bool
    enforce_foreign_keys: bool
    enforce_unique_ids: bool
    enforce_timestamp_rules: bool


@dataclass(frozen=True)
class GenerationConfig:
    label_completed_transactions: bool
    include_failed_transactions: bool
    optional_volatility_scenario_enabled: bool


@dataclass(frozen=True)
class HistoricalGeneratorConfig:
    generator_version: str
    mode: str
    random_seed: int
    time_range: TimeRange
    scale: ScaleConfig
    assets: list[str]
    market_products: dict[str, str]
    output: OutputConfig
    validation: ValidationConfig
    generation: GenerationConfig
    config_path: Path
    project_root: Path


def project_root_from_config(config_path: Path) -> Path:
    return config_path.resolve().parents[1]


def load_yaml_file(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as file:
        data = yaml.safe_load(file)
    if not isinstance(data, dict):
        raise ValueError(f"Expected YAML mapping in {path}")
    return data


def load_historical_generator_config(
    config_path: str | Path,
    *,
    output_root: str | Path | None = None,
    overwrite: bool | None = None,
) -> HistoricalGeneratorConfig:
    path = Path(config_path)
    data = load_yaml_file(path)
    project_root = project_root_from_config(path)

    output_data = data["output"].copy()
    if output_root is not None:
        output_data["root_path"] = str(output_root)
    if overwrite is not None:
        output_data["overwrite_existing"] = overwrite

    root_path = Path(output_data["root_path"])
    if not root_path.is_absolute():
        root_path = project_root / root_path

    return HistoricalGeneratorConfig(
        generator_version=str(data["generator_version"]),
        mode=str(data["mode"]),
        random_seed=int(data["random_seed"]),
        time_range=TimeRange(
            start=parse_utc(data["time_range"]["start"]),
            end=parse_utc(data["time_range"]["end"]),
            timezone=str(data["time_range"]["timezone"]),
        ),
        scale=ScaleConfig(
            account_count=int(data["scale"]["account_count"]),
            target_transaction_count=int(data["scale"]["target_transaction_count"]),
            target_authentication_event_count=int(
                data["scale"]["target_authentication_event_count"]
            ),
            target_fraud_rate=float(data["scale"]["target_fraud_rate"]),
        ),
        assets=list(data["assets"]),
        market_products=dict(data["market_products"]),
        output=OutputConfig(
            root_path=root_path,
            format=str(output_data["format"]),
            overwrite_existing=bool(output_data["overwrite_existing"]),
        ),
        validation=ValidationConfig(**data["validation"]),
        generation=GenerationConfig(**data["generation"]),
        config_path=path,
        project_root=project_root,
    )
