from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from crypto_fraud_platform.data_generator.common.config import HistoricalGeneratorConfig
from crypto_fraud_platform.data_generator.common.identifiers import DeterministicIdFactory
from crypto_fraud_platform.data_generator.common.parquet_io import (
    read_json,
    read_parquet_dataset,
    write_json,
    write_parquet_file,
    write_partitioned_parquet,
)
from crypto_fraud_platform.data_generator.common.schema_validation import SchemaValidator
from crypto_fraud_platform.data_generator.common.time_utils import date_string, format_utc, utc_now
from crypto_fraud_platform.data_generator.fraud_scenarios.scenario_engine import ScenarioEngine
from crypto_fraud_platform.data_generator.historical.authentication_generator import (
    generate_normal_authentication_events,
)
from crypto_fraud_platform.data_generator.historical.entity_generator import generate_entities
from crypto_fraud_platform.data_generator.historical.label_generator import generate_fraud_labels
from crypto_fraud_platform.data_generator.historical.market_candle_downloader import (
    MarketCandleDownloader,
    calculate_missing_intervals,
)
from crypto_fraud_platform.data_generator.historical.quality_report import (
    assert_quality_passed,
    build_quality_report,
)
from crypto_fraud_platform.data_generator.historical.transaction_generator import (
    MarketPriceLookup,
    generate_normal_transactions,
    generate_scenario_activity,
)


@dataclass(frozen=True)
class PilotRunResult:
    output_root: Path
    manifest: dict[str, Any]
    quality_report: dict[str, Any]
    dataset_paths: dict[str, str]


def run_historical_pilot(
    *,
    config: HistoricalGeneratorConfig,
    skip_market_download: bool = False,
) -> PilotRunResult:
    output_root = config.output.root_path
    metadata_root = output_root / "_metadata"
    audit_root = output_root / "_audit"
    manifest_path = metadata_root / "generation_manifest.json"
    generation_started_at = utc_now()
    preserved_market_candles: pd.DataFrame | None = None

    if skip_market_download and config.output.overwrite_existing:
        market_root = output_root / "market_candles"
        if market_root.exists():
            preserved_market_candles = read_parquet_dataset(market_root).drop(
                columns=["event_date"],
                errors="ignore",
            )

    if manifest_path.exists() and not config.output.overwrite_existing:
        raise FileExistsError(
            f"Existing completed pilot run found at {output_root}. Use --overwrite to replace it."
        )
    if config.output.overwrite_existing and output_root.exists():
        shutil.rmtree(output_root)
    metadata_root.mkdir(parents=True, exist_ok=True)
    audit_root.mkdir(parents=True, exist_ok=True)

    schema_validator = SchemaValidator(config.project_root)
    rng = np.random.default_rng(config.random_seed)
    id_factory = DeterministicIdFactory(seed=config.random_seed)

    market_candles, market_download_report = _load_or_download_market_candles(
        config=config,
        skip_market_download=skip_market_download,
        metadata_root=metadata_root,
        preserved_market_candles=preserved_market_candles,
    )
    market_candles = _with_event_date(market_candles, "candle_start_timestamp")
    market_lookup = MarketPriceLookup(
        market_candles.drop(columns=["event_date"], errors="ignore"),
        config.market_products,
    )

    entities = generate_entities(
        account_count=config.scale.account_count,
        start_timestamp=config.time_range.start,
        end_timestamp=config.time_range.end,
        rng=rng,
        id_factory=id_factory,
    )
    scenario_engine = ScenarioEngine(
        project_root=config.project_root,
        rng=rng,
        optional_volatility_enabled=config.generation.optional_volatility_scenario_enabled,
    )
    scenario_result = generate_scenario_activity(
        config=config,
        entities=entities,
        market_lookup=market_lookup,
        scenario_engine=scenario_engine,
        rng=rng,
        id_factory=id_factory,
    )
    authentication_events = generate_normal_authentication_events(
        config=config,
        entities=entities,
        existing_events=scenario_result.authentication_events,
        rng=rng,
        id_factory=id_factory,
    )
    customer_transactions, market_matches = generate_normal_transactions(
        config=config,
        entities=entities,
        market_lookup=market_lookup,
        existing_transactions=scenario_result.transactions,
        existing_market_matches=scenario_result.market_matches,
        rng=rng,
        id_factory=id_factory,
    )
    fraud_labels = generate_fraud_labels(
        config=config,
        transactions=customer_transactions,
        scenario_fraud_rows=scenario_result.fraud_rows,
        rng=rng,
        id_factory=id_factory,
    )
    scenario_audit = pd.DataFrame(scenario_result.audit_rows)
    if not scenario_audit.empty:
        scenario_audit["event_timestamp"] = pd.to_datetime(scenario_audit["event_timestamp"], utc=True)

    quality_report = build_quality_report(
        config=config,
        accounts=entities.accounts,
        devices=entities.devices,
        wallets=entities.wallets,
        authentication_events=authentication_events,
        customer_transactions=customer_transactions,
        fraud_labels=fraud_labels,
        market_candles=market_candles,
        scenario_audit=scenario_audit,
        market_matches=market_matches,
        market_download_report=market_download_report,
        schema_validator=schema_validator,
    )
    assert_quality_passed(quality_report)

    dataset_paths = _write_outputs(
        config=config,
        market_candles=market_candles,
        accounts=entities.accounts,
        devices=entities.devices,
        wallets=entities.wallets,
        authentication_events=authentication_events,
        customer_transactions=customer_transactions,
        fraud_labels=fraud_labels,
        scenario_audit=scenario_audit,
        market_matches=market_matches,
    )

    generation_completed_at = utc_now()
    manifest = _build_manifest(
        config=config,
        generation_started_at=generation_started_at,
        generation_completed_at=generation_completed_at,
        dataset_paths=dataset_paths,
        quality_report=quality_report,
        scenario_engine=scenario_engine,
        market_download_report=market_download_report,
    )
    write_json(metadata_root / "quality_report.json", quality_report)
    write_json(manifest_path, manifest)
    return PilotRunResult(
        output_root=output_root,
        manifest=manifest,
        quality_report=quality_report,
        dataset_paths=dataset_paths,
    )


def validate_existing_pilot_output(*, config: HistoricalGeneratorConfig) -> dict[str, Any]:
    output_root = config.output.root_path
    if not output_root.exists():
        raise FileNotFoundError(f"Pilot output root does not exist: {output_root}")
    quality_path = output_root / "_metadata" / "quality_report.json"
    if quality_path.exists():
        return read_json(quality_path)
    raise FileNotFoundError(f"Quality report does not exist: {quality_path}")


def _load_or_download_market_candles(
    *,
    config: HistoricalGeneratorConfig,
    skip_market_download: bool,
    metadata_root: Path,
    preserved_market_candles: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    market_root = config.output.root_path / "market_candles"
    if skip_market_download:
        if preserved_market_candles is not None:
            candles = preserved_market_candles
        elif market_root.exists():
            candles = read_parquet_dataset(market_root)
        else:
            raise FileNotFoundError(
                "--skip-market-download requires existing market candle Parquet files"
            )
        products = set(candles["product_id"])
        expected_products = set(config.market_products.values())
        if products != expected_products:
            raise ValueError(f"Existing market candles contain {products}, expected {expected_products}")
        product_reports = {}
        total_missing = 0
        for product_id in sorted(products):
            product_candles = candles[candles["product_id"] == product_id]
            missing = calculate_missing_intervals(
                product_candles,
                product_id=product_id,
                start=config.time_range.start,
                end=config.time_range.end,
                granularity_seconds=60,
            )
            total_missing += len(missing)
            product_reports[product_id] = {
                "candle_count": int(len(product_candles)),
                "missing_interval_count": int(len(missing)),
                "missing_interval_samples": [format_utc(value) for value in missing[:20]],
            }
        checkpoint = {
            "source": "existing_local_parquet",
            "completed_windows": {},
            "products": product_reports,
            "total_candles": int(len(candles)),
            "total_missing_intervals": int(total_missing),
        }
        write_json(metadata_root / "market_download_checkpoint.json", checkpoint)
        return candles.drop(columns=["event_date"], errors="ignore"), checkpoint

    downloader = MarketCandleDownloader()
    return downloader.download(
        products=list(config.market_products.values()),
        start=config.time_range.start,
        end=config.time_range.end,
        checkpoint_path=metadata_root / "market_download_checkpoint.json",
        granularity_seconds=60,
        max_candles_per_request=300,
    )


def _write_outputs(
    *,
    config: HistoricalGeneratorConfig,
    market_candles: pd.DataFrame,
    accounts: pd.DataFrame,
    devices: pd.DataFrame,
    wallets: pd.DataFrame,
    authentication_events: pd.DataFrame,
    customer_transactions: pd.DataFrame,
    fraud_labels: pd.DataFrame,
    scenario_audit: pd.DataFrame,
    market_matches: pd.DataFrame,
) -> dict[str, str]:
    root = config.output.root_path
    write_partitioned_parquet(
        market_candles,
        root / "market_candles",
        partition_columns=["product_id", "event_date"],
        overwrite=True,
    )
    write_parquet_file(accounts, root / "accounts" / "accounts.parquet", overwrite=True)
    write_parquet_file(devices, root / "devices" / "devices.parquet", overwrite=True)
    write_parquet_file(wallets, root / "wallets" / "wallets.parquet", overwrite=True)
    write_partitioned_parquet(
        _with_event_date(authentication_events, "event_timestamp"),
        root / "authentication_events",
        partition_columns=["event_date"],
        overwrite=True,
    )
    write_partitioned_parquet(
        _with_event_date(customer_transactions, "event_timestamp"),
        root / "customer_transactions",
        partition_columns=["event_date"],
        overwrite=True,
    )
    write_partitioned_parquet(
        _with_label_date(fraud_labels),
        root / "fraud_labels",
        partition_columns=["label_date"],
        overwrite=True,
    )
    write_parquet_file(
        scenario_audit,
        root / "_audit" / "scenario_assignments.parquet",
        overwrite=True,
    )
    write_parquet_file(
        market_matches,
        root / "_audit" / "market_enrichment_matches.parquet",
        overwrite=True,
    )
    return {
        "market_candles": str(root / "market_candles"),
        "accounts": str(root / "accounts" / "accounts.parquet"),
        "devices": str(root / "devices" / "devices.parquet"),
        "wallets": str(root / "wallets" / "wallets.parquet"),
        "authentication_events": str(root / "authentication_events"),
        "customer_transactions": str(root / "customer_transactions"),
        "fraud_labels": str(root / "fraud_labels"),
        "scenario_assignments": str(root / "_audit" / "scenario_assignments.parquet"),
        "market_enrichment_matches": str(root / "_audit" / "market_enrichment_matches.parquet"),
    }


def _build_manifest(
    *,
    config: HistoricalGeneratorConfig,
    generation_started_at: Any,
    generation_completed_at: Any,
    dataset_paths: dict[str, str],
    quality_report: dict[str, Any],
    scenario_engine: ScenarioEngine,
    market_download_report: dict[str, Any],
) -> dict[str, Any]:
    return {
        "status": "success",
        "generator_version": config.generator_version,
        "configuration_path": str(config.config_path),
        "random_seed": config.random_seed,
        "start_timestamp": format_utc(config.time_range.start),
        "end_timestamp": format_utc(config.time_range.end),
        "generation_started_at": format_utc(generation_started_at),
        "generation_completed_at": format_utc(generation_completed_at),
        "dataset_names": list(dataset_paths),
        "record_counts": quality_report["record_counts"],
        "output_paths": dataset_paths,
        "schema_versions": {
            "account": "1.0",
            "device": "1.0",
            "wallet": "1.0",
            "authentication_event": "1.0",
            "customer_transaction": "1.0",
            "fraud_label": "1.0",
            "historical_market_candle": "1.0",
        },
        "enabled_fraud_scenarios": scenario_engine.enabled_scenario_ids,
        "fraud_count": quality_report["fraud_distribution"]["confirmed_fraud_count"],
        "fraud_rate": quality_report["fraud_distribution"]["fraud_rate"],
        "scenario_distribution": quality_report["fraud_distribution"]["scenario_distribution"],
        "coinbase_products": list(config.market_products.values()),
        "market_candle_counts": {
            product_id: product_report["candle_count"]
            for product_id, product_report in market_download_report["products"].items()
        },
        "missing_candle_intervals": market_download_report.get("total_missing_intervals"),
        "code_version": None,
    }


def _with_event_date(dataframe: pd.DataFrame, timestamp_column: str) -> pd.DataFrame:
    with_date = dataframe.copy()
    with_date["event_date"] = with_date[timestamp_column].apply(date_string)
    return with_date


def _with_label_date(dataframe: pd.DataFrame) -> pd.DataFrame:
    with_date = dataframe.copy()
    with_date["label_date"] = with_date["label_timestamp"].apply(date_string)
    return with_date
