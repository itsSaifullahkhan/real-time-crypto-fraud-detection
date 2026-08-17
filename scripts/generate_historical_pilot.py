from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from crypto_fraud_platform.data_generator.common.config import load_historical_generator_config
from crypto_fraud_platform.data_generator.historical.orchestrator import (  # noqa: E402
    run_historical_pilot,
    validate_existing_pilot_output,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate the local historical pilot dataset.")
    parser.add_argument(
        "--config",
        default="config/historical-generator.yaml",
        help="Path to the historical generator YAML configuration.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing completed local pilot run.",
    )
    parser.add_argument(
        "--skip-market-download",
        action="store_true",
        help="Reuse previously downloaded Coinbase candle Parquet files.",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Read the existing pilot quality report without generating new data.",
    )
    parser.add_argument(
        "--output-root",
        help="Override the output root path from the YAML configuration.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_historical_generator_config(
        PROJECT_ROOT / args.config,
        output_root=args.output_root,
        overwrite=True if args.overwrite else None,
    )
    try:
        if args.validate_only:
            report = validate_existing_pilot_output(config=config)
            print(json.dumps({"status": report["status"], "quality_report": report}, indent=2))
            return 0 if report["status"] == "PASS" else 1

        result = run_historical_pilot(
            config=config,
            skip_market_download=args.skip_market_download,
        )
        summary = {
            "status": result.manifest["status"],
            "output_root": str(result.output_root),
            "record_counts": result.manifest["record_counts"],
            "fraud_count": result.manifest["fraud_count"],
            "fraud_rate": result.manifest["fraud_rate"],
            "scenario_distribution": result.manifest["scenario_distribution"],
            "dataset_paths": result.dataset_paths,
        }
        print(json.dumps(summary, indent=2))
        return 0
    except Exception as exc:
        print(f"Historical pilot generation failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
