#!/usr/bin/env python3
"""Entry-point for the Facial Analysis runtime pipeline.

Current status:
- Contracts and region mapping are locked and validated.
- Runtime scaffolding is in place for next build steps.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
import os
os.environ['QT_QPA_PLATFORM'] = 'xcb'
from src.config import RuntimeConfig, validate_runtime_config
from src.contracts import ContractError, load_feature_contract
from src.pipeline import PipelineError, run_raw_extraction_pipeline
from src.regions import RegionMappingError, validate_runtime_contract


def bootstrap(project_root: Path):
    cfg = RuntimeConfig()
    validate_runtime_config(cfg)

    contract = load_feature_contract(project_root)
    region_report = validate_runtime_contract()

    print("[OK] Runtime bootstrap complete")
    print(f"[OK] Feature contract version: {contract.schema_version}")
    print(f"[OK] Frozen features: {contract.feature_count}")
    print(
        "[OK] Region mapping: "
        f"{region_report.region_count} regions, "
        f"{region_report.unique_assigned} unique ids"
    )
    return contract


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Facial analysis pipeline runner")
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Validate contracts and exit without opening webcam.",
    )
    return parser.parse_args()


def main() -> int:
    project_root = Path(__file__).resolve().parent
    args = parse_args()

    try:
        cfg = RuntimeConfig()
        validate_runtime_config(cfg)
        contract = bootstrap(project_root)

        if args.check_only:
            print("[OK] Check-only mode complete.")
            return 0

        run_raw_extraction_pipeline(cfg, contract.feature_order, project_root)
        return 0
    except (ValueError, ContractError, RegionMappingError, PipelineError) as exc:
        print(f"[ERROR] {exc}")
        return 2
    except Exception as exc:  # defensive catch for unexpected failures
        print(f"[ERROR] Unexpected bootstrap failure: {exc}")
        return 99


if __name__ == "__main__":
    sys.exit(main())
