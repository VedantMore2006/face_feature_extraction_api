"""Runtime contract loaders for frozen docs definitions.

These helpers keep JSON contract parsing centralized and defensive.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Mapping


class ContractError(ValueError):
    """Raised when a project contract file is missing or malformed."""


@dataclass(frozen=True)
class FeatureContract:
    schema_version: str
    frozen_on: str
    feature_count: int
    feature_order: List[str]
    definitions: Dict[str, Dict[str, object]]


def _read_json(path: Path) -> Mapping[str, object]:
    if not path.exists():
        raise ContractError(f"Contract file not found: {path}")

    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ContractError(f"Invalid JSON in contract file {path}: {exc}") from exc
    except OSError as exc:
        raise ContractError(f"Unable to read contract file {path}: {exc}") from exc


def load_feature_contract(project_root: Path) -> FeatureContract:
    path = project_root / "docs" / "feature_definitions_v1.json"
    payload = _read_json(path)

    required_fields = ["schema_version", "frozen_on", "feature_count", "feature_order", "definitions"]
    missing = [name for name in required_fields if name not in payload]
    if missing:
        raise ContractError(f"Feature contract missing required fields: {missing}")

    feature_order = payload["feature_order"]
    definitions = payload["definitions"]

    if not isinstance(feature_order, list) or not all(isinstance(x, str) for x in feature_order):
        raise ContractError("feature_order must be a list of strings.")

    if not isinstance(definitions, dict):
        raise ContractError("definitions must be a dictionary.")

    if len(feature_order) != int(payload["feature_count"]):
        raise ContractError(
            "feature_count mismatch: "
            f"declared {payload['feature_count']}, got {len(feature_order)} in feature_order."
        )

    missing_defs = [name for name in feature_order if name not in definitions]
    if missing_defs:
        raise ContractError(f"Missing feature definitions for: {missing_defs}")

    return FeatureContract(
        schema_version=str(payload["schema_version"]),
        frozen_on=str(payload["frozen_on"]),
        feature_count=int(payload["feature_count"]),
        feature_order=list(feature_order),
        definitions={str(k): v for k, v in definitions.items()},
    )
