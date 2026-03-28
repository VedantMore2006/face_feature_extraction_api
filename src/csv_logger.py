"""CSV output logger for normalized feature vectors."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Sequence


class CsvLoggerError(RuntimeError):
    """Raised when CSV output cannot be initialized or written."""


@dataclass
class FeatureCsvLogger:
    output_dir: Path
    feature_order: Sequence[str]
    allow_raw_features: bool = False  # Set True for raw (unnormalized) features, False for [0,1] normalized

    def __post_init__(self) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.path = self.output_dir / f"features_{timestamp}.csv"

        self._file_handle = self.path.open("w", encoding="utf-8", newline="")
        self._writer = csv.DictWriter(
            self._file_handle,
            fieldnames=["timestamp", *self.feature_order],
            extrasaction="raise",
        )
        self._writer.writeheader()
        self._file_handle.flush()

    def write_row(self, timestamp: float, normalized_features: Dict[str, float]) -> None:
        missing = [name for name in self.feature_order if name not in normalized_features]
        if missing:
            raise CsvLoggerError(f"Cannot write CSV row; missing features: {missing}")

        row = {"timestamp": round(float(timestamp), 8)}
        for feature_name in self.feature_order:
            value = float(normalized_features[feature_name])
            # Only enforce [0,1] bounds if writing normalized features
            if not self.allow_raw_features and (value < 0.0 or value > 1.0):
                raise CsvLoggerError(
                    f"Feature '{feature_name}' out of [0,1] range: {value}"
                )
            row[feature_name] = round(value, 8)

        self._writer.writerow(row)
        self._file_handle.flush()

    def close(self) -> None:
        if getattr(self, "_file_handle", None) is not None:
            self._file_handle.close()
            self._file_handle = None

    def __enter__(self) -> "FeatureCsvLogger":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()
