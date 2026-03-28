"""Baseline collection and normalization utilities.

Pipeline normalization logic:
1) Collect baseline raw feature values for first N seconds
2) Compute per-feature mean and std
3) z-score normalize and sigmoid scale to [0, 1]
4) clamp and round to 8 decimals
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass, field
from typing import DefaultDict, Dict, Iterable, List, Mapping, Sequence


class NormalizationError(RuntimeError):
    """Raised when baseline or normalization state is invalid."""


@dataclass
class BaselineStats:
    mean: Dict[str, float] = field(default_factory=dict)
    std: Dict[str, float] = field(default_factory=dict)


@dataclass
class BaselineNormalizer:
    feature_order: Sequence[str]
    baseline_seconds: float
    # Minimum std used during normalisation.
    # 1e-6 (original) caused event-rate features with zero baseline to saturate;
    # 0.02 over-compressed continuous features with natural std 0.005-0.015.
    # 0.01 is a middle ground: keeps continuous-feature resolution while giving
    # event-rate features ~1 event/100 s of distinguishable resolution from zero.
    sigma_floor: float = 0.01

    _baseline_values: DefaultDict[str, List[float]] = field(default_factory=lambda: defaultdict(list))
    _start_timestamp: float | None = None
    _locked: bool = False
    _stats: BaselineStats = field(default_factory=BaselineStats)

    def reset(self) -> None:
        self._baseline_values.clear()
        self._start_timestamp = None
        self._locked = False
        self._stats = BaselineStats()

    @property
    def is_locked(self) -> bool:
        return self._locked

    @property
    def stats(self) -> BaselineStats:
        if not self._locked:
            raise NormalizationError("Baseline statistics requested before lock.")
        return self._stats

    def _ensure_features(self, raw_features: Mapping[str, float]) -> None:
        missing = [name for name in self.feature_order if name not in raw_features]
        if missing:
            raise NormalizationError(f"Raw feature vector missing keys: {missing}")

    @staticmethod
    def _mean(values: Iterable[float]) -> float:
        values = [float(v) for v in values]
        if not values:
            return 0.0
        return sum(values) / len(values)

    def _std(self, values: Sequence[float]) -> float:
        if len(values) < 2:
            return self.sigma_floor
        mu = self._mean(values)
        variance = self._mean([(v - mu) ** 2 for v in values])
        return max(math.sqrt(variance), self.sigma_floor)

    @staticmethod
    def _sigmoid(z: float) -> float:
        # Stable sigmoid for large-magnitude z.
        if z >= 0:
            ez = math.exp(-z)
            return 1.0 / (1.0 + ez)
        ez = math.exp(z)
        return ez / (1.0 + ez)

    @staticmethod
    def _clip01(value: float) -> float:
        return min(1.0, max(0.0, value))

    def _compute_stats(self) -> None:
        mean_dict: Dict[str, float] = {}
        std_dict: Dict[str, float] = {}
        for feature_name in self.feature_order:
            values = self._baseline_values[feature_name]
            mean_dict[feature_name] = self._mean(values)
            std_dict[feature_name] = self._std(values)
        self._stats = BaselineStats(mean=mean_dict, std=std_dict)

    def update_and_maybe_lock(self, timestamp: float, raw_features: Mapping[str, float]) -> bool:
        """Collect baseline samples until baseline duration elapses.

        Returns True only when baseline becomes locked on this call.
        """
        self._ensure_features(raw_features)

        if self._locked:
            return False

        if self._start_timestamp is None:
            self._start_timestamp = float(timestamp)

        for feature_name in self.feature_order:
            self._baseline_values[feature_name].append(float(raw_features[feature_name]))

        elapsed = float(timestamp) - self._start_timestamp
        if elapsed >= self.baseline_seconds:
            self._compute_stats()
            self._locked = True
            return True

        return False

    def normalize(self, raw_features: Mapping[str, float]) -> Dict[str, float]:
        """Normalize raw features into [0,1] with 8 decimal precision."""
        self._ensure_features(raw_features)

        if not self._locked:
            raise NormalizationError("Baseline is not locked; cannot normalize yet.")

        output: Dict[str, float] = {}
        for feature_name in self.feature_order:
            x = float(raw_features[feature_name])
            mu = self._stats.mean[feature_name]
            sigma = max(self._stats.std[feature_name], self.sigma_floor)

            z = (x - mu) / sigma
            scaled = self._sigmoid(z)
            output[feature_name] = round(self._clip01(scaled), 8)

        return output
