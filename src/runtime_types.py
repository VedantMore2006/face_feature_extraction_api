"""Common runtime data structures shared by pipeline modules."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

LandmarkPoint = Tuple[float, float, float]


@dataclass(frozen=True)
class FrameData:
    timestamp: float
    landmarks: List[LandmarkPoint]


@dataclass(frozen=True)
class PipelineStatus:
    phase: str
    message: str


FeatureDict = Dict[str, float]
