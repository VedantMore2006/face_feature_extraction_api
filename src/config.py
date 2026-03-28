"""Configuration values for the facial analysis runtime."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class FeatureThresholdConfig:
    event_window_seconds: float = 60.0

    smile_threshold: float = 0.06
    mouth_open_threshold: float = 0.02
    lip_compression_threshold: float = 0.01
    brow_tension_threshold: float = 0.12

    blink_ear_threshold: float = 0.21
    blink_min_frames: int = 3
    ear_threshold_scale: float = 0.70
    ear_median_window: int = 5
    ear_drop_rate_threshold: float = 0.35
    ear_rise_rate_threshold: float = 0.30
    ear_min_drop: float = 0.015
    eye_asymmetry_ratio_threshold: float = 0.45
    gaze_shift_threshold: float = 0.012
    downward_gaze_threshold: float = 0.006

    motion_transition_threshold: float = 0.0025
    nod_velocity_threshold: float = 0.02
    extended_silence_threshold: float = 2.0


@dataclass(frozen=True)
class RuntimeConfig:
    camera_index: int = 0
    frame_width: int = 1280
    frame_height: int = 720
    max_faces: int = 1
    min_detection_confidence: float = 0.5
    min_tracking_confidence: float = 0.5
    baseline_seconds: float = 20.0  # 20 s: enough stats for normaliser, fast enough to reach protocol
    target_fps: float = 15.0
    smoothing_window: int = 5
    output_dir: str = "data"
    thresholds: FeatureThresholdConfig = FeatureThresholdConfig()

    @property
    def baseline_min_frames(self) -> int:
        return int(self.baseline_seconds * self.target_fps)

    def output_path(self, project_root: Path) -> Path:
        path = project_root / self.output_dir
        path.mkdir(parents=True, exist_ok=True)
        return path


def validate_runtime_config(cfg: RuntimeConfig) -> None:
    if cfg.frame_width <= 0 or cfg.frame_height <= 0:
        raise ValueError("Frame dimensions must be positive integers.")
    if cfg.max_faces <= 0:
        raise ValueError("max_faces must be >= 1.")
    if not (0.0 < cfg.min_detection_confidence <= 1.0):
        raise ValueError("min_detection_confidence must be in (0, 1].")
    if not (0.0 < cfg.min_tracking_confidence <= 1.0):
        raise ValueError("min_tracking_confidence must be in (0, 1].")
    if cfg.baseline_seconds <= 0:
        raise ValueError("baseline_seconds must be > 0.")
    if cfg.target_fps <= 0:
        raise ValueError("target_fps must be > 0.")
    if cfg.smoothing_window <= 0:
        raise ValueError("smoothing_window must be > 0.")

    t = cfg.thresholds
    if t.event_window_seconds <= 0:
        raise ValueError("event_window_seconds must be > 0.")
    if t.smile_threshold <= 0:
        raise ValueError("smile_threshold must be > 0.")
    if t.mouth_open_threshold <= 0:
        raise ValueError("mouth_open_threshold must be > 0.")
    if t.lip_compression_threshold <= 0:
        raise ValueError("lip_compression_threshold must be > 0.")
    if t.brow_tension_threshold <= 0:
        raise ValueError("brow_tension_threshold must be > 0.")
    if t.blink_ear_threshold <= 0:
        raise ValueError("blink_ear_threshold must be > 0.")
    if t.blink_min_frames <= 0:
        raise ValueError("blink_min_frames must be > 0.")
    if not (0.0 < t.ear_threshold_scale < 1.0):
        raise ValueError("ear_threshold_scale must be in (0, 1).")
    if t.ear_median_window <= 0:
        raise ValueError("ear_median_window must be > 0.")
    if t.ear_drop_rate_threshold <= 0:
        raise ValueError("ear_drop_rate_threshold must be > 0.")
    if t.ear_rise_rate_threshold <= 0:
        raise ValueError("ear_rise_rate_threshold must be > 0.")
    if t.ear_min_drop <= 0:
        raise ValueError("ear_min_drop must be > 0.")
    if not (0.0 < t.eye_asymmetry_ratio_threshold <= 1.0):
        raise ValueError("eye_asymmetry_ratio_threshold must be in (0, 1].")
    if t.gaze_shift_threshold <= 0:
        raise ValueError("gaze_shift_threshold must be > 0.")
    if t.downward_gaze_threshold <= 0:
        raise ValueError("downward_gaze_threshold must be > 0.")
    if t.motion_transition_threshold <= 0:
        raise ValueError("motion_transition_threshold must be > 0.")
    if t.nod_velocity_threshold <= 0:
        raise ValueError("nod_velocity_threshold must be > 0.")
    if t.extended_silence_threshold <= 0:
        raise ValueError("extended_silence_threshold must be > 0.")
