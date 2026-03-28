"""Stateful raw feature computation for all frozen behavioral features.

This module intentionally outputs RAW values (not normalized yet).
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from statistics import mean, median
from typing import Deque, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from src.regions import LEFT_EYE_EAR, RIGHT_EYE_EAR, extract_regions, flatten_region_ids
from src.runtime_types import FeatureDict, FrameData, LandmarkPoint


class FeatureEngineError(RuntimeError):
    """Raised for unrecoverable feature computation issues."""


@dataclass(frozen=True)
class FeatureEngineConfig:
    smoothing_window: int = 5
    event_window_seconds: float = 60.0
    # Shorter window for high-frequency events (gesture, gaze_shift, transition).
    # A 10 s window means the rate fluctuates naturally during the 30 s baseline
    # as events enter and leave, giving the normaliser a real mean/sigma rather
    # than a near-zero baseline from the slow accumulation of a 60 s window.
    short_event_window_seconds: float = 10.0

    smile_threshold: float = 0.06
    mouth_open_threshold: float = 0.02
    # Speaking threshold: higher than mouth_open so resting mouth doesn't trigger speech.
    speaking_threshold: float = 0.04
    # Lip compression: tight absolute floor only.  Resting closed mouth is ~0.018-0.025;
    # genuine active lip press drops to ~0.005-0.012.  This threshold only fires on
    # deliberate presses — not normal speech→close cycles.  The feature will sit near
    # 0.5 in neutral sessions (correct: it's a rare clinical marker, not baseline behaviour).
    lip_compression_threshold: float = 0.010
    brow_tension_threshold: float = 0.12  # retained for future use; AU4 now stores continuous activation
    # AU20 (lip stretch) threshold — separate from mouth_open to avoid false fires.
    au20_threshold: float = 0.055
    # Eye contact: soft-threshold scale — how many IPD units = zero contact score.
    gaze_contact_threshold: float = 0.15  # raised from 0.10; used as soft-scale, not hard gate

    blink_ear_threshold: float = 0.21
    blink_min_frames: int = 2  # relaxed from 3 — allows 2-frame blinks at 15 FPS
    ear_threshold_scale: float = 0.70
    ear_median_window: int = 5
    ear_drop_rate_threshold: float = 0.25  # relaxed from 0.35
    ear_rise_rate_threshold: float = 0.20  # relaxed from 0.30
    ear_min_drop: float = 0.015
    eye_asymmetry_ratio_threshold: float = 0.45
    # Blink cluster: inter-blink interval ≤ window is "burst blink" (anxiety/cognitive load).
    blink_cluster_window: float = 2.0    # raised from 1.0 — normal blinks are ~4-5 s apart
    gaze_shift_threshold: float = 0.30   # raised from 0.20 — only significant deliberate saccades
    # Minimum inter-event gap for gaze-shift events.
    gaze_shift_cooldown_seconds: float = 0.30  # raised from 0.15 — max ~3 shifts/s
    # Downward gaze: threshold for significant avoidance gaze (lap/floor, not screen).
    # Camera users naturally look slightly down; 0.06 IPD = notable downward shift.
    downward_gaze_threshold: float = 0.06
    # Minimum inter-event gap for gesture (nod) detection — prevents rapid re-fires
    # on continuous chin motion during speech articulation.
    gesture_cooldown_seconds: float = 1.0   # raised from 0.4 — allow ≤1 nod/s max

    motion_transition_threshold: float = 0.120   # raised from 0.080 — only major expression shifts
    # Minimum inter-event gap for facial-transition events.
    facial_transition_cooldown_seconds: float = 1.0  # raised from 0.50 — max ~1 macro-shift/s
    nod_velocity_threshold: float = 0.40          # deliberate nods only
    extended_silence_threshold: float = 2.0
    # Max seconds after speech onset to still count a nod latency.
    nod_latency_window: float = 5.0


@dataclass
class FeatureEngineState:
    prev_timestamp: Optional[float] = None
    prev_landmarks: Optional[List[LandmarkPoint]] = None

    prev_smile_active: bool = False
    prev_lip_compressed: bool = False
    prev_au20_active: bool = False
    prev_blink_active: bool = False
    prev_gaze_anchor: Optional[Tuple[float, float]] = None
    prev_downward_active: bool = False
    prev_expression_score: Optional[float] = None
    prev_speaking: bool = False
    last_blink_timestamp: Optional[float] = None
    last_adaptive_blink_threshold: float = 0.21
    blink_started_at: Optional[float] = None
    blink_active_frames: int = 0
    blink_min_ear: Optional[float] = None
    blink_peak_drop_rate: float = 0.0
    blink_peak_rise_rate: float = 0.0
    prev_eye_openness_smoothed: Optional[float] = None
    open_eye_baseline: Deque[float] = field(default_factory=lambda: deque(maxlen=300))
    ear_median_buffer: Deque[float] = field(default_factory=lambda: deque(maxlen=5))

    pause_started_at: Optional[float] = None
    last_speech_onset_at: Optional[float] = None
    last_gesture_timestamp: Optional[float] = None
    last_gaze_shift_timestamp: Optional[float] = None
    last_transition_timestamp: Optional[float] = None

    smile_events: Deque[float] = field(default_factory=lambda: deque(maxlen=600))
    au20_events: Deque[float] = field(default_factory=lambda: deque(maxlen=600))
    lip_compression_events: Deque[float] = field(default_factory=lambda: deque(maxlen=600))
    transition_events: Deque[float] = field(default_factory=lambda: deque(maxlen=1200))
    blink_events: Deque[float] = field(default_factory=lambda: deque(maxlen=1200))
    blink_cluster_events: Deque[float] = field(default_factory=lambda: deque(maxlen=1200))
    blink_durations: Deque[float] = field(default_factory=lambda: deque(maxlen=1200))
    downward_gaze_events: Deque[float] = field(default_factory=lambda: deque(maxlen=1200))
    gaze_shift_events: Deque[float] = field(default_factory=lambda: deque(maxlen=1200))
    gesture_events: Deque[float] = field(default_factory=lambda: deque(maxlen=1200))

    expression_history: Deque[float] = field(default_factory=lambda: deque(maxlen=300))
    mouth_open_history: Deque[float] = field(default_factory=lambda: deque(maxlen=90))
    brow_tension_history: Deque[float] = field(default_factory=lambda: deque(maxlen=300))
    head_velocity_history: Deque[float] = field(default_factory=lambda: deque(maxlen=300))
    motion_energy_history: Deque[float] = field(default_factory=lambda: deque(maxlen=300))
    eye_openness_history: Deque[float] = field(default_factory=lambda: deque(maxlen=300))
    eye_contact_flags: Deque[float] = field(default_factory=lambda: deque(maxlen=300))
    gaze_horizontal_history: Deque[float] = field(default_factory=lambda: deque(maxlen=300))
    gaze_vertical_history: Deque[float] = field(default_factory=lambda: deque(maxlen=300))

    pause_durations: Deque[float] = field(default_factory=lambda: deque(maxlen=300))
    response_latencies: Deque[float] = field(default_factory=lambda: deque(maxlen=300))
    nod_latencies: Deque[float] = field(default_factory=lambda: deque(maxlen=300))

    smoothing_buffers: Dict[str, Deque[float]] = field(default_factory=dict)


def _safe_mean(values: Iterable[float], default: float = 0.0) -> float:
    values_list = [float(v) for v in values]
    return mean(values_list) if values_list else default


def _safe_variance(values: Sequence[float]) -> float:
    if len(values) < 2:
        return 0.0
    m = _safe_mean(values)
    return _safe_mean([(v - m) ** 2 for v in values])


def _distance(a: LandmarkPoint, b: LandmarkPoint) -> float:
    dx = float(a[0]) - float(b[0])
    dy = float(a[1]) - float(b[1])
    dz = float(a[2]) - float(b[2])
    return math.sqrt(dx * dx + dy * dy + dz * dz)


def _distance_xy(a: LandmarkPoint, b: LandmarkPoint) -> float:
    dx = float(a[0]) - float(b[0])
    dy = float(a[1]) - float(b[1])
    return math.sqrt(dx * dx + dy * dy)


def _centroid(points: Sequence[LandmarkPoint]) -> LandmarkPoint:
    if not points:
        raise FeatureEngineError("Cannot compute centroid of empty point list.")
    return (
        _safe_mean([p[0] for p in points]),
        _safe_mean([p[1] for p in points]),
        _safe_mean([p[2] for p in points]),
    )


def _event_rate(events: Deque[float], now_ts: float, window_seconds: float) -> float:
    while events and (now_ts - events[0]) > window_seconds:
        events.popleft()
    if window_seconds <= 0:
        return 0.0
    return float(len(events)) / float(window_seconds)


def _ear(landmarks: Sequence[LandmarkPoint], eye_map: Mapping[str, int]) -> float:
    p1 = landmarks[eye_map["p1_outer"]]
    p2 = landmarks[eye_map["p2_upper1"]]
    p3 = landmarks[eye_map["p3_upper2"]]
    p4 = landmarks[eye_map["p4_inner"]]
    p5 = landmarks[eye_map["p5_lower1"]]
    p6 = landmarks[eye_map["p6_lower2"]]

    horiz = _distance_xy(p1, p4)
    if horiz <= 1e-9:
        return 0.0
    return (_distance_xy(p2, p6) + _distance_xy(p3, p5)) / (2.0 * horiz)


def _eye_horizontal_span(landmarks: Sequence[LandmarkPoint], eye_map: Mapping[str, int]) -> float:
    p1 = landmarks[eye_map["p1_outer"]]
    p4 = landmarks[eye_map["p4_inner"]]
    return _distance_xy(p1, p4)


def _all_displacements(
    curr: Sequence[LandmarkPoint],
    prev: Sequence[LandmarkPoint],
    ids: Sequence[int],
) -> List[float]:
    displacements: List[float] = []
    for idx in ids:
        if idx < len(curr) and idx < len(prev):
            displacements.append(_distance(curr[idx], prev[idx]))
    return displacements


class FeatureEngine:
    """Compute the 33 raw features for each frame with resilient fallbacks."""

    def __init__(self, feature_order: Sequence[str], cfg: FeatureEngineConfig) -> None:
        if not feature_order:
            raise FeatureEngineError("feature_order cannot be empty.")
        if cfg.smoothing_window <= 0:
            raise FeatureEngineError("smoothing_window must be > 0.")
        if cfg.blink_min_frames <= 0:
            raise FeatureEngineError("blink_min_frames must be > 0.")
        if cfg.ear_median_window <= 0:
            raise FeatureEngineError("ear_median_window must be > 0.")
        if not (0.0 < cfg.ear_threshold_scale < 1.0):
            raise FeatureEngineError("ear_threshold_scale must be in (0, 1).")

        self.feature_order = list(feature_order)
        self.cfg = cfg
        self.state = FeatureEngineState()
        self.state.ear_median_buffer = deque(maxlen=cfg.ear_median_window)
        self.region_landmark_ids = flatten_region_ids()

        for feature_name in self.feature_order:
            self.state.smoothing_buffers[feature_name] = deque(maxlen=cfg.smoothing_window)

    def _smooth(self, feature_name: str, value: float) -> float:
        buffer_ = self.state.smoothing_buffers[feature_name]
        buffer_.append(float(value))
        return _safe_mean(buffer_)

    def _region_centroid(self, regions: Mapping[str, List[LandmarkPoint]], name: str) -> LandmarkPoint:
        if name not in regions or not regions[name]:
            raise FeatureEngineError(f"Missing region points for '{name}'.")
        return _centroid(regions[name])

    def compute(self, frame_data: FrameData) -> FeatureDict:
        if not frame_data.landmarks:
            raise FeatureEngineError("Empty landmarks in frame_data.")

        now_ts = float(frame_data.timestamp)
        dt = 1.0 / 15.0
        if self.state.prev_timestamp is not None:
            dt = max(1e-6, now_ts - self.state.prev_timestamp)

        try:
            regions = extract_regions(frame_data.landmarks, strict=True)
        except Exception as exc:
            raise FeatureEngineError(f"Region extraction failed in feature engine: {exc}") from exc

        # Geometric primitives
        left_lip = self._region_centroid(regions, "left_lip_corner")
        right_lip = self._region_centroid(regions, "right_lip_corner")
        upper_lip = self._region_centroid(regions, "upperlip_border")
        lower_lip = self._region_centroid(regions, "lowerlip_border")
        left_brow = self._region_centroid(regions, "left_eyebrow")
        right_brow = self._region_centroid(regions, "right_eyebrow")
        between_brows = self._region_centroid(regions, "between_eyebrows")
        forehead = self._region_centroid(regions, "forehead")
        nose_tip = self._region_centroid(regions, "nose_tip")
        nose_bridge = self._region_centroid(regions, "nose_bridge")
        chin = self._region_centroid(regions, "chin")
        left_cheek = self._region_centroid(regions, "left_cheek")
        right_cheek = self._region_centroid(regions, "right_cheek")
        moustache = self._region_centroid(regions, "moustache")
        left_iris = self._region_centroid(regions, "left_iris")
        right_iris = self._region_centroid(regions, "right_iris")
        left_eye_center = self._region_centroid(regions, "left_eye")
        right_eye_center = self._region_centroid(regions, "right_eye")

        mouth_width = _distance(left_lip, right_lip)
        mouth_open = _distance(upper_lip, lower_lip)

        # Face height used to normalise AU4 brow metrics so results are stable
        # across different distances from the camera.
        face_height = max(abs(chin[1] - forehead[1]), 1e-6)

        brow_center = _centroid([left_brow, right_brow])
        # AU4 (brow lowerer): measure vertical gap from brow centroid to glabella,
        # normalised by face height.  Smaller gap = brows more furrowed/lowered.
        brow_vertical_gap = abs(brow_center[1] - between_brows[1])
        brow_gap_ratio = brow_vertical_gap / face_height  # ~0.10-0.25 neutral range
        # Express as "descent index": higher value = brows lower = more AU4 activation.
        # Subtracting from 1.0 flips direction so that the normaliser correctly maps
        # furrow → value above baseline → sigmoid > 0.5.
        au4_activation_raw = 1.0 - brow_gap_ratio
        # Tension binary: brow is tense when vertical gap ratio drops below threshold.
        brow_tense = brow_gap_ratio < self.cfg.brow_tension_threshold

        brow_raise_proxy = _distance(brow_center, forehead)

        left_ear = _ear(frame_data.landmarks, LEFT_EYE_EAR)
        right_ear = _ear(frame_data.landmarks, RIGHT_EYE_EAR)

        left_eye_width = _eye_horizontal_span(frame_data.landmarks, LEFT_EYE_EAR)
        right_eye_width = _eye_horizontal_span(frame_data.landmarks, RIGHT_EYE_EAR)
        max_eye_width = max(left_eye_width, right_eye_width, 1e-9)
        eye_asymmetry_ratio = abs(left_eye_width - right_eye_width) / max_eye_width

        # Robust fusion: use the minimum eye EAR to reduce asymmetry-induced false positives.
        eye_openness_raw = min(left_ear, right_ear)
        self.state.ear_median_buffer.append(eye_openness_raw)
        eye_openness = float(median(self.state.ear_median_buffer))

        expression_score = _safe_mean([mouth_width, mouth_open, brow_raise_proxy])
        self.state.expression_history.append(expression_score)

        # Motion primitives
        head_velocity = 0.0
        motion_energy = 0.0
        landmark_displacement_mean = 0.0
        micro_motion_energy = 0.0
        chin_vertical_velocity = 0.0

        if self.state.prev_landmarks is not None:
            prev_landmarks = self.state.prev_landmarks
            head_velocity = _distance(nose_tip, prev_landmarks[4]) / dt if len(prev_landmarks) > 4 else 0.0

            tracked = _all_displacements(frame_data.landmarks, prev_landmarks, self.region_landmark_ids)
            landmark_displacement_mean = _safe_mean(tracked)
            # Scale by 1e4: squared landmark distances in [0,1] space are ~1e-6;
            # without scaling, raw values are below sigma_floor and the normaliser
            # cannot distinguish variation.
            motion_energy = _safe_mean([d * d for d in tracked]) * 1e4

            # Micro-motion: compute frame-to-frame displacement across cheek and
            # moustache landmarks, then average squared displacements and scale to a
            # visible range (squared landmark distances in [0,1] space are ~1e-6).
            micro_ids = [50, 100, 101, 111, 117, 121, 123, 137,   # left_cheek
                         280, 329, 330, 340, 346, 350, 352, 366,   # right_cheek
                         57, 92, 164, 165, 212, 216, 287, 322]     # moustache
            micro_disps = _all_displacements(frame_data.landmarks, prev_landmarks, micro_ids)
            micro_motion_energy = _safe_mean([d * d for d in micro_disps]) * 1e4 if micro_disps else 0.0

            if 152 < len(prev_landmarks):
                chin_vertical_velocity = abs(chin[1] - prev_landmarks[152][1]) / dt

        self.state.head_velocity_history.append(head_velocity)
        self.state.motion_energy_history.append(motion_energy)
        self.state.eye_openness_history.append(eye_openness)

        # Normalize gaze offsets by interpupillary distance so thresholds are stable across faces.
        interpupillary = max(_distance(left_eye_center, right_eye_center), 1e-6)
        gaze_horizontal = _safe_mean(
            [left_iris[0] - left_eye_center[0], right_iris[0] - right_eye_center[0]]
        ) / interpupillary
        gaze_vertical = _safe_mean(
            [left_iris[1] - left_eye_center[1], right_iris[1] - right_eye_center[1]]
        ) / interpupillary
        self.state.gaze_horizontal_history.append(gaze_horizontal)
        self.state.gaze_vertical_history.append(gaze_vertical)

        # Event detection
        smile_active = mouth_width > self.cfg.smile_threshold
        if smile_active and not self.state.prev_smile_active:
            self.state.smile_events.append(now_ts)
        self.state.prev_smile_active = smile_active

        self.state.mouth_open_history.append(mouth_open)
        # Lip compression: simple tight absolute threshold (0.010).
        # Resting mouth is typically 0.018-0.025; only genuine deliberate lip presses
        # drop below 0.010.  The feature will sit near 0.5 in neutral sessions —
        # by design, it's a rare stress/anxiety marker, not a baseline measure.
        lip_compressed = mouth_open < self.cfg.lip_compression_threshold
        if lip_compressed and not self.state.prev_lip_compressed:
            self.state.lip_compression_events.append(now_ts)
        self.state.prev_lip_compressed = lip_compressed

        # AU20 (lip stretcher): edge-detect only — fires on the onset of the stretch,
        # not every frame it persists.  Raised threshold avoids false fires on minor
        # jaw relaxation.
        au20_active = mouth_open > self.cfg.au20_threshold
        if au20_active and not self.state.prev_au20_active:
            self.state.au20_events.append(now_ts)
        self.state.prev_au20_active = au20_active

        # au4_duration_ratio: store continuous activation (1 - brow_gap_ratio) so the
        # normaliser sees real variance.  Binary threshold was always True for neutral
        # faces (gap_ratio < threshold) → constant history → std≈0 → stuck at 0.5.
        self.state.brow_tension_history.append(au4_activation_raw)

        if self.state.prev_expression_score is not None:
            transition_cooldown_ok = (
                self.state.last_transition_timestamp is None
                or (now_ts - self.state.last_transition_timestamp) >= self.cfg.facial_transition_cooldown_seconds
            )
            if (abs(expression_score - self.state.prev_expression_score) > self.cfg.motion_transition_threshold
                    and transition_cooldown_ok):
                self.state.transition_events.append(now_ts)
                self.state.last_transition_timestamp = now_ts
        self.state.prev_expression_score = expression_score

        recent_eye_open = list(self.state.eye_openness_history)
        ear_derivative = 0.0
        if self.state.prev_eye_openness_smoothed is not None:
            ear_derivative = (eye_openness - self.state.prev_eye_openness_smoothed) / max(dt, 1e-6)

        quality_ok = left_eye_width > 1e-6 and right_eye_width > 1e-6 and eye_asymmetry_ratio <= self.cfg.eye_asymmetry_ratio_threshold

        if quality_ok and eye_openness > 0:
            # Baseline follows open-eye behavior only, updated while not in blink-active phase.
            if not self.state.prev_blink_active:
                self.state.open_eye_baseline.append(eye_openness)

        baseline_ear = _safe_mean(self.state.open_eye_baseline, default=eye_openness)
        adaptive_blink_threshold = max(0.08, self.cfg.ear_threshold_scale * baseline_ear)

        blink_active = quality_ok and (eye_openness < adaptive_blink_threshold)
        self.state.last_adaptive_blink_threshold = adaptive_blink_threshold

        if blink_active:
            if not self.state.prev_blink_active:
                self.state.blink_started_at = now_ts
                self.state.blink_active_frames = 1
                self.state.blink_min_ear = eye_openness
                self.state.blink_peak_drop_rate = max(0.0, -ear_derivative)
                self.state.blink_peak_rise_rate = 0.0
            else:
                self.state.blink_active_frames += 1
                self.state.blink_min_ear = eye_openness if self.state.blink_min_ear is None else min(self.state.blink_min_ear, eye_openness)
                self.state.blink_peak_drop_rate = max(self.state.blink_peak_drop_rate, max(0.0, -ear_derivative))
                self.state.blink_peak_rise_rate = max(self.state.blink_peak_rise_rate, max(0.0, ear_derivative))
        elif self.state.prev_blink_active:
            blink_frame_count = self.state.blink_active_frames
            blink_start = self.state.blink_started_at if self.state.blink_started_at is not None else now_ts
            blink_duration = max(0.0, now_ts - blink_start)
            blink_drop = max(0.0, baseline_ear - (self.state.blink_min_ear if self.state.blink_min_ear is not None else eye_openness))

            drop_ok = self.state.blink_peak_drop_rate >= self.cfg.ear_drop_rate_threshold
            rise_ok = max(self.state.blink_peak_rise_rate, max(0.0, ear_derivative)) >= self.cfg.ear_rise_rate_threshold
            depth_ok = blink_drop >= self.cfg.ear_min_drop

            # Relaxed gate: depth is mandatory; only one of drop/rise rate needs to
            # pass — at 15 FPS a 2-frame blink may miss the recovery phase entirely.
            if blink_frame_count >= self.cfg.blink_min_frames and depth_ok and (drop_ok or rise_ok):
                if self.state.last_blink_timestamp is None or (now_ts - self.state.last_blink_timestamp) >= 0.08:
                    self.state.blink_events.append(now_ts)
                    self.state.blink_durations.append(blink_duration)
                    # Cluster: inter-blink gap ≤ blink_cluster_window.  Normal blinks
                    # are ~4-5 s apart; anything under 2 s is burst/clustered blinking.
                    if len(self.state.blink_events) >= 2 and (self.state.blink_events[-1] - self.state.blink_events[-2]) <= self.cfg.blink_cluster_window:
                        self.state.blink_cluster_events.append(now_ts)
                    self.state.last_blink_timestamp = now_ts

            self.state.blink_started_at = None
            self.state.blink_active_frames = 0
            self.state.blink_min_ear = None
            self.state.blink_peak_drop_rate = 0.0
            self.state.blink_peak_rise_rate = 0.0

        self.state.prev_blink_active = blink_active
        self.state.prev_eye_openness_smoothed = eye_openness

        gaze_anchor = (gaze_horizontal, gaze_vertical)

        # Soft contact score: 1.0 when dead-centre, drops linearly to 0.0 at
        # gaze_contact_threshold units from centre (IPD-normalised).
        # Binary flag was stuck at 1.0 whenever gaze was within 0.10 IPD → constant
        # history → std≈0 → stuck at 0.5 after normalization.
        frontal_contact = max(0.0, 1.0 - abs(gaze_horizontal) / self.cfg.gaze_contact_threshold)
        self.state.eye_contact_flags.append(frontal_contact)

        if self.state.prev_gaze_anchor is not None:
            gaze_shift_mag = math.sqrt(
                (gaze_anchor[0] - self.state.prev_gaze_anchor[0]) ** 2 +
                (gaze_anchor[1] - self.state.prev_gaze_anchor[1]) ** 2
            )
            gaze_cooldown_ok = (
                self.state.last_gaze_shift_timestamp is None
                or (now_ts - self.state.last_gaze_shift_timestamp) >= self.cfg.gaze_shift_cooldown_seconds
            )
            if gaze_shift_mag > self.cfg.gaze_shift_threshold and gaze_cooldown_ok:
                self.state.gaze_shift_events.append(now_ts)
                self.state.last_gaze_shift_timestamp = now_ts
        self.state.prev_gaze_anchor = gaze_anchor

        # Downward gaze: threshold 0.06 IPD ≫ normal screen-gaze offset (~0.01-0.02).
        # Users looking at their webcam are always slightly down; only notable avoidance
        # gaze (looking at lap, floor) produces gaze_vertical > 0.06.
        downward_active = gaze_vertical > self.cfg.downward_gaze_threshold
        if downward_active and not self.state.prev_downward_active:
            self.state.downward_gaze_events.append(now_ts)
        self.state.prev_downward_active = downward_active

        # Speech/pause model from mouth opening proxy.
        # speaking_threshold is higher than mouth_open_threshold so that a slightly
        # parted resting mouth is not counted as speech, preventing perpetual
        # "speaking" state that would kill all pause/latency features.
        speaking = mouth_open > self.cfg.speaking_threshold
        if not speaking and self.state.prev_speaking:
            self.state.pause_started_at = now_ts

        if speaking and not self.state.prev_speaking:
            self.state.last_speech_onset_at = now_ts
            if self.state.pause_started_at is not None:
                latency = max(0.0, now_ts - self.state.pause_started_at)
                self.state.response_latencies.append(latency)
                self.state.pause_durations.append(latency)
                self.state.pause_started_at = None

        self.state.prev_speaking = speaking

        cooldown_ok = (
            self.state.last_gesture_timestamp is None
            or (now_ts - self.state.last_gesture_timestamp) >= self.cfg.gesture_cooldown_seconds
        )
        if chin_vertical_velocity > self.cfg.nod_velocity_threshold and cooldown_ok:
            self.state.gesture_events.append(now_ts)
            self.state.last_gesture_timestamp = now_ts
            if self.state.last_speech_onset_at is not None:
                nod_lag = now_ts - self.state.last_speech_onset_at
                # Only count nods that occur within the latency window of a speech
                # onset — a chin movement 40 s after the last word is not a response.
                if 0.0 <= nod_lag <= self.cfg.nod_latency_window:
                    self.state.nod_latencies.append(nod_lag)

        # Feature assembly (raw values)
        recent_expr = list(self.state.expression_history)
        recent_motion = list(self.state.motion_energy_history)
        recent_head_vel = list(self.state.head_velocity_history)
        recent_eye_open = list(self.state.eye_openness_history)

        eye_contact_ratio = _safe_mean(self.state.eye_contact_flags, default=0.0)

        # near_zero_au_activation_ratio: fraction of frames where the face was nearly
        # motionless — affective blunting detector.  Compare frame-to-frame DELTAS,
        # not absolute expression values (which are structurally non-zero).
        near_zero_ratio = 0.0
        if len(recent_expr) >= 2:
            expr_deltas = [abs(recent_expr[i] - recent_expr[i - 1]) for i in range(1, len(recent_expr))]
            near_zero_ratio = _safe_mean([1.0 if d < self.cfg.motion_transition_threshold else 0.0 for d in expr_deltas])

        extended_ratio = 0.0
        if self.state.pause_durations:
            extended_ratio = _safe_mean([
                1.0 if d > self.cfg.extended_silence_threshold else 0.0
                for d in self.state.pause_durations
            ])

        features: Dict[str, float] = {
            "au12_mean_amplitude": mouth_width,
            # Scale by 1e2: variance of expression_score (~1e-4 raw) would sit below
            # sigma_floor=0.01 without scaling, pinning normalised output to 0.5.
            "au12_variance": _safe_variance(recent_expr) * 1e2,
            "au12_activation_frequency": _event_rate(self.state.smile_events, now_ts, self.cfg.event_window_seconds),
            "au15_mean_amplitude": _distance(_centroid([left_lip, right_lip]), lower_lip),
            # au4_activation_raw is 1 - brow_gap_ratio: higher = brows more lowered.
            "au4_mean_activation": au4_activation_raw,
            "au4_duration_ratio": _safe_mean(self.state.brow_tension_history),
            "au1_au2_peak_intensity": max([brow_raise_proxy] + recent_expr[-10:]),
            "au20_activation_rate": _event_rate(self.state.au20_events, now_ts, self.cfg.event_window_seconds),
            "lip_compression_frequency": _event_rate(self.state.lip_compression_events, now_ts, self.cfg.event_window_seconds),
            # overall_au_variance: composite of mouth (AU12) + brow (AU4) channels,
            # so it captures whole-face variability rather than duplicating au12_variance.
            "overall_au_variance": _safe_variance(
                [(e + b) / 2.0 for e, b in zip(
                    recent_expr[-len(self.state.brow_tension_history):],
                    list(self.state.brow_tension_history)
                )]
            ) * 1e2,
            "facial_emotional_range": (max(recent_expr) - min(recent_expr)) if recent_expr else 0.0,
            "facial_transition_frequency": _event_rate(self.state.transition_events, now_ts, self.cfg.short_event_window_seconds),
            "near_zero_au_activation_ratio": near_zero_ratio,
            "mean_head_velocity": _safe_mean(recent_head_vel),
            "head_velocity_peak": max(recent_head_vel) if recent_head_vel else 0.0,
            "head_motion_energy": _safe_mean(recent_motion),
            # motion_energy_floor_score: fraction of recent frames where motion energy is
            # below the "near-still" floor (~jitter level for a resting face at webcam
            # distance).  High value = mostly frozen — psychomotor retardation signal.
            # Threshold 0.2: motion_energy is now scaled by 1e4, so 2e-5 raw → 0.2 here.
            "motion_energy_floor_score": _safe_mean([1.0 if e < 0.2 else 0.0 for e in recent_motion]) if recent_motion else 0.0,
            "landmark_displacement_mean": landmark_displacement_mean,
            # gesture/gaze_shift/transition use a shorter 10 s window so the rate
            # fluctuates during the 30 s baseline, giving the normaliser real variance
            # instead of a near-zero baseline from slow 60 s accumulation.
            "gesture_frequency": _event_rate(self.state.gesture_events, now_ts, self.cfg.short_event_window_seconds),
            "posture_rigidity_index": 1.0 / (1.0 + _safe_variance(recent_head_vel)),
            "micro_motion_energy": micro_motion_energy,
            "blink_rate": _event_rate(self.state.blink_events, now_ts, 60.0) * 60.0,
            "blink_duration": _safe_mean(self.state.blink_durations),
            "blink_cluster_density": _event_rate(self.state.blink_cluster_events, now_ts, self.cfg.event_window_seconds),
            "eye_contact_ratio": eye_contact_ratio,
            "downward_gaze_frequency": _event_rate(self.state.downward_gaze_events, now_ts, self.cfg.event_window_seconds),
            "gaze_shift_frequency": _event_rate(self.state.gaze_shift_events, now_ts, self.cfg.short_event_window_seconds),
            "baseline_eye_openness": eye_openness,
            "response_latency_mean": _safe_mean(self.state.response_latencies),
            "speech_onset_delay": self.state.response_latencies[-1] if self.state.response_latencies else 0.0,
            "nod_onset_latency": _safe_mean(self.state.nod_latencies),
            "pause_duration_mean": _safe_mean(self.state.pause_durations),
            "extended_silence_ratio": extended_ratio,
            "reaction_time_instability_index": math.sqrt(_safe_variance(list(self.state.response_latencies))),
        }

        # Guarantee all contract keys exist, then smooth.
        output: Dict[str, float] = {}
        for feature_name in self.feature_order:
            raw_value = float(features.get(feature_name, 0.0))
            output[feature_name] = self._smooth(feature_name, raw_value)

        self.state.prev_timestamp = now_ts
        self.state.prev_landmarks = list(frame_data.landmarks)
        return output
