"""Main runtime pipeline orchestration (foundation phase)."""

from __future__ import annotations

import cv2
from dataclasses import dataclass
import mediapipe as mp
import numpy as np
from pathlib import Path

from src.config import RuntimeConfig
from src.csv_logger import CsvLoggerError, FeatureCsvLogger
from src.feature_engine import FeatureEngine, FeatureEngineConfig, FeatureEngineError
from src.landmark_stream import LandmarkStream, LandmarkStreamError
from src.normalization import BaselineNormalizer, NormalizationError
from src.regions import RegionMappingError, extract_regions, flatten_region_ids


class PipelineError(RuntimeError):
    """Raised when pipeline processing encounters unrecoverable errors."""


# ---------------------------------------------------------------------------
# Timed behavioral protocol — shown after baseline locks.
# Each entry: (start_s, end_s, group_label, instruction, BGR_color)
# Covers all 34 feature groups so every sensor gets exercised.
# ---------------------------------------------------------------------------
_PROTOCOL: list[tuple[float, float, str, str, tuple[int, int, int]]] = [
    # -- GAZE & EYE CONTACT --
    (  0,  10, "GAZE",       "Look directly at the camera (hold steady)", (0, 220, 255)),
    ( 10,  18, "GAZE",       "Slowly look LEFT, hold for 3 s", (0, 200, 255)),
    ( 18,  26, "GAZE",       "Slowly look RIGHT, hold for 3 s", (0, 200, 255)),
    ( 26,  36, "GAZE",       "Look DOWN at your lap / keyboard, hold 5 s", (0, 180, 255)),
    ( 36,  42, "GAZE",       "Return eyes to camera", (0, 220, 255)),
    # -- EXPRESSION --
    ( 42,  52, "EXPRESSION", "Raise eyebrows high, then frown, then back to neutral", (255, 165, 0)),
    ( 52,  58, "EXPRESSION", "Smile naturally, hold 3 s, return to neutral", (255, 165, 0)),
    # -- SPEECH & PAUSE (response_latency, pause_duration, silence_ratio) --
    ( 58,  73, "SPEECH",     "Say: 'Hello, how are you today?'  then PAUSE silently for 5 s", (80, 220, 100)),
    ( 73,  88, "SPEECH",     "Say a full sentence, then PAUSE again for 5 s", (80, 220, 100)),
    # -- NOD / GESTURE (nod_onset_latency fires within ~1 s of speech) --
    ( 88,  98, "GESTURE",    "Slowly nod your head 3 times (chin down-and-up each time)", (255, 220, 50)),
    # -- LIP COMPRESSION (lip_compression_frequency) --
    ( 98, 106, "LIP",        "Press lips firmly together, hold 1 s, release. Repeat 3 x", (255, 100, 100)),
    # -- BLINK CLUSTER (blink_cluster_density) --
    (106, 116, "BLINK",      "Blink rapidly 4-5 times, then blink normally", (200, 200, 255)),
    # -- WIND-DOWN --
    (116, 130, "DONE",       "Sit naturally — data collection finishing. Close window to stop.", (160, 160, 160)),
]


def _get_protocol_step(
    elapsed_s: float,
) -> tuple[str, str, tuple[int, int, int], float, float] | None:
    """Return (group, instruction, color, step_elapsed, step_duration) for current time."""
    for start, end, group, text, color in _PROTOCOL:
        if start <= elapsed_s < end:
            return group, text, color, elapsed_s - start, end - start
    return None


@dataclass
class RuntimeViewConfig:
    show_mapped_points: bool = True
    show_mapped_labels: bool = True
    show_mapped_connections: bool = False
    show_background_mesh: bool = False
    point_radius: int = 1
    line_thickness: int = 1


def _normalize_connection(connection: tuple[int, int] | list[int]) -> tuple[int, int]:
    start, end = int(connection[0]), int(connection[1])
    return (start, end) if start <= end else (end, start)


def _get_connection_catalog() -> dict[str, set[tuple[int, int]]]:
    mesh = mp.solutions.face_mesh
    return {
        "tessellation": {_normalize_connection(conn) for conn in mesh.FACEMESH_TESSELATION},
        "contours": {_normalize_connection(conn) for conn in mesh.FACEMESH_CONTOURS},
        "irises": {_normalize_connection(conn) for conn in mesh.FACEMESH_IRISES},
    }


def _get_connections_between_landmarks(
    landmark_ids: list[int],
    catalog: dict[str, set[tuple[int, int]]],
) -> list[tuple[int, int]]:
    id_set = set(landmark_ids)
    visible: set[tuple[int, int]] = set()
    for source_name in ("contours", "irises", "tessellation"):
        for start_idx, end_idx in catalog[source_name]:
            if start_idx in id_set and end_idx in id_set:
                visible.add((start_idx, end_idx))
    return sorted(visible)


def _draw_connections_from_tuples(
    frame: np.ndarray,
    landmarks: list[tuple[float, float, float]],
    connections: list[tuple[int, int]],
    *,
    color: tuple[int, int, int],
    thickness: int,
) -> None:
    height, width, _ = frame.shape
    for start_idx, end_idx in connections:
        if start_idx >= len(landmarks) or end_idx >= len(landmarks):
            continue
        sx = int(landmarks[start_idx][0] * width)
        sy = int(landmarks[start_idx][1] * height)
        ex = int(landmarks[end_idx][0] * width)
        ey = int(landmarks[end_idx][1] * height)
        cv2.line(frame, (sx, sy), (ex, ey), color, thickness, cv2.LINE_AA)


def _draw_mapped_landmark_ids(
    frame: np.ndarray,
    landmarks: list[tuple[float, float, float]],
    mapped_ids: list[int],
    view: RuntimeViewConfig,
) -> None:
    """Draw IDs only for mapped landmark subset defined by region mapping."""
    height, width, _ = frame.shape

    for idx in mapped_ids:
        if idx >= len(landmarks):
            continue

        x_norm, y_norm, _ = landmarks[idx]
        x = int(x_norm * width)
        y = int(y_norm * height)

        if view.show_mapped_points:
            cv2.circle(frame, (x, y), view.point_radius, (0, 255, 255), -1, cv2.LINE_AA)
        if view.show_mapped_labels:
            cv2.putText(
                frame,
                str(idx),
                (x + 2, y - 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.28,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )


def _build_feature_panel(
    feature_order: list[str],
    feature_values: dict[str, float],
    phase_label: str,
    baseline_remaining: float,
    normalized_elapsed: float = 0.0,
) -> np.ndarray:
    """Render all feature values on a dedicated panel for live monitoring."""
    line_height = 22
    panel_width = 980

    # Extra header rows: 4 baseline lines + optional 3 protocol lines
    header_h = 160
    panel_height = header_h + (len(feature_order) * line_height)
    panel = np.zeros((panel_height, panel_width, 3), dtype=np.uint8)

    # Title
    cv2.putText(panel, f"Live Feature Values ({len(feature_order)})",
                (20, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.80, (80, 220, 100), 2, cv2.LINE_AA)

    # Phase + baseline countdown
    phase_color = (0, 180, 255) if phase_label == "BASELINE" else (80, 220, 100)
    cv2.putText(panel, f"Phase: {phase_label}", (20, 62),
                cv2.FONT_HERSHEY_SIMPLEX, 0.62, phase_color, 2, cv2.LINE_AA)
    if phase_label == "BASELINE":
        cv2.putText(panel, f"Sit naturally — baseline remaining: {baseline_remaining:.1f}s",
                    (230, 62), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 180, 255), 1, cv2.LINE_AA)

    # Protocol step (NORMALIZED phase only)
    if phase_label == "NORMALIZED":
        step = _get_protocol_step(normalized_elapsed)
        if step:
            group, instruction, color, step_elapsed, step_dur = step
            # Group label + instruction
            cv2.putText(panel, f"[{group}]  {instruction}",
                        (20, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.60, color, 2, cv2.LINE_AA)
            # Progress bar for this step
            bar_x, bar_y, bar_w, bar_h = 20, 108, 940, 10
            progress = min(step_elapsed / max(step_dur, 1e-6), 1.0)
            cv2.rectangle(panel, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (60, 60, 60), -1)
            cv2.rectangle(panel, (bar_x, bar_y), (bar_x + int(bar_w * progress), bar_y + bar_h), color, -1)
            remaining = max(0.0, step_dur - step_elapsed)
            cv2.putText(panel, f"{remaining:.0f}s remaining in this step",
                        (20, 132), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1, cv2.LINE_AA)
        else:
            cv2.putText(panel, "Protocol complete — close window to stop",
                        (20, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.60, (160, 160, 160), 1, cv2.LINE_AA)

    # Column header
    cv2.putText(panel, "Index | Feature Name                   | Value",
                (20, 152), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (255, 255, 255), 1, cv2.LINE_AA)

    decimals = 8 if phase_label == "NORMALIZED" else 6
    for idx, feature_name in enumerate(feature_order, start=1):
        y = header_h + (idx * line_height) - 6
        value = float(feature_values.get(feature_name, 0.0))
        line = f"{idx:02d}. {feature_name:<34} {value:.{decimals}f}"
        cv2.putText(panel, line, (20, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220), 1, cv2.LINE_AA)

    return panel


def run_raw_extraction_pipeline(
    cfg: RuntimeConfig,
    feature_order: list[str],
    project_root: Path,
    video_file_path: str | None = None,
) -> None:
    """Run raw feature extraction pipeline WITHOUT normalization or protocol overlay.

    Processing stages:
    1) video file OR webcam + face mesh + regions
    2) raw feature extraction (no baseline normalization)
    3) direct CSV export with raw values
    
    This is the POC approach: extract raw landmarks/features from a single case
    (depression patient video), use it as ground truth, and later generate
    synthetic variants for training.
    
    Args:
        cfg: Runtime configuration
        feature_order: List of feature names in order
        project_root: Root directory of the project
        video_file_path: Optional path to video file. If None, uses webcam.
    """
    window_name = "Facial Analysis - Runtime (RAW EXTRACTION)"
    feature_engine = FeatureEngine(
        feature_order=feature_order,
        cfg=FeatureEngineConfig(
            smoothing_window=cfg.smoothing_window,
            event_window_seconds=cfg.thresholds.event_window_seconds,
            smile_threshold=cfg.thresholds.smile_threshold,
            mouth_open_threshold=cfg.thresholds.mouth_open_threshold,
            lip_compression_threshold=cfg.thresholds.lip_compression_threshold,
            brow_tension_threshold=cfg.thresholds.brow_tension_threshold,
            blink_ear_threshold=cfg.thresholds.blink_ear_threshold,
            blink_min_frames=cfg.thresholds.blink_min_frames,
            ear_threshold_scale=cfg.thresholds.ear_threshold_scale,
            ear_median_window=cfg.thresholds.ear_median_window,
            ear_drop_rate_threshold=cfg.thresholds.ear_drop_rate_threshold,
            ear_rise_rate_threshold=cfg.thresholds.ear_rise_rate_threshold,
            ear_min_drop=cfg.thresholds.ear_min_drop,
            eye_asymmetry_ratio_threshold=cfg.thresholds.eye_asymmetry_ratio_threshold,
            gaze_shift_threshold=cfg.thresholds.gaze_shift_threshold,
            downward_gaze_threshold=cfg.thresholds.downward_gaze_threshold,
            motion_transition_threshold=cfg.thresholds.motion_transition_threshold,
            nod_velocity_threshold=cfg.thresholds.nod_velocity_threshold,
            extended_silence_threshold=cfg.thresholds.extended_silence_threshold,
        ),
    )
    output_dir = cfg.output_path(project_root)
    mapped_ids = flatten_region_ids()
    view = RuntimeViewConfig()
    connection_catalog = _get_connection_catalog()
    mapped_connections = _get_connections_between_landmarks(mapped_ids, connection_catalog)
    full_mesh_connections = sorted(connection_catalog["tessellation"])

    try:
        with LandmarkStream(cfg, video_file_path=video_file_path) as stream, FeatureCsvLogger(output_dir=output_dir, feature_order=feature_order, allow_raw_features=True) as logger:
            start_ts = None
            frame_count = 0

            for frame_data, frame in stream.frames():
                try:
                    regions = extract_regions(frame_data.landmarks, strict=True)
                    region_count = len(regions)
                except RegionMappingError as exc:
                    raise PipelineError(f"Region extraction failed: {exc}") from exc

                try:
                    raw_features = feature_engine.compute(frame_data)
                except FeatureEngineError as exc:
                    raise PipelineError(f"Feature computation failed: {exc}") from exc

                if start_ts is None:
                    start_ts = frame_data.timestamp
                    print(f"[OK] Starting raw feature extraction from {video_file_path if video_file_path else 'webcam'}")
                    print(f"[OK] Writing raw features to: {logger.path}")

                frame_count += 1
                
                # Write raw features directly (no normalization)
                try:
                    logger.write_row(frame_data.timestamp, raw_features)
                except CsvLoggerError as exc:
                    raise PipelineError(f"CSV write failed: {exc}") from exc

                # Drawing and visualization
                cv2.putText(
                    frame,
                    f"Frame: {frame_count} | Landmarks: {len(frame_data.landmarks)} | Regions: {region_count}",
                    (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 255),
                    2,
                    cv2.LINE_AA,
                )

                if view.show_background_mesh:
                    _draw_connections_from_tuples(
                        frame,
                        frame_data.landmarks,
                        full_mesh_connections,
                        color=(90, 90, 90),
                        thickness=view.line_thickness,
                    )

                if view.show_mapped_connections:
                    _draw_connections_from_tuples(
                        frame,
                        frame_data.landmarks,
                        mapped_connections,
                        color=(255, 120, 30),
                        thickness=view.line_thickness,
                    )

                _draw_mapped_landmark_ids(
                    frame=frame,
                    landmarks=frame_data.landmarks,
                    mapped_ids=mapped_ids,
                    view=view,
                )
                
                elapsed = frame_data.timestamp - start_ts
                cv2.putText(
                    frame,
                    f"Elapsed: {elapsed:.2f}s",
                    (20, 60),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (80, 220, 100),
                    2,
                    cv2.LINE_AA,
                )
                
                cv2.putText(
                    frame,
                    f"Raw features extracted: {len(raw_features)}/34",
                    (20, 90),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 180, 255),
                    1,
                    cv2.LINE_AA,
                )
                
                cv2.putText(
                    frame,
                    f"Eye openness: {raw_features.get('baseline_eye_openness', 0):.6f}",
                    (20, 120),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (180, 220, 255),
                    1,
                    cv2.LINE_AA,
                )
                
                cv2.putText(
                    frame,
                    f"Blink rate: {raw_features.get('blink_rate', 0):.6f}",
                    (20, 145),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (180, 220, 255),
                    1,
                    cv2.LINE_AA,
                )

                cv2.putText(
                    frame,
                    "q quit | e connections | f full mesh | n labels | d dots | +/- thickness | [/] radius",
                    (20, 170),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.48,
                    (255, 255, 255),
                    1,
                    cv2.LINE_AA,
                )

                cv2.imshow(window_name, frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                if key == ord("e"):
                    view.show_mapped_connections = not view.show_mapped_connections
                if key == ord("f"):
                    view.show_background_mesh = not view.show_background_mesh
                if key == ord("n"):
                    view.show_mapped_labels = not view.show_mapped_labels
                if key == ord("d"):
                    view.show_mapped_points = not view.show_mapped_points
                if key in (ord("+"), ord("=")):
                    view.line_thickness = min(view.line_thickness + 1, 5)
                if key == ord("-"):
                    view.line_thickness = max(view.line_thickness - 1, 1)
                if key == ord("]"):
                    view.point_radius = min(view.point_radius + 1, 6)
                if key == ord("["):
                    view.point_radius = max(view.point_radius - 1, 1)

    except LandmarkStreamError as exc:
        raise PipelineError(f"Landmark stream setup failed: {exc}") from exc
    except cv2.error as exc:
        raise PipelineError(f"OpenCV runtime error: {exc}") from exc
    finally:
        cv2.destroyAllWindows()
        print(f"[OK] Raw feature extraction complete. {frame_count} frames processed.")
