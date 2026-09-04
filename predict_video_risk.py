#!/usr/bin/env python3
"""End-to-end video inference for the trained mental-health risk classifier.

Pipeline:
1) Video -> per-frame 34 raw behavioral features
2) Session aggregation -> engineered patient-level feature vector
3) Exact schema projection -> trained 63-feature model input
4) Model inference -> class probabilities + prediction output JSON
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd

from src.config import RuntimeConfig, validate_runtime_config
from src.contracts import ContractError, load_feature_contract
from src.feature_engine import FeatureEngine, FeatureEngineConfig, FeatureEngineError
from src.landmark_stream import LandmarkStream, LandmarkStreamError
from src.regions import RegionMappingError, validate_runtime_contract


MODE_PROFILES: Dict[str, Dict[str, int]] = {
    "accurate": {"frame_stride": 1},
    "balanced": {"frame_stride": 2},
    "fast": {"frame_stride": 3},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run end-to-end video prediction with the trained model artifacts."
    )
    parser.add_argument("--video-file", required=True, help="Path to input video file.")
    parser.add_argument(
        "--model-dir",
        default="reports/model_training/run_20260324_171117",
        help="Directory containing best_model.joblib, label_encoder.joblib, and training_report.json.",
    )
    parser.add_argument(
        "--training-report",
        default="",
        help="Optional explicit path to training_report.json. Defaults to <model-dir>/training_report.json.",
    )
    parser.add_argument(
        "--label-col",
        default="condition_label",
        help="Label column name used in the training CSV.",
    )
    parser.add_argument(
        "--min-duration-seconds",
        type=float,
        default=60.0,
        help="Soft minimum recommended video duration for stable extraction.",
    )
    parser.add_argument(
        "--allow-short",
        action="store_true",
        help="Allow prediction even if video duration is shorter than minimum recommendation.",
    )
    parser.add_argument(
        "--output-root",
        default="reports/inference",
        help="Root directory for inference artifacts.",
    )
    parser.add_argument(
        "--mode",
        choices=["accurate", "balanced", "fast"],
        default="balanced",
        help="Inference speed/accuracy profile. Default is balanced.",
    )
    parser.add_argument(
        "--frame-stride",
        type=int,
        default=0,
        help="Optional override for frame sampling stride (>=1). 0 uses mode profile.",
    )
    return parser.parse_args()


def _build_engine_cfg(cfg: RuntimeConfig) -> FeatureEngineConfig:
    t = cfg.thresholds
    return FeatureEngineConfig(
        smoothing_window=cfg.smoothing_window,
        event_window_seconds=t.event_window_seconds,
        smile_threshold=t.smile_threshold,
        mouth_open_threshold=t.mouth_open_threshold,
        lip_compression_threshold=t.lip_compression_threshold,
        brow_tension_threshold=t.brow_tension_threshold,
        blink_ear_threshold=t.blink_ear_threshold,
        blink_min_frames=t.blink_min_frames,
        ear_threshold_scale=t.ear_threshold_scale,
        ear_median_window=t.ear_median_window,
        ear_drop_rate_threshold=t.ear_drop_rate_threshold,
        ear_rise_rate_threshold=t.ear_rise_rate_threshold,
        ear_min_drop=t.ear_min_drop,
        eye_asymmetry_ratio_threshold=t.eye_asymmetry_ratio_threshold,
        gaze_shift_threshold=t.gaze_shift_threshold,
        downward_gaze_threshold=t.downward_gaze_threshold,
        motion_transition_threshold=t.motion_transition_threshold,
        nod_velocity_threshold=t.nod_velocity_threshold,
        extended_silence_threshold=t.extended_silence_threshold,
    )


def extract_raw_timeseries(
    video_file: Path,
    cfg: RuntimeConfig,
    feature_order: List[str],
    frame_stride: int,
) -> Tuple[pd.DataFrame, Dict[str, int]]:
    if frame_stride <= 0:
        raise ValueError(f"frame_stride must be >=1, got {frame_stride}")

    engine = FeatureEngine(feature_order=feature_order, cfg=_build_engine_cfg(cfg))

    rows: List[Dict[str, float]] = []
    frames_seen = 0
    frames_kept = 0
    with LandmarkStream(cfg, video_file_path=str(video_file)) as stream:
        for frame_data, _frame in stream.frames():
            frames_seen += 1
            if (frames_seen - 1) % frame_stride != 0:
                continue

            try:
                raw = engine.compute(frame_data)
            except FeatureEngineError as exc:
                raise RuntimeError(f"Feature extraction failed: {exc}") from exc

            row: Dict[str, float] = {"timestamp": float(frame_data.timestamp)}
            for name in feature_order:
                row[name] = float(raw[name])
            rows.append(row)
            frames_kept += 1

    if not rows:
        raise RuntimeError("No valid face frames were extracted from the video.")

    stats = {
        "frames_seen_with_face": frames_seen,
        "frames_used_for_features": frames_kept,
        "frame_stride": frame_stride,
    }
    return pd.DataFrame(rows), stats


def engineer_session_vector(raw_df: pd.DataFrame, feature_cols: List[str]) -> Dict[str, float]:
    if "timestamp" not in raw_df.columns:
        raise ValueError("raw_df is missing timestamp column.")

    t = raw_df["timestamp"].to_numpy(dtype=np.float64)
    t_shifted = t - t.mean()
    t_var = float(np.var(t_shifted)) if len(t_shifted) > 1 else 0.0

    profile: Dict[str, float] = {}
    for feat in feature_cols:
        x = raw_df[feat].to_numpy(dtype=np.float64)

        mean_v = float(np.mean(x))
        std_v = float(np.std(x, ddof=1)) if len(x) > 1 else 0.0
        min_v = float(np.min(x))
        max_v = float(np.max(x))
        range_v = max_v - min_v

        if len(x) > 1 and t_var > 1e-12:
            x_shifted = x - x.mean()
            slope_v = float(np.sum(t_shifted * x_shifted) / (len(x) * t_var))
        else:
            slope_v = 0.0

        profile[f"{feat}__mean"] = mean_v
        profile[f"{feat}__std"] = std_v
        profile[f"{feat}__min"] = min_v
        profile[f"{feat}__max"] = max_v
        profile[f"{feat}__range"] = range_v
        profile[f"{feat}__slope"] = slope_v

    return profile


def resolve_expected_feature_order(
    project_root: Path,
    training_report_path: Path,
    label_col: str,
) -> Tuple[List[str], Path, List[str]]:
    report = json.loads(training_report_path.read_text(encoding="utf-8"))
    report_classes = [str(c) for c in report.get("classes", [])]
    input_csv = report.get("input", "")

    if input_csv:
        input_path = Path(input_csv)
        if not input_path.is_absolute():
            input_path = (project_root / input_path).resolve()

        if input_path.exists():
            header_df = pd.read_csv(input_path, nrows=0)
            expected = [c for c in header_df.columns.tolist() if c != label_col]
            if expected:
                return expected, input_path, report_classes

    # Fallback: resolve schema from saved feature-importance artifact.
    artifacts = report.get("artifacts", {}) if isinstance(report.get("artifacts", {}), dict) else {}
    importance_csv = artifacts.get("feature_importance_csv", "")
    if importance_csv:
        importance_path = Path(str(importance_csv))
        if not importance_path.is_absolute():
            importance_path = (project_root / importance_path).resolve()
        if importance_path.exists():
            imp_df = pd.read_csv(importance_path)
            if "feature" in imp_df.columns:
                expected = [str(f) for f in imp_df["feature"].tolist()]
                if expected:
                    return expected, importance_path, report_classes

    raise FileNotFoundError(
        "Could not resolve expected feature schema. Neither training input CSV nor feature_importance CSV is available."
    )


def _probability_output(model: Any, X_row: pd.DataFrame, class_labels: List[str], pred_idx: int) -> Tuple[Dict[str, float], float]:
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X_row)
        prob_vec = np.asarray(probs[0], dtype=float)
    else:
        # Defensive fallback if a model lacks predict_proba.
        prob_vec = np.zeros(len(class_labels), dtype=float)
        prob_vec[pred_idx] = 1.0

    probabilities = {
        class_labels[i]: float(prob_vec[i]) for i in range(min(len(class_labels), len(prob_vec)))
    }

    confidence = float(prob_vec[pred_idx]) if pred_idx < len(prob_vec) else 1.0
    return probabilities, confidence


def main() -> int:
    args = parse_args()

    project_root = Path(__file__).resolve().parent
    cfg = RuntimeConfig()
    validate_runtime_config(cfg)

    _ = validate_runtime_contract()
    contract = load_feature_contract(project_root)

    video_file = Path(args.video_file)
    if not video_file.exists() or not video_file.is_file():
        raise FileNotFoundError(f"Video file not found: {video_file}")

    model_dir = Path(args.model_dir)
    if not model_dir.is_absolute():
        model_dir = (project_root / model_dir).resolve()

    model_path = model_dir / "best_model.joblib"
    encoder_path = model_dir / "label_encoder.joblib"
    if not model_path.exists():
        raise FileNotFoundError(f"Model artifact missing: {model_path}")

    training_report_path = Path(args.training_report) if args.training_report else (model_dir / "training_report.json")
    if not training_report_path.is_absolute():
        training_report_path = (project_root / training_report_path).resolve()
    if not training_report_path.exists():
        raise FileNotFoundError(f"Training report not found: {training_report_path}")

    expected_features, training_input_path, report_classes = resolve_expected_feature_order(
        project_root=project_root,
        training_report_path=training_report_path,
        label_col=args.label_col,
    )

    mode_profile = MODE_PROFILES[args.mode]
    frame_stride = int(args.frame_stride) if int(args.frame_stride) > 0 else int(mode_profile["frame_stride"])
    if frame_stride < 1:
        raise ValueError(f"Invalid frame stride: {frame_stride}")

    raw_df, extraction_stats = extract_raw_timeseries(
        video_file=video_file,
        cfg=cfg,
        feature_order=contract.feature_order,
        frame_stride=frame_stride,
    )

    duration_seconds = float(raw_df["timestamp"].iloc[-1] - raw_df["timestamp"].iloc[0])
    if duration_seconds < float(args.min_duration_seconds) and not args.allow_short:
        raise ValueError(
            f"Video too short ({duration_seconds:.2f}s). Recommended >= {args.min_duration_seconds:.2f}s. "
            "Use --allow-short to override."
        )

    engineered = engineer_session_vector(raw_df=raw_df, feature_cols=contract.feature_order)

    model = joblib.load(model_path)
    model_features: List[str]
    if hasattr(model, "feature_names_in_"):
        model_features = [str(name) for name in model.feature_names_in_.tolist()]
    else:
        model_features = list(expected_features)

    missing = [name for name in model_features if name not in engineered]
    if missing:
        raise ValueError(f"Engineered vector missing expected model features: {missing[:10]} (total {len(missing)})")

    X_model = pd.DataFrame([{name: engineered[name] for name in model_features}], columns=model_features)
    expected_features = model_features

    pred_idx = int(model.predict(X_model)[0])

    if report_classes:
        class_labels = report_classes
    elif encoder_path.exists():
        label_encoder = joblib.load(encoder_path)
        class_labels = [str(c) for c in label_encoder.classes_.tolist()]
    else:
        raise FileNotFoundError(
            "Could not resolve class labels: training_report has no classes and label_encoder.joblib not found."
        )

    if pred_idx < 0 or pred_idx >= len(class_labels):
        raise ValueError(f"Predicted class index {pred_idx} is out of range for {len(class_labels)} classes")

    pred_label = class_labels[pred_idx]

    probabilities, confidence = _probability_output(
        model=model,
        X_row=X_model,
        class_labels=class_labels,
        pred_idx=pred_idx,
    )

    # Apply a conservative fallback: if the top predicted probability is low,
    # treat the session as `normal` (no notable condition). This prevents
    # spurious labels when the model is uncertain or the pattern is outside
    # the trained classes.
    top_prob = max(probabilities.values()) if probabilities else 0.0
    fallback_applied = False
    if top_prob < 0.50:
        pred_label = "normal"
        fallback_applied = True
        risk_score = 0.0
    else:
        # retain original model confidence as risk score
        risk_score = confidence

    ranked = sorted(probabilities.items(), key=lambda kv: kv[1], reverse=True)
    top3 = [{"label": k, "probability": float(v)} for k, v in ranked[:3]]

    run_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(args.output_root) / f"run_{run_stamp}"
    if not run_dir.is_absolute():
        run_dir = (project_root / run_dir).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)

    raw_csv = run_dir / "raw_frame_features.csv"
    eng_csv = run_dir / "session_engineered_204.csv"
    model_input_csv = run_dir / "model_input_vector.csv"
    pred_json = run_dir / "prediction.json"

    raw_df.to_csv(raw_csv, index=False)
    pd.DataFrame([engineered]).to_csv(eng_csv, index=False)
    X_model.to_csv(model_input_csv, index=False)

    output_payload = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "video_file": str(video_file),
        "model_dir": str(model_dir),
        "training_report": str(training_report_path),
        "training_input_schema_source": str(training_input_path),
        "session": {
            "frames_with_face": int(extraction_stats["frames_seen_with_face"]),
            "frames_used_for_features": int(extraction_stats["frames_used_for_features"]),
            "duration_seconds": duration_seconds,
            "min_duration_seconds": float(args.min_duration_seconds),
            "duration_check_passed": bool(duration_seconds >= float(args.min_duration_seconds)),
        },
        "runtime": {
            "mode": args.mode,
            "frame_stride": frame_stride,
            "mode_default_stride": int(mode_profile["frame_stride"]),
        },
        "schema": {
            "runtime_base_feature_count": int(len(contract.feature_order)),
            "engineered_feature_count": int(len(engineered)),
            "model_feature_count": int(len(expected_features)),
        },
        "prediction": {
            "predicted_label": pred_label,
            "predicted_index": pred_idx,
            "confidence": confidence,
            "risk_score": risk_score,
            "fallback_applied": fallback_applied,
            "probabilities": probabilities,
            "top3": top3,
        },
        "artifacts": {
            "run_dir": str(run_dir),
            "raw_frame_features_csv": str(raw_csv),
            "session_engineered_csv": str(eng_csv),
            "model_input_csv": str(model_input_csv),
            "prediction_json": str(pred_json),
        },
    }

    pred_json.write_text(json.dumps(output_payload, indent=2), encoding="utf-8")

    print("[INFER] Video inference complete")
    print(f"[INFER] Mode: {args.mode} (frame_stride={frame_stride})")
    print(f"[INFER] Frames with face: {extraction_stats['frames_seen_with_face']}")
    print(f"[INFER] Frames used: {extraction_stats['frames_used_for_features']}")
    print(f"[INFER] Duration (s): {duration_seconds:.2f}")
    print(f"[INFER] Runtime base features: {len(contract.feature_order)}")
    print(f"[INFER] Engineered features: {len(engineered)}")
    print(f"[INFER] Model input features: {len(expected_features)}")
    print(f"[INFER] Predicted label: {pred_label}")
    print(f"[INFER] Confidence: {confidence:.6f}")
    print(f"[INFER] Artifacts: {run_dir}")

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, FileNotFoundError, ContractError, RegionMappingError, LandmarkStreamError) as exc:
        print(f"[INFER ERROR] {exc}")
        raise SystemExit(2)
    except Exception as exc:
        print(f"[INFER ERROR] Unexpected failure: {exc}")
        raise SystemExit(99)
