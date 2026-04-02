from __future__ import annotations

import json
import os
import secrets
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

from fastapi import Depends, FastAPI, File, HTTPException, Query, UploadFile
from fastapi.security import APIKeyHeader
from fastapi.openapi.utils import get_openapi
from dotenv import load_dotenv

from predict_video_risk import (
    MODE_PROFILES,
    extract_raw_timeseries,
    engineer_session_vector,
    resolve_expected_feature_order,
)
from src.config import RuntimeConfig, validate_runtime_config
from src.contracts import load_feature_contract
from src.regions import validate_runtime_contract


APP_TITLE = "Facial Extraction API"
DEFAULT_MODEL_DIR = "reports/model_training/run_20260324_171117"
DEFAULT_LABEL_COL = "condition_label"
API_KEY_ENV_VAR = "EXTRACTION_API_KEY"
API_KEY_HEADER_NAME = "X-API-Key"

project_root = Path(__file__).resolve().parent
load_dotenv(project_root / ".env")

sessions_dir = project_root / "reports" / "api_sessions"
upload_dir = project_root / "reports" / "api_uploads"
vectors_dir = project_root / "reports" / "api_vectors"
sessions_dir.mkdir(parents=True, exist_ok=True)
upload_dir.mkdir(parents=True, exist_ok=True)
vectors_dir.mkdir(parents=True, exist_ok=True)

app = FastAPI(
    title=APP_TITLE,
    version="1.0.0",
    description="Privacy-first facial feature extraction API with authentication.",
)
api_key_header = APIKeyHeader(
    name=API_KEY_HEADER_NAME,
    description="API key required for authentication. Obtain from deployment configuration.",
    auto_error=False,
)


def _load_api_key() -> str:
    key = os.getenv(API_KEY_ENV_VAR, "").strip()
    if not key:
        raise RuntimeError(
            f"Missing required environment variable {API_KEY_ENV_VAR}. "
            f"Set it before starting {APP_TITLE}."
        )
    if key == "CHANGE_ME":
        raise RuntimeError(
            f"Invalid {API_KEY_ENV_VAR} value. Replace placeholder value 'CHANGE_ME' with a real key."
        )
    return key


EXPECTED_API_KEY = _load_api_key()


def require_api_key(provided_key: str | None = Depends(api_key_header)) -> None:
    """Validate API key from X-API-Key header."""
    if not provided_key or not secrets.compare_digest(provided_key, EXPECTED_API_KEY):
        raise HTTPException(status_code=401, detail="Invalid or missing API key")


def custom_openapi():
    """Add security scheme to OpenAPI spec for Swagger UI display."""
    if app.openapi_schema:
        return app.openapi_schema
    
    openapi_schema = get_openapi(
        title=APP_TITLE,
        version="1.0.0",
        description="Privacy-first facial feature extraction API with authentication.",
        routes=app.routes,
    )
    
    openapi_schema["components"]["securitySchemes"] = {
        "api_key": {
            "type": "apiKey",
            "in": "header",
            "name": API_KEY_HEADER_NAME,
            "description": "API key for authentication. Required for all extraction endpoints.",
        }
    }
    
    # Apply security to protected endpoints, but not /health
    for path, path_item in openapi_schema["paths"].items():
        if path != "/health":
            for operation in path_item.values():
                if isinstance(operation, dict) and "operationId" in operation:
                    operation["security"] = [{"api_key": []}]
    
    app.openapi_schema = openapi_schema
    return app.openapi_schema


app.openapi = custom_openapi


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "service": APP_TITLE}


def _expected_schema(model_dir: Path, training_report: str, label_col: str) -> tuple[list[str], Path]:
    training_report_path = Path(training_report) if training_report else (model_dir / "training_report.json")
    if not training_report_path.is_absolute():
        training_report_path = (project_root / training_report_path).resolve()
    if not training_report_path.exists():
        raise FileNotFoundError(f"Training report not found: {training_report_path}")

    expected_features, _training_input_path, _report_classes = resolve_expected_feature_order(
        project_root=project_root,
        training_report_path=training_report_path,
        label_col=label_col,
    )
    return expected_features, training_report_path


def _save_session_outputs(session_id: str, payload: Dict[str, Any]) -> None:
    session_file = sessions_dir / f"{session_id}.json"
    vector_file = vectors_dir / f"{session_id}.json"

    vector_payload = {
        "session_id": session_id,
        "vector": payload["vector"],
    }
    session_file.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    vector_file.write_text(json.dumps(vector_payload, indent=2), encoding="utf-8")


@app.post("/extract/video")
async def extract_from_video(
    video: UploadFile = File(...),
    mode: str = Query("balanced", pattern="^(accurate|balanced|fast)$"),
    frame_stride: int = Query(0, ge=0),
    min_duration_seconds: float = Query(150.0, gt=0.0),
    allow_short: bool = Query(False),
    model_dir: str = Query(DEFAULT_MODEL_DIR),
    training_report: str = Query(""),
    label_col: str = Query(DEFAULT_LABEL_COL),
    _auth: None = Depends(require_api_key),
) -> Dict[str, Any]:
    try:
        cfg = RuntimeConfig()
        validate_runtime_config(cfg)
        _ = validate_runtime_contract()
        contract = load_feature_contract(project_root)

        model_dir_path = Path(model_dir)
        if not model_dir_path.is_absolute():
            model_dir_path = (project_root / model_dir_path).resolve()
        expected_features, training_report_path = _expected_schema(
            model_dir=model_dir_path,
            training_report=training_report,
            label_col=label_col,
        )

        session_id = uuid.uuid4().hex
        suffix = Path(video.filename or "upload.mp4").suffix or ".mp4"
        saved_video = upload_dir / f"{session_id}{suffix}"

        content = await video.read()
        saved_video.write_bytes(content)

        mode_profile = MODE_PROFILES[mode]
        active_stride = int(frame_stride) if int(frame_stride) > 0 else int(mode_profile["frame_stride"])

        raw_df, extraction_stats = extract_raw_timeseries(
            video_file=saved_video,
            cfg=cfg,
            feature_order=contract.feature_order,
            frame_stride=active_stride,
        )

        duration_seconds = float(raw_df["timestamp"].iloc[-1] - raw_df["timestamp"].iloc[0])
        if duration_seconds < float(min_duration_seconds) and not allow_short:
            raise ValueError(
                f"Video too short ({duration_seconds:.2f}s). Recommended >= {min_duration_seconds:.2f}s."
            )

        engineered = engineer_session_vector(raw_df=raw_df, feature_cols=contract.feature_order)
        missing = [name for name in expected_features if name not in engineered]
        if missing:
            raise ValueError(f"Engineered vector missing expected features: {missing[:10]} (total {len(missing)})")

        ml_vector = {name: float(engineered[name]) for name in expected_features}

        payload = {
            "session_id": session_id,
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "video_file": str(saved_video),
            "training_report": str(training_report_path),
            "mode": mode,
            "frame_stride": active_stride,
            "min_duration_seconds": float(min_duration_seconds),
            "allow_short": bool(allow_short),
            "frames_with_face": int(extraction_stats["frames_seen_with_face"]),
            "frames_used_for_features": int(extraction_stats["frames_used_for_features"]),
            "duration_seconds": duration_seconds,
            "vector_feature_count": int(len(ml_vector)),
            "vector": ml_vector,
        }
        _save_session_outputs(session_id=session_id, payload=payload)

        return {
            "session_id": session_id,
            "vector_feature_count": int(len(ml_vector)),
        }

    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Extraction failed: {exc}") from exc


@app.get("/extract/session/{session_id}/vector")
def get_session_vector(session_id: str, _auth: None = Depends(require_api_key)) -> Dict[str, Any]:
    session_file = sessions_dir / f"{session_id}.json"
    if not session_file.exists():
        raise HTTPException(status_code=404, detail=f"Session not found: {session_id}")

    payload = json.loads(session_file.read_text(encoding="utf-8"))
    return {
        "session_id": payload["session_id"],
        "vector": payload["vector"],
    }
