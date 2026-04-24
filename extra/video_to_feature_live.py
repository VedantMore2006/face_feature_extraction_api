"""Face-only live API client for video feature extraction debugging.

This module intentionally tests only:
1) POST /extract/video
2) GET  /extract/session/{session_id}/vector

It does not call voice, text, scoring, or fusion services.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

BASE_URL = os.getenv("ML_API_BASE_URL", "http://88.222.212.15").rstrip("/")

# Keep defaults aligned with successful CLI checks.
FACE_EXTRACT_URL = os.getenv("FACE_EXTRACT_URL", f"{BASE_URL}:8010/extract/video")
# Important: include /extract in this path, otherwise vector polling stays at 404.
FACE_VECTOR_URL_TEMPLATE = os.getenv(
    "FACE_VECTOR_URL_TEMPLATE",
    f"{BASE_URL}:8010/extract/session/{{session_id}}/vector",
)

EXTRACT_TIMEOUT_SECONDS = int(os.getenv("FACE_EXTRACT_TIMEOUT_SECONDS", "600"))
VECTOR_REQUEST_TIMEOUT_SECONDS = int(os.getenv("FACE_VECTOR_REQUEST_TIMEOUT_SECONDS", "30"))
VECTOR_MAX_WAIT_SECONDS = int(os.getenv("FACE_VECTOR_MAX_WAIT_SECONDS", "180"))
VECTOR_RETRY_DELAY_SECONDS = int(os.getenv("FACE_VECTOR_RETRY_DELAY_SECONDS", "3"))


class LiveAPIError(RuntimeError):
    """Raised when a live API call fails or returns invalid data."""


def _require_env(name: str) -> str:
    raw = os.getenv(name, "")
    value = raw.strip().strip('"\'')
    if not value:
        raise LiveAPIError(
            f"Missing or empty environment variable: {name}. Check your .env file."
        )
    return value


def _seconds(start: float) -> float:
    return round(time.time() - start, 4)


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _safe_raise(resp: requests.Response, context: str) -> None:
    if resp.ok:
        return
    try:
        body: Any = resp.json()
    except Exception:
        body = resp.text[:500]
    raise LiveAPIError(f"{context} failed with {resp.status_code}: {body}")


def _log_api_call(method: str, url: str, header_name: str, key: str) -> None:
    masked = key[:4] + "****" if len(key) >= 4 else "****"
    logger.info("[FACE] %s %s header=%s key=%s", method, url, header_name, masked)
    print(f"[FACE] {method} {url}  header={header_name}  key={masked}")


def validate_env_keys() -> list[str]:
    missing: list[str] = []
    if not os.getenv("EXTRACTION_API_KEY", "").strip().strip('"\''):
        missing.append("EXTRACTION_API_KEY")
    return missing


def process_face_extraction_only(
    video_path: str | None = None,
    *,
    video_bytes: bytes | None = None,
    video_filename: str | None = None,
    video_mimetype: str | None = None,
    mode: str = "fast",
    allow_short: bool = True,
    min_duration_seconds: float = 1.0,
) -> dict[str, Any]:
    """Run face extraction flow only and return extraction/vector payloads with timing."""
    if video_bytes is None and not video_path:
        raise LiveAPIError("No face input provided. Expected video_path or video_bytes.")

    extraction_api_key = _require_env("EXTRACTION_API_KEY")
    api_headers = {"X-API-Key": extraction_api_key}
    latencies: dict[str, float] = {}

    extract_params = {
        "mode": mode,
        "allow_short": str(allow_short).lower(),
        "min_duration_seconds": min_duration_seconds,
    }

    started_at = _utc_now_iso()

    # Step 1: extract session id from uploaded video.
    t0 = time.time()
    _log_api_call("POST", FACE_EXTRACT_URL, "X-API-Key", extraction_api_key)
    if video_bytes is not None:
        upload_name = video_filename or "video.mp4"
        upload_type = video_mimetype or "video/mp4"
        extract_resp = requests.post(
            FACE_EXTRACT_URL,
            params=extract_params,
            files={"video": (upload_name, video_bytes, upload_type)},
            headers=api_headers,
            timeout=EXTRACT_TIMEOUT_SECONDS,
        )
    else:
        file_path = Path(video_path) # type: ignore
        mime = video_mimetype or "video/mp4"
        with file_path.open("rb") as handle:
            extract_resp = requests.post(
                FACE_EXTRACT_URL,
                params=extract_params,
                files={"video": (file_path.name, handle, mime)},
                headers=api_headers,
                timeout=EXTRACT_TIMEOUT_SECONDS,
            )

    latencies["face_extract_post_s"] = _seconds(t0)
    _safe_raise(extract_resp, f"POST {FACE_EXTRACT_URL}")
    extract_json = extract_resp.json()

    session_id = extract_json.get("session_id") or extract_json.get("id")
    if not session_id:
        raise LiveAPIError(f"Extract response missing session_id: {extract_json}")

    # Step 2: poll vector endpoint with explicit timeout diagnostics.
    t1 = time.time()
    vector_url = FACE_VECTOR_URL_TEMPLATE.format(session_id=session_id)
    max_retries = max(1, int(VECTOR_MAX_WAIT_SECONDS / VECTOR_RETRY_DELAY_SECONDS))
    attempts: list[dict[str, Any]] = []
    vector_resp: requests.Response | None = None

    for attempt in range(1, max_retries + 1):
        _log_api_call("GET", vector_url, "X-API-Key", extraction_api_key)
        try:
            current_resp = requests.get(
                vector_url,
                headers=api_headers,
                timeout=VECTOR_REQUEST_TIMEOUT_SECONDS,
            )
            attempts.append({"attempt": attempt, "status": current_resp.status_code})
            if current_resp.status_code == 200:
                vector_resp = current_resp
                break
            if current_resp.status_code != 404:
                _safe_raise(current_resp, f"GET {vector_url}")
            if attempt < max_retries:
                time.sleep(VECTOR_RETRY_DELAY_SECONDS)
        except requests.exceptions.Timeout:
            attempts.append({"attempt": attempt, "status": "timeout"})
            if attempt < max_retries:
                time.sleep(VECTOR_RETRY_DELAY_SECONDS)
        except requests.exceptions.RequestException as exc:
            attempts.append({"attempt": attempt, "status": f"error:{type(exc).__name__}"})
            if attempt < max_retries:
                time.sleep(VECTOR_RETRY_DELAY_SECONDS)
            else:
                raise LiveAPIError(f"GET {vector_url} failed: {exc}") from exc

    latencies["face_vector_get_s"] = _seconds(t1)

    if vector_resp is None:
        raise LiveAPIError(
            "Vector polling timed out. "
            f"session_id={session_id} vector_url={vector_url} "
            f"waited={VECTOR_MAX_WAIT_SECONDS}s attempts={attempts}"
        )

    vector_json = vector_resp.json()
    latencies["face_total_s"] = round(
        latencies["face_extract_post_s"] + latencies["face_vector_get_s"],
        4,
    )

    # Explicit timing modality for quick diagnosis and dashboards.
    timing = {
        "started_at": started_at,
        "finished_at": _utc_now_iso(),
        "extraction_only_s": latencies["face_extract_post_s"],
        "vector_fetch_s": latencies["face_vector_get_s"],
        "end_to_end_s": latencies["face_total_s"],
    }

    return {
        "extract_response": extract_json,
        "vector_response": vector_json,
        "latency": latencies,
        "timing": timing,
        "debug": {
            "session_id": session_id,
            "extract_url": FACE_EXTRACT_URL,
            "vector_url": vector_url,
            "attempts": attempts,
            "timeouts": {
                "extract_timeout_s": EXTRACT_TIMEOUT_SECONDS,
                "vector_request_timeout_s": VECTOR_REQUEST_TIMEOUT_SECONDS,
                "vector_max_wait_s": VECTOR_MAX_WAIT_SECONDS,
                "vector_retry_delay_s": VECTOR_RETRY_DELAY_SECONDS,
            },
        },
    }


def process_face(
    video_path: str | None = None,
    *,
    video_bytes: bytes | None = None,
    video_filename: str | None = None,
    video_mimetype: str | None = None,
) -> dict[str, Any]:
    """Backward-compatible alias; now face-extraction only."""
    return process_face_extraction_only(
        video_path=video_path,
        video_bytes=video_bytes,
        video_filename=video_filename,
        video_mimetype=video_mimetype,
    )


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Face-only extraction API tester")
    parser.add_argument("--video", default="test.mp4", help="Path to video file")
    parser.add_argument("--mode", default="fast", choices=["fast", "balanced", "accurate"])
    parser.add_argument("--allow-short", action="store_true", default=True)
    parser.add_argument("--min-duration", type=float, default=1.0)
    parser.add_argument("--pretty", action="store_true", help="Pretty-print JSON output")
    parser.add_argument(
        "--timing-only",
        action="store_true",
        help="Print only timing metrics as compact JSON",
    )
    parser.add_argument(
        "--print-timing-summary",
        action="store_true",
        help="Print human-readable timing summary line",
    )
    return parser


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    args = _build_arg_parser().parse_args()
    missing_keys = validate_env_keys()
    if missing_keys:
        raise SystemExit(f"Missing env keys: {missing_keys}")

    result = process_face_extraction_only(
        video_path=args.video,
        mode=args.mode,
        allow_short=args.allow_short,
        min_duration_seconds=args.min_duration,
    )

    if args.print_timing_summary:
        timing = result.get("timing", {})
        print(
            "TIMING "
            f"extract={timing.get('extraction_only_s', 'n/a')}s "
            f"vector={timing.get('vector_fetch_s', 'n/a')}s "
            f"total={timing.get('end_to_end_s', 'n/a')}s"
        )

    if args.timing_only:
        print(json.dumps(result.get("timing", {}), indent=2 if args.pretty else None))
    elif args.pretty:
        print(json.dumps(result, indent=2))
    else:
        print(json.dumps(result))
