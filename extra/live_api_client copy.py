"""
Live API client for multimodal pipeline.

All functions call real remote APIs on http://88.222.212.15
and return both model outputs and precise latency metrics.
"""

import logging
import os
import time
from typing import Any

import requests
from dotenv import load_dotenv

from services.multmodal import call_fusion_predict
from services.text_to_features import call_text_predict
from services.voice_to_fetures import call_voice_predict

# Ensure .env is loaded even if this module is imported before app.py calls load_dotenv.
load_dotenv()

BASE_URL = os.getenv('ML_API_BASE_URL', 'http://88.222.212.15').rstrip('/')
TIMEOUT_SECONDS = 300

# Endpoint configuration (override any of these in .env if API paths change)
FACE_EXTRACT_URL = os.getenv('FACE_EXTRACT_URL', f'{BASE_URL}:8010/extract/video')
FACE_VECTOR_URL_TEMPLATE = os.getenv('FACE_VECTOR_URL_TEMPLATE', f'{BASE_URL}:8010/session/{{session_id}}/vector')
FACE_SCORE_URL = os.getenv('FACE_SCORE_URL', f'{BASE_URL}:8011/score')
VOICE_EXTRACT_URL = os.getenv('VOICE_EXTRACT_URL', f'{BASE_URL}:8013/extract')
VOICE_SCORE_URL = os.getenv('VOICE_SCORE_URL', f'{BASE_URL}:9100/predict')
TEXT_EXTRACT_URL = os.getenv('TEXT_EXTRACT_URL', f'{BASE_URL}:8025/analyze')
TEXT_SCORE_URL = os.getenv('TEXT_SCORE_URL', f'{BASE_URL}:9000/predict')
FUSION_SCORE_URL = os.getenv('FUSION_SCORE_URL', f'{BASE_URL}:8000/predict')

# Header names can vary by API gateway; keep configurable for quick debugging.
VOICE_API_HEADER = os.getenv('VOICE_API_HEADER', 'X-API-Key')
TEXT_API_HEADER = os.getenv('TEXT_API_HEADER', 'X-API-Key')
FUSION_API_HEADER = os.getenv('FUSION_API_HEADER', 'X-API-Key')

logger = logging.getLogger(__name__)


class LiveAPIError(RuntimeError):
    """Raised when a live API call fails or returns invalid data."""


REQUIRED_ENV_GROUPS = {
    'face_extract': ('EXTRACTION_API_KEY',),
    'face_score': ('SCORING_API_KEY',),
    'voice_extract': ('VOICE_EXTRACT_API_KEY', 'AUDIO_API_KEY'),
    'voice_model': ('VOICE_MODEL_API_KEY', 'AUDIO_API_KEY', 'API_KEY'),
    'text_extract': ('TEXT_EXTRACT_API_KEY', 'TEXT_ANALYSIS_API_KEY'),
    'text_model': ('TEXT_MODEL_API_KEY', 'TEXT_ANALYSIS_API_KEY', 'API_KEY'),
    'fusion_pipeline': ('MODEL_API_KEY', 'FUSION_API_KEY'),
}


def validate_env_keys() -> list[str]:
    """Return list of required key-groups that are unresolved."""
    missing: list[str] = []
    for group_name, candidates in REQUIRED_ENV_GROUPS.items():
        if all(not _optional_env(k) for k in candidates):
            missing.append(f'{group_name}: one of {candidates}')
    return missing


def _require_env(name: str) -> str:
    """Read an env var, strip whitespace and literal quote chars, raise if missing."""
    raw = os.getenv(name, '')
    value = raw.strip().strip('"\'')
    if not value:
        raise LiveAPIError(
            f'Missing or empty environment variable: {name}. '
            f'Check your .env file.'
        )
    return value


def _log_request(method: str, url: str, key_name: str, key_value: str) -> None:
    """Log the target URL and a masked version of the key for debugging."""
    masked = key_value[:4] + '****' if len(key_value) >= 4 else '****'
    logger.info('[API] %s %s  key=%s(%s)', method, url, key_name, masked)


def _debug_terminal(message: str) -> None:
    """Print debug lines to terminal and logger for live troubleshooting."""
    print(message)
    logger.info(message)


def _optional_env(name: str) -> str | None:
    raw = os.getenv(name, '')
    value = raw.strip().strip('"\'')
    return value or None


def _require_any_env(*names: str) -> tuple[str, str]:
    """Return the first non-empty env value and its key name."""
    for name in names:
        val = _optional_env(name)
        if val:
            return name, val
    raise LiveAPIError(
        f'Missing required environment keys. Expected one of: {", ".join(names)}'
    )


def _safe_raise(
    resp: requests.Response,
    context: str,
    *,
    sent_headers: dict | None = None,
    key_value: str | None = None,
) -> None:
    """Raise a LiveAPIError with context info on non-2xx responses."""
    if not resp.ok:
        try:
            body = resp.json()
        except Exception:
            body = resp.text[:300]
        diag_parts = [f'{context} failed with {resp.status_code}: {body}']
        if sent_headers is not None:
            diag_parts.append(f'Headers Sent: {list(sent_headers.keys())}')
        if key_value is not None:
            diag_parts.append(f'Key Length: {len(key_value)}')
        raise LiveAPIError('  |  '.join(diag_parts))


def _seconds(start: float) -> float:
    return round(time.time() - start, 4)


def _pick_feature_vector(payload: dict[str, Any]) -> Any:
    """Extract vector/features payload from heterogeneous API responses."""
    for key in ('vector', 'features', 'embedding', 'data'):
        if key in payload:
            return payload[key]
    return payload


def _preview_mapping(payload: Any, limit: int = 8) -> dict[str, Any]:
    if isinstance(payload, dict):
        items = list(payload.items())[:limit]
        return {k: v for k, v in items}
    return {'value_type': type(payload).__name__}


def process_face(
    video_path: str | None = None,
    *,
    video_bytes: bytes | None = None,
    video_filename: str | None = None,
    video_mimetype: str | None = None,
) -> dict[str, Any]:
    """
    Face pipeline:
      1) POST /8010/extract/video (file upload)
      2) GET  /8010/session/{session_id}/vector
      3) POST /8011/score with extracted vector
    """
    latencies: dict[str, float] = {}
    extraction_api_key = _require_env('EXTRACTION_API_KEY')
    scoring_api_key = _require_env('SCORING_API_KEY')
    _debug_terminal(
        f'[FACE] extract_key=EXTRACTION_API_KEY len={len(extraction_api_key)} '
        f'score_key=SCORING_API_KEY len={len(scoring_api_key)}'
    )

    # 1) extract session
    if video_bytes is None and not video_path:
        raise LiveAPIError('No face input provided. Expected video_path or video_bytes.')

    t0 = time.time()
    url_extract = FACE_EXTRACT_URL
    _log_request('POST', url_extract, 'EXTRACTION_API_KEY', extraction_api_key)
    _headers_extract = {'X-API-Key': extraction_api_key}
    if video_bytes is not None:
        upload_name = video_filename or 'video.webm'
        upload_mimetype = video_mimetype or 'video/webm'
        extract_resp = requests.post(
            url_extract,
            files={'video': (upload_name, video_bytes, upload_mimetype)},
            headers=_headers_extract,
            timeout=TIMEOUT_SECONDS,
        )
    else:
        with open(video_path, 'rb') as f:
            extract_resp = requests.post(
                url_extract,
                files={'video': (os.path.basename(video_path), f, 'video/webm')},
                headers=_headers_extract,
                timeout=TIMEOUT_SECONDS,
            )
    latencies['face_extract_post_s'] = _seconds(t0)
    _safe_raise(extract_resp, f'POST {url_extract}', sent_headers=_headers_extract, key_value=extraction_api_key)
    extract_json = extract_resp.json()

    session_id = extract_json.get('session_id') or extract_json.get('id')
    if not session_id:
        raise LiveAPIError(f'Face extract response missing session_id: {extract_json}')

    # 2) fetch vector (with polling for async processing)
    t1 = time.time()
    url_vector = FACE_VECTOR_URL_TEMPLATE.format(session_id=session_id)
    _headers_vector = {'X-API-Key': extraction_api_key}
    
    RETRY_DELAY_SECONDS = 5
    MAX_RETRIES = max(1, TIMEOUT_SECONDS // RETRY_DELAY_SECONDS)
    vector_resp = None
    
    for attempt in range(MAX_RETRIES):
        _log_request('GET', url_vector, 'EXTRACTION_API_KEY', extraction_api_key)
        try:
            vector_resp = requests.get(
                url_vector,
                headers=_headers_vector,
                timeout=TIMEOUT_SECONDS,
            )
            if vector_resp.status_code == 200:
                logger.info(
                    '[API] Vector ready on attempt %d/%d', attempt + 1, MAX_RETRIES
                )
                break
            elif vector_resp.status_code == 404:
                if attempt < MAX_RETRIES - 1:
                    logger.info(
                        '[API] Vector not ready (404), retrying in %ds... (attempt %d/%d)',
                        RETRY_DELAY_SECONDS,
                        attempt + 1,
                        MAX_RETRIES,
                    )
                    time.sleep(RETRY_DELAY_SECONDS)
                    continue
                else:
                    raise LiveAPIError(
                        f'GET {url_vector} failed: Vector not ready after {MAX_RETRIES} attempts ({MAX_RETRIES * RETRY_DELAY_SECONDS}s timeout)'
                    )
            else:
                # Non-404, non-200: treat as error
                _safe_raise(
                    vector_resp,
                    f'GET {url_vector}',
                    sent_headers=_headers_vector,
                    key_value=extraction_api_key,
                )
        except requests.exceptions.RequestException as e:
            if attempt < MAX_RETRIES - 1:
                logger.warning(
                    '[API] Vector request failed (%s), retrying... (attempt %d/%d)',
                    str(e),
                    attempt + 1,
                    MAX_RETRIES,
                )
                time.sleep(RETRY_DELAY_SECONDS)
                continue
            else:
                raise LiveAPIError(
                    f'GET {url_vector} failed after {MAX_RETRIES} attempts: {str(e)}'
                )
    
    if vector_resp is None:
        raise LiveAPIError(
            f'GET {url_vector} failed: No response after {MAX_RETRIES} retries'
        )
    
    latencies['face_vector_get_s'] = _seconds(t1)
    vector_json = vector_resp.json()
    face_vector = _pick_feature_vector(vector_json)
    _debug_terminal(f'[FACE] vector_preview={_preview_mapping(face_vector)}')

    # 3) score
    t2 = time.time()
    url_score = FACE_SCORE_URL
    _log_request('POST', url_score, 'SCORING_API_KEY', scoring_api_key)
    _headers_score = {'X-API-Key': scoring_api_key}
    score_resp = requests.post(
        url_score,
        json={'vector': face_vector},
        headers=_headers_score,
        timeout=TIMEOUT_SECONDS,
    )
    latencies['face_score_post_s'] = _seconds(t2)
    _safe_raise(score_resp, f'POST {url_score}', sent_headers=_headers_score, key_value=scoring_api_key)
    score_json = score_resp.json()
    _debug_terminal(f'[FACE] score_response={_preview_mapping(score_json)}')

    latencies['face_total_s'] = round(
        latencies['face_extract_post_s'] + latencies['face_vector_get_s'] + latencies['face_score_post_s'],
        4,
    )

    return {
        'extract_response': extract_json,
        'vector_response': vector_json,
        'score_response': score_json,
        'latency': latencies,
    }


def process_voice(
    audio_path: str | None = None,
    *,
    audio_bytes: bytes | None = None,
    audio_filename: str | None = None,
    audio_mimetype: str | None = None,
) -> dict[str, Any]:
    """
    Voice pipeline:
      1) POST /8013/extract (file upload)
      2) POST /9100/predict with features object
    
    If audio_path is None (e.g., developer test mode), return dummy data.
    """
    latencies: dict[str, float] = {}

    # Handle None audio_path (developer test mode)
    if audio_path is None and audio_bytes is None:
        logger.warning('[Voice] No audio path provided; returning dummy voice response')
        return {
            'extract_response': {'features': [0.0] * 128},
            'score_response': {'label': 'neutral', 'probability': 0.5},
            'latency': {'voice_extract_post_s': 0.0, 'voice_score_post_s': 0.0, 'voice_total_s': 0.0},
        }

    voice_extract_key_name, voice_extract_api_key = _require_any_env(
        'VOICE_EXTRACT_API_KEY', 'AUDIO_API_KEY'
    )
    voice_model_key_name, voice_model_api_key = _require_any_env(
        'VOICE_MODEL_API_KEY', 'AUDIO_API_KEY', 'API_KEY'
    )
    _debug_terminal(
        f'[VOICE] extract_key={voice_extract_key_name} len={len(voice_extract_api_key)} '
        f'model_key={voice_model_key_name} len={len(voice_model_api_key)} header={VOICE_API_HEADER}'
    )

    t0 = time.time()
    url_extract = VOICE_EXTRACT_URL
    _log_request('POST', url_extract, voice_extract_key_name, voice_extract_api_key)
    extract_headers = {VOICE_API_HEADER: voice_extract_api_key}
    if audio_bytes is not None:
        upload_name = audio_filename or 'audio.webm'
        upload_mimetype = audio_mimetype or 'audio/wav'
        extract_resp = requests.post(
            url_extract,
            files={'file': (upload_name, audio_bytes, upload_mimetype)},
            headers=extract_headers,
            timeout=TIMEOUT_SECONDS,
        )
    else:
        with open(audio_path, 'rb') as f:
            extract_resp = requests.post(
                url_extract,
                files={'file': (os.path.basename(audio_path), f, 'audio/wav')},
                headers=extract_headers,
                timeout=TIMEOUT_SECONDS,
            )

    latencies['voice_extract_post_s'] = _seconds(t0)
    if extract_resp is None:
        raise LiveAPIError(f'POST {url_extract} failed: no response')

    _safe_raise(
        extract_resp,
        f'POST {url_extract}',
        sent_headers=extract_headers,
        key_value=voice_extract_api_key,
    )

    extract_json = extract_resp.json()
    _debug_terminal(f'[VOICE] extract_response={_preview_mapping(extract_json)}')

    features = extract_json.get('features', extract_json)
    _debug_terminal(f'[VOICE] extracted_features={_preview_mapping(features)}')

    t1 = time.time()
    url_predict = VOICE_SCORE_URL
    _log_request('POST', url_predict, voice_model_key_name, voice_model_api_key)
    score_json = call_voice_predict(
        {'features': features},
        timeout=TIMEOUT_SECONDS,
        predict_url_override=url_predict,
        header_name_override=VOICE_API_HEADER,
        api_key_override=voice_model_api_key,
    )
    _debug_terminal(f'[VOICE] score_response={_preview_mapping(score_json)}')

    latencies['voice_score_post_s'] = _seconds(t1)

    latencies['voice_total_s'] = round(
        latencies['voice_extract_post_s'] + latencies['voice_score_post_s'],
        4,
    )

    return {
        'extract_response': extract_json,
        'score_response': score_json,
        'latency': latencies,
    }


def process_text(transcript_text: str) -> dict[str, Any]:
    """
    Text pipeline:
      1) POST /8025/analyze with transcript text
      2) POST /9000/predict with features object
    """
    latencies: dict[str, float] = {}
    text_extract_key_name, text_extract_api_key = _require_any_env(
        'TEXT_EXTRACT_API_KEY', 'TEXT_ANALYSIS_API_KEY'
    )
    text_model_key_name, text_model_api_key = _require_any_env(
        'TEXT_MODEL_API_KEY', 'TEXT_ANALYSIS_API_KEY', 'API_KEY'
    )
    _debug_terminal(
        f'[TEXT] extract_key={text_extract_key_name} len={len(text_extract_api_key)} '
        f'model_key={text_model_key_name} len={len(text_model_api_key)} header={TEXT_API_HEADER}'
    )

    if not transcript_text or not transcript_text.strip():
        raise LiveAPIError('Empty transcript text for text pipeline.')

    t0 = time.time()
    url_analyze = TEXT_EXTRACT_URL
    _log_request('POST', url_analyze, text_extract_key_name, text_extract_api_key)
    analyze_headers = {TEXT_API_HEADER: text_extract_api_key}
    extract_resp = requests.post(
        url_analyze,
        json={'text': transcript_text.strip()},
        headers=analyze_headers,
        timeout=TIMEOUT_SECONDS,
    )

    latencies['text_extract_post_s'] = _seconds(t0)
    _safe_raise(
        extract_resp,
        f'POST {url_analyze}',
        sent_headers=analyze_headers,
        key_value=text_extract_api_key,
    )

    extract_json = extract_resp.json()
    _debug_terminal(f'[TEXT] extract_response={_preview_mapping(extract_json)}')

    features = extract_json.get('features', extract_json)
    _debug_terminal(f'[TEXT] extracted_features={_preview_mapping(features)}')

    t1 = time.time()
    url_predict = TEXT_SCORE_URL
    _log_request('POST', url_predict, text_model_key_name, text_model_api_key)
    score_json = call_text_predict(
        {'features': features},
        timeout=TIMEOUT_SECONDS,
        predict_url_override=url_predict,
        header_name_override=TEXT_API_HEADER,
        api_key_override=text_model_api_key,
    )
    _debug_terminal(f'[TEXT] score_response={_preview_mapping(score_json)}')
    latencies['text_score_post_s'] = _seconds(t1)

    latencies['text_total_s'] = round(
        latencies['text_extract_post_s'] + latencies['text_score_post_s'],
        4,
    )

    return {
        'extract_response': extract_json,
        'score_response': score_json,
        'latency': latencies,
    }


def fuse_multimodal(combined_features: dict[str, Any]) -> dict[str, Any]:
    """
    Fusion pipeline:
      POST /8000/predict with combined feature object.
    """
    fusion_key_name, fusion_api_key = _require_any_env('MODEL_API_KEY', 'FUSION_API_KEY')
    _debug_terminal(
        f'[FUSION] using_key={fusion_key_name} len={len(fusion_api_key)} header={FUSION_API_HEADER}'
    )
    t0 = time.time()
    url_fuse = FUSION_SCORE_URL
    _log_request('POST', url_fuse, fusion_key_name, fusion_api_key)
    score_json = call_fusion_predict(
        combined_features,
        timeout=TIMEOUT_SECONDS,
        predict_url_override=url_fuse,
        header_name_override=FUSION_API_HEADER,
        api_key_override=fusion_api_key,
    )
    _debug_terminal(f'[FUSION] score_response={_preview_mapping(score_json)}')
    latency = _seconds(t0)

    return {
        'score_response': score_json,
        'latency': {'fusion_post_s': latency, 'fusion_total_s': latency},
    }
