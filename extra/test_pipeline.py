"""
End-to-end API test: upload video → extract vector → score risk.

Usage (from project root, with both APIs running):
    conda run -n face_env python3 api/test_pipeline.py assets/test.mp4

Optional flags:
    --extract-port  PORT   (default 8010)
    --score-port    PORT   (default 8002)
    --mode          accurate|balanced|fast  (default balanced)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import requests


EXTRACT_BASE = "http://127.0.0.1:{port}"
SCORE_BASE   = "http://127.0.0.1:{port}"


def upload_video(video_path: Path, extract_port: int, mode: str) -> str:
    url = f"http://127.0.0.1:{extract_port}/extract/video"
    params = {"mode": mode, "allow_short": "true"}
    print(f"[1/3] Uploading {video_path.name} → {url}")
    with video_path.open("rb") as fh:
        resp = requests.post(url, params=params, files={"video": fh}, timeout=120)
    resp.raise_for_status()
    data = resp.json()
    session_id = data["session_id"]
    print(f"      session_id={session_id}  features={data['vector_feature_count']}")
    return session_id


def fetch_vector(session_id: str, extract_port: int) -> dict:
    url = f"http://127.0.0.1:{extract_port}/extract/session/{session_id}/vector"
    print(f"[2/3] Fetching vector → {url}")
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    vector = resp.json()["vector"]
    print(f"      {len(vector)} features retrieved")
    return vector


def score_vector(vector: dict, score_port: int) -> dict:
    url = f"http://127.0.0.1:{score_port}/score"
    body = {"vector": vector}
    print(f"[3/3] Scoring vector → {url}")
    resp = requests.post(url, json=body, timeout=30)
    resp.raise_for_status()
    return resp.json()


def main() -> None:
    parser = argparse.ArgumentParser(description="End-to-end API test")
    parser.add_argument("video", type=Path, help="Path to video file")
    parser.add_argument("--extract-port", type=int, default=8010)
    parser.add_argument("--score-port",   type=int, default=8011)
    parser.add_argument("--mode", default="balanced", choices=["accurate", "balanced", "fast"])
    args = parser.parse_args()

    if not args.video.exists():
        sys.exit(f"Video not found: {args.video}")

    try:
        session_id = upload_video(args.video, args.extract_port, args.mode)
        vector     = fetch_vector(session_id, args.extract_port)
        result     = score_vector(vector, args.score_port)
    except requests.HTTPError as exc:
        sys.exit(f"HTTP error: {exc.response.status_code} — {exc.response.text}")
    except requests.ConnectionError as exc:
        sys.exit(f"Connection refused — is the API running?\n  {exc}")

    dominant = result["dominant_risk"]
    others   = result["other_risks"]

    print()
    print("=" * 45)
    print(f"  DOMINANT RISK: {dominant['label'].upper():<20} {dominant['probability']*100:5.1f}%")
    print("-" * 45)
    for row in others:
        print(f"  {row['label']:<28} {row['probability']*100:5.1f}%")
    print("=" * 45)


if __name__ == "__main__":
    main()
