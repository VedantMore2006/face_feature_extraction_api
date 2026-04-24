from __future__ import annotations

import json
import os
from pathlib import Path

import requests


PROJECT_ROOT = Path(__file__).resolve().parent
PAYLOAD_PATH = PROJECT_ROOT / "payload.json"


def main() -> int:
    payload = json.loads(PAYLOAD_PATH.read_text(encoding="utf-8"))
    endpoint = payload.get("endpoint", "/extract/video")
    method = payload.get("method", "POST").upper()
    query_params = payload.get("query_params", {})
    headers = payload.get("headers", {})
    video_file_name = payload.get("input", {}).get("video_file")

    base_url = os.getenv("API_BASE_URL", "http://127.0.0.1:8010").rstrip("/")
    api_url = f"{base_url}{endpoint}"

    video_path = PROJECT_ROOT / video_file_name
    with video_path.open("rb") as handle:
        files = {"video": (video_path.name, handle, "video/mp4")}
        response = requests.request(
            method,
            api_url,
            params=query_params,
            headers=headers,
            files=files,
            timeout=300,
        )

    print(response.status_code)
    print(response.json())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
