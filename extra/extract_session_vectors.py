#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


DEFAULT_SESSIONS_DIR = "reports/api_sessions"
DEFAULT_VECTORS_DIR = "reports/api_vectors"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract vector-only payloads from API session JSON files."
    )
    parser.add_argument(
        "--project-root",
        default=str(Path(__file__).resolve().parent),
        help="Project root path. Defaults to this script directory.",
    )
    parser.add_argument(
        "--session-id",
        default="",
        help="Specific session ID to process. If omitted, latest session file is used.",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Process all session files in reports/api_sessions.",
    )
    return parser.parse_args()


def _load_session_payload(session_file: Path) -> dict[str, Any]:
    payload = json.loads(session_file.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Invalid JSON object in {session_file}")
    if "session_id" not in payload:
        raise ValueError(f"Missing session_id in {session_file}")
    if "vector" not in payload or not isinstance(payload["vector"], dict):
        raise ValueError(f"Missing or invalid vector in {session_file}")
    return payload


def _write_vector_file(payload: dict[str, Any], output_file: Path) -> None:
    vector_payload = {
        "session_id": str(payload["session_id"]),
        "vector": payload["vector"],
    }
    output_file.write_text(json.dumps(vector_payload, indent=2), encoding="utf-8")


def _find_latest_session_file(sessions_dir: Path) -> Path:
    candidates = [p for p in sessions_dir.glob("*.json") if p.is_file()]
    if not candidates:
        raise FileNotFoundError(f"No session files found in {sessions_dir}")
    return max(candidates, key=lambda p: p.stat().st_mtime)


def _process_one(session_file: Path, vectors_dir: Path) -> Path:
    payload = _load_session_payload(session_file)
    output_file = vectors_dir / f"{payload['session_id']}.json"
    _write_vector_file(payload, output_file)
    return output_file


def main() -> int:
    args = parse_args()
    project_root = Path(args.project_root).resolve()
    sessions_dir = project_root / DEFAULT_SESSIONS_DIR
    vectors_dir = project_root / DEFAULT_VECTORS_DIR

    if not sessions_dir.exists():
        raise FileNotFoundError(f"Sessions directory not found: {sessions_dir}")
    vectors_dir.mkdir(parents=True, exist_ok=True)

    if args.all:
        session_files = sorted(p for p in sessions_dir.glob("*.json") if p.is_file())
        if not session_files:
            raise FileNotFoundError(f"No session files found in {sessions_dir}")
        for session_file in session_files:
            output_file = _process_one(session_file, vectors_dir)
            print(f"[OK] {session_file.name} -> {output_file}")
        print(f"[OK] Wrote {len(session_files)} vector files to {vectors_dir}")
        return 0

    if args.session_id:
        session_file = sessions_dir / f"{args.session_id}.json"
        if not session_file.exists():
            raise FileNotFoundError(f"Session file not found: {session_file}")
    else:
        session_file = _find_latest_session_file(sessions_dir)

    output_file = _process_one(session_file, vectors_dir)
    print(f"[OK] Source session: {session_file}")
    print(f"[OK] Vector file:   {output_file}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
