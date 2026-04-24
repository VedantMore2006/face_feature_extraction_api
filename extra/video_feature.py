from __future__ import annotations

import argparse
import mimetypes
from pathlib import Path

import requests

from common import (
    base_url,
    clean_env,
    debug_error,
    debug_step,
    download_url_binary,
    fetch_gridfs_file,
    fetch_recent_gridfs_files,
    first_env,
    show_response,
    write_temp_binary,
)


def main() -> None:
    debug_step('S1', 'Parsing command-line arguments')
    parser = argparse.ArgumentParser(description='Test face/video extractor API only.')
    parser.add_argument('--source', choices=['local', 'mongo', 'cloudinary'], default='mongo')
    parser.add_argument('--file', help='Path to local video file (.mp4/.webm).')
    parser.add_argument('--mongo-file-id', help='GridFS ObjectId to load from Mongo (default: latest).')
    parser.add_argument('--mongo-collection', default='fs', help='GridFS collection name (default: fs).')
    parser.add_argument('--mongo-try-latest', type=int, default=3, help='When mongo-file-id not set, try N latest files.')
    parser.add_argument('--url', help='Cloudinary/public URL for video source.')
    parser.add_argument('--timeout', type=int, default=120, help='Request timeout in seconds.')
    args = parser.parse_args()

    debug_step('S2', f'Resolving video input source: {args.source}')
    try:
        if args.source == 'local':
            if not args.file:
                raise RuntimeError('For --source local, provide --file')
            p = Path(args.file)
            if not p.exists():
                raise FileNotFoundError(f'Video file not found: {p}')
            video_candidates = [(p.name, p.read_bytes(), None, 'local')]
            debug_step('S2', f'Using local video file: {p}')
        elif args.source == 'mongo':
            if args.mongo_file_id:
                video_candidates = [fetch_gridfs_file(args.mongo_collection, args.mongo_file_id)]
            else:
                video_candidates = fetch_recent_gridfs_files(args.mongo_collection, args.mongo_try_latest)
            if not video_candidates:
                raise RuntimeError('No video candidates found in Mongo GridFS')
            debug_step('S2', f'Loaded {len(video_candidates)} Mongo candidate(s) from {args.mongo_collection}')
        else:
            if not args.url:
                raise RuntimeError('For --source cloudinary, provide --url')
            name, data, _ = download_url_binary(args.url, timeout=args.timeout)
            video_candidates = [(name, data, None, 'url')]
            debug_step('S2', f'Downloaded cloudinary/public URL payload: {args.url}')
    except Exception as exc:
        debug_error('S2', exc)
        raise

    debug_step('S3', 'Resolving endpoint and API auth')
    url = clean_env('FACE_EXTRACT_URL') or f"{base_url()}:8010/extract/video"
    header_name = clean_env('FACE_EXTRACT_HEADER') or 'X-API-Key'
    key_name, api_key = first_env('EXTRACTION_API_KEY')
    headers = {header_name: api_key}

    print(f'[VIDEO] URL={url}')
    print(f'[VIDEO] KEY={key_name} LEN={len(api_key)} HEADER={header_name}')

    debug_step('S4', 'Sending extractor request with video multipart payload')
    resp = None
    for idx, (name, data, _, source_id) in enumerate(video_candidates, start=1):
        suffix = Path(name).suffix or '.webm'
        temp_path = write_temp_binary(data, suffix=suffix)
        mime_type, _ = mimetypes.guess_type(str(temp_path))
        if not mime_type:
            mime_type = 'video/webm'

        print(f'[VIDEO] ATTEMPT={idx}/{len(video_candidates)} source_id={source_id} file={name}')
        try:
            with temp_path.open('rb') as f:
                resp = requests.post(
                    url,
                    files={'video': (Path(name).name, f, mime_type)},
                    headers=headers,
                    timeout=args.timeout,
                )
            debug_step('S4', f'Attempt {idx} -> HTTP {resp.status_code}')
            if resp.ok:
                break
        finally:
            if temp_path.exists():
                debug_step('S5', f'Removing temp file: {temp_path}')
                temp_path.unlink(missing_ok=True)

    if resp is None:
        raise RuntimeError('No response returned from video extractor attempts')

    debug_step('S6', 'Parsing response body')
    try:
        body = resp.json()
        debug_step('S6', 'Response parsed as JSON')
    except Exception:
        body = resp.text
        debug_step('S6', 'Response parsed as plain text')
    show_response('POST video extractor', resp.status_code, body)


if __name__ == '__main__':
    main()
