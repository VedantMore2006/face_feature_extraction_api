# FastAPI API Key Authentication Implementation Guide

## Overview

This document describes the complete process for adding secure API key authentication to a FastAPI service using environment variables and OpenAPI/Swagger UI integration.

**Target Audience:** Developers implementing API key authentication in FastAPI applications.

---

## Requirements

- FastAPI application (any service)
- Environment variable support via `python-dotenv`
- Swagger UI documentation (built-in with FastAPI)

---

## Implementation Steps

### Step 1: Add Required Dependencies

Add to `requirements.txt`:
```
python-dotenv==1.1.1
```

These packages provide:
- `python-dotenv`: Load environment variables from `.env` file
- FastAPI already includes `fastapi.security` and `fastapi.openapi.utils` needed for authentication

---

### Step 2: Create Environment Template File

Create `.env.example` in project root:
```
# Copy this file to .env and set a real secret before running the API.
API_KEY_NAME=CHANGE_ME
```

Replace:
- `API_KEY_NAME`: Name of your environment variable (e.g., `EXTRACTION_API_KEY`, `SCORING_API_KEY`)
- `CHANGE_ME`: Placeholder value to guide developers

Add `.env` to `.gitignore` to prevent committing secrets.

---

### Step 3: Update FastAPI Application

In your main API file (e.g., `extraction_api.py`, `scoring_api.py`):

#### 3.1 Add Imports

```python
import os
import secrets
from fastapi import Depends, FastAPI, HTTPException
from fastapi.security import APIKeyHeader
from fastapi.openapi.utils import get_openapi
from dotenv import load_dotenv
```

#### 3.2 Configure Environment Loading

After app initialization, add:

```python
project_root = Path(__file__).resolve().parent
load_dotenv(project_root / ".env")
```

This loads environment variables from `.env` file in the project root.

#### 3.3 Define API Key Configuration

```python
API_KEY_ENV_VAR = "YOUR_API_KEY_NAME"  # e.g., "EXTRACTION_API_KEY"
API_KEY_HEADER_NAME = "X-API-Key"
```

#### 3.4 Create API Key Loader Function

```python
def _load_api_key() -> str:
    key = os.getenv(API_KEY_ENV_VAR, "").strip()
    if not key:
        raise RuntimeError(
            f"Missing required environment variable {API_KEY_ENV_VAR}. "
            f"Set it before starting the API."
        )
    if key == "CHANGE_ME":
        raise RuntimeError(
            f"Invalid {API_KEY_ENV_VAR} value. Replace placeholder with a real key."
        )
    return key

EXPECTED_API_KEY = _load_api_key()
```

This:
- Loads key from environment on startup
- Fails loudly if key is missing or still has placeholder value
- Happens once at module load time

#### 3.5 Initialize FastAPI App with Description

```python
app = FastAPI(
    title="Your API Title",
    version="1.0.0",
    description="Your API description with authentication.",
)
```

#### 3.6 Create API Key Header Dependency

```python
api_key_header = APIKeyHeader(
    name=API_KEY_HEADER_NAME,
    description="API key required for authentication. Obtain from deployment configuration.",
    auto_error=False,
)
```

#### 3.7 Create Validation Function

```python
def require_api_key(provided_key: str | None = Depends(api_key_header)) -> None:
    """Validate API key from X-API-Key header."""
    if not provided_key or not secrets.compare_digest(provided_key, EXPECTED_API_KEY):
        raise HTTPException(status_code=401, detail="Invalid or missing API key")
```

Key points:
- Uses `secrets.compare_digest()` for timing-attack resistance
- Returns `None` (dependency only for side effects)
- Raises `HTTPException(401)` on auth failure

#### 3.8 Create OpenAPI Security Scheme

```python
def custom_openapi():
    """Add security scheme to OpenAPI spec for Swagger UI display."""
    if app.openapi_schema:
        return app.openapi_schema
    
    openapi_schema = get_openapi(
        title=app.title,
        version=app.version,
        description=app.description,
        routes=app.routes,
    )
    
    openapi_schema["components"]["securitySchemes"] = {
        "api_key": {
            "type": "apiKey",
            "in": "header",
            "name": API_KEY_HEADER_NAME,
            "description": "API key for authentication.",
        }
    }
    
    # Apply security to protected endpoints
    for path, path_item in openapi_schema["paths"].items():
        if path != "/health":  # Exclude health check
            for operation in path_item.values():
                if isinstance(operation, dict) and "operationId" in operation:
                    operation["security"] = [{"api_key": []}]
    
    app.openapi_schema = openapi_schema
    return app.openapi_schema

app.openapi = custom_openapi
```

This:
- Registers `X-API-Key` as a security scheme in OpenAPI spec
- Makes it appear in Swagger UI with "Authorize" button
- Applies security requirement to all endpoints except `/health`

#### 3.9 Protect Endpoints

Add `_auth: None = Depends(require_api_key)` parameter to protected endpoints:

```python
@app.post("/extract/video")
async def extract_from_video(
    video: UploadFile = File(...),
    mode: str = Query("balanced"),
    _auth: None = Depends(require_api_key),  # <- Add this
) -> Dict[str, Any]:
    # endpoint logic
    pass

@app.get("/extract/session/{session_id}/vector")
def get_session_vector(
    session_id: str,
    _auth: None = Depends(require_api_key),  # <- Add this
) -> Dict[str, Any]:
    # endpoint logic
    pass
```

Leave `/health` unprotected for load balancers and health checks.

---

## Step 4: User Documentation

Update API documentation (README, API_SERVICES.md) to include:

### Local Development

```bash
# 1. Create .env from template
cp .env.example .env

# 2. Set API key in .env
EXTRACTION_API_KEY=your_random_secret

# 3. Generate random key (optional)
python3 -c "import secrets; print(secrets.token_urlsafe(32))"
```

### Swagger UI Usage

1. Start API: `uvicorn your_api:app --reload`
2. Open: `http://127.0.0.1:8000/docs`
3. Click "Authorize" button (top right)
4. Enter API key in `X-API-Key` field
5. All requests automatically include key

### curl Usage

```bash
curl -X POST "http://127.0.0.1:8001/endpoint" \
  -H "X-API-Key: your_api_key_here" \
  -d '{"param": "value"}'
```

### Python Usage

```python
import requests

headers = {"X-API-Key": "your_api_key_here"}
response = requests.post(
    "http://127.0.0.1:8001/endpoint",
    headers=headers,
    json={"param": "value"}
)
```

### Production Deployment

- Store API key in cloud secret manager (AWS Secrets Manager, GCP Secret Manager, Azure Key Vault)
- Inject as environment variable at container startup
- Rotate keys periodically
- Use separate keys for dev/staging/prod

---

## Validation Checklist

- ✅ `python-dotenv` added to `requirements.txt`
- ✅ `.env.example` created with placeholder value
- ✅ `.env` added to `.gitignore`
- ✅ `os`, `secrets`, `dotenv.load_dotenv` imported
- ✅ `load_dotenv()` called in app startup
- ✅ API key loader function created with error handling
- ✅ `APIKeyHeader` dependency created
- ✅ `require_api_key()` validation function created
- ✅ OpenAPI security scheme defined
- ✅ Security applied to protected endpoints (via `Depends(require_api_key)`)
- ✅ `/health` endpoint left unprotected
- ✅ Application starts without errors
- ✅ Users documented on Swagger UI usage
- ✅ Users documented on curl/Python usage

---

## Error Handling

| Scenario | Error | Resolution |
|----------|-------|-----------|
| Missing `.env` file | App starts normally, key loaded from environment | Ensure `EXTRACTION_API_KEY` env var is set |
| Missing `EXTRACTION_API_KEY` env var | `RuntimeError` on startup | Set environment variable before starting |
| `API_KEY=CHANGE_ME` | `RuntimeError` on startup | Replace placeholder with real secret |
| Missing header in request | HTTP 401 | Add `X-API-Key: <key>` to request |
| Wrong API key value | HTTP 401 | Verify key matches `EXTRACTION_API_KEY` value |

---

## Security Notes

1. **Timing-attack resistant:** Uses `secrets.compare_digest()` instead of `==`
2. **No key logging:** Key value never printed to stdout/logs
3. **Early validation:** Key validated on app startup, not on first request
4. **Placeholder detection:** Catches accidental deployment with placeholder value
5. **Separate dev/prod:** `.env` never committed; cloud secrets used in production
6. **Health check bypass:** `/health` public for load balancer compatibility

---

## Quick Reference

| Component | What It Does |
|-----------|--------------|
| `.env.example` | Template for developers |
| `load_dotenv()` | Loads `.env` into environment on startup |
| `_load_api_key()` | Reads and validates key from environment |
| `api_key_header` | FastAPI security scheme for header parsing |
| `require_api_key()` | Validates key on each request |
| `custom_openapi()` | Adds security scheme to Swagger UI |
| `Depends(require_api_key)` | Protects individual endpoints |

