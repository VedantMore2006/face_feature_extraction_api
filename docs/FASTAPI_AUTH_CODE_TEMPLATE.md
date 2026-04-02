# FastAPI API Key Authentication - Code Template

Copy these code snippets into your FastAPI application.

---

## 1. Imports (Add to Top of File)

```python
import os
import secrets
from pathlib import Path

from fastapi import Depends, FastAPI, HTTPException
from fastapi.security import APIKeyHeader
from fastapi.openapi.utils import get_openapi
from dotenv import load_dotenv
```

---

## 2. Environment Setup (After Imports, Before App Creation)

```python
# Project root for .env loading
project_root = Path(__file__).resolve().parent

# Load environment variables from .env file
load_dotenv(project_root / ".env")

# API key configuration
API_KEY_ENV_VAR = "YOUR_API_KEY_NAME"  # e.g., "EXTRACTION_API_KEY", "SCORING_API_KEY"
API_KEY_HEADER_NAME = "X-API-Key"
```

---

## 3. API Key Loader (Before App Creation)

```python
def _load_api_key() -> str:
    """Load and validate API key from environment variable on startup."""
    key = os.getenv(API_KEY_ENV_VAR, "").strip()
    if not key:
        raise RuntimeError(
            f"Missing required environment variable {API_KEY_ENV_VAR}. "
            f"Set it before starting the API."
        )
    if key == "CHANGE_ME":
        raise RuntimeError(
            f"Invalid {API_KEY_ENV_VAR} value. Replace placeholder value 'CHANGE_ME' with a real key."
        )
    return key


EXPECTED_API_KEY = _load_api_key()
```

---

## 4. FastAPI App Creation (With Description)

```python
app = FastAPI(
    title="Your API Title",
    version="1.0.0",
    description="Your API description. This API requires authentication via X-API-Key header.",
)
```

---

## 5. API Key Header & Validation (After App Creation)

```python
api_key_header = APIKeyHeader(
    name=API_KEY_HEADER_NAME,
    description="API key required for authentication. Obtain from deployment configuration.",
    auto_error=False,
)


def require_api_key(provided_key: str | None = Depends(api_key_header)) -> None:
    """Validate API key from X-API-Key header using timing-attack resistant comparison."""
    if not provided_key or not secrets.compare_digest(provided_key, EXPECTED_API_KEY):
        raise HTTPException(status_code=401, detail="Invalid or missing API key")
```

---

## 6. OpenAPI Security Scheme (After Validation Function)

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
    
    # Register X-API-Key as security scheme
    openapi_schema["components"]["securitySchemes"] = {
        "api_key": {
            "type": "apiKey",
            "in": "header",
            "name": API_KEY_HEADER_NAME,
            "description": "API key for authentication. Required for all extraction endpoints.",
        }
    }
    
    # Apply security requirement to all endpoints except /health
    for path, path_item in openapi_schema["paths"].items():
        if path != "/health":
            for operation in path_item.values():
                if isinstance(operation, dict) and "operationId" in operation:
                    operation["security"] = [{"api_key": []}]
    
    app.openapi_schema = openapi_schema
    return app.openapi_schema


app.openapi = custom_openapi
```

---

## 7. Protect Endpoints (Add to Each Protected Route)

### Unprotected Health Check (Always Public)

```python
@app.get("/health")
def health() -> dict:
    return {"status": "ok", "service": "Your Service"}
```

### Protected POST Endpoint

```python
@app.post("/your/protected/endpoint")
async def your_endpoint(
    # ... your normal parameters ...
    _auth: None = Depends(require_api_key),  # <- ADD THIS LINE
) -> dict:
    """Protected endpoint that requires API key."""
    # Your endpoint logic
    return {"result": "success"}
```

### Protected GET Endpoint

```python
@app.get("/your/protected/endpoint/{id}")
def get_protected_resource(
    id: str,
    _auth: None = Depends(require_api_key),  # <- ADD THIS LINE
) -> dict:
    """Protected endpoint that requires API key."""
    # Your endpoint logic
    return {"id": id, "data": "..."}
```

---

## 8. Configuration Files

### requirements.txt (Add)

```
python-dotenv==1.1.1
```

### .env.example (Create)

```
# Copy this file to .env and set a real secret before running the API.
YOUR_API_KEY_NAME=CHANGE_ME
```

Replace `YOUR_API_KEY_NAME` with your actual variable name (e.g., `EXTRACTION_API_KEY`).

### .gitignore (Add)

```
.env
```

---

## Complete Example Usage

### For curl:

```bash
# Without key (fails with 401)
curl -X GET "http://127.0.0.1:8000/your/protected/endpoint/123"

# With key (succeeds)
curl -X GET "http://127.0.0.1:8000/your/protected/endpoint/123" \
  -H "X-API-Key: your_actual_api_key_here"
```

### For Python:

```python
import requests

headers = {"X-API-Key": "your_actual_api_key_here"}

response = requests.get(
    "http://127.0.0.1:8000/your/protected/endpoint/123",
    headers=headers
)

print(response.json())
```

### For Swagger UI:

1. Start your API: `uvicorn your_api:app --reload`
2. Open: `http://127.0.0.1:8000/docs`
3. Click "Authorize" button (top right)
4. Enter your API key in the popup
5. All requests will automatically include the `X-API-Key` header

---

## Validation Steps

Test your implementation:

```bash
# 1. Start the API (should not error)
uvicorn your_api:app --reload

# 2. Test unprotected endpoint (should work)
curl "http://127.0.0.1:8000/health"

# 3. Test protected endpoint without key (should get 401)
curl "http://127.0.0.1:8000/your/protected/endpoint"

# 4. Test protected endpoint with wrong key (should get 401)
curl "http://127.0.0.1:8000/your/protected/endpoint" \
  -H "X-API-Key: wrong_key"

# 5. Test protected endpoint with correct key (should work)
curl "http://127.0.0.1:8000/your/protected/endpoint" \
  -H "X-API-Key: your_actual_api_key_from_.env"

# 6. Open Swagger UI at http://127.0.0.1:8000/docs and verify "Authorize" button is visible
```

---

## Integration Checklist

- [ ] Added imports to top of file
- [ ] Added environment setup section
- [ ] Added `_load_api_key()` function
- [ ] Added `EXPECTED_API_KEY = _load_api_key()`
- [ ] Updated FastAPI app creation with description
- [ ] Added `api_key_header` object
- [ ] Added `require_api_key()` function
- [ ] Added `custom_openapi()` function
- [ ] Set `app.openapi = custom_openapi`
- [ ] Added `_auth: None = Depends(require_api_key)` to protected endpoints
- [ ] Left `/health` endpoint unprotected
- [ ] Added `python-dotenv==1.1.1` to requirements.txt
- [ ] Created `.env.example` file
- [ ] Added `.env` to `.gitignore`
- [ ] Tested all endpoints with curl
- [ ] Verified Swagger UI shows "Authorize" button

