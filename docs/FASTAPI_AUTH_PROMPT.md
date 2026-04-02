# Prompt: Add API Key Authentication to FastAPI Service

Use this prompt when instructing an AI assistant to add secure API key authentication to a FastAPI application.

---

## Quick Prompt (Copy & Use)

```
I have a FastAPI application and need to add secure API key authentication.

Here's what I need implemented:

1. **Environment Variables & .env Support**
   - Add `python-dotenv==1.1.1` to requirements.txt
   - Create `.env.example` with placeholder: `API_KEY_NAME=CHANGE_ME`
   - Load .env in the app with: `load_dotenv(project_root / ".env")`
   - Ensure `.env` is in `.gitignore`

2. **API Key Validation Function**
   - Create `_load_api_key()` that reads from environment variable on startup
   - Validate the key is not empty and not the placeholder value "CHANGE_ME"
   - Raise RuntimeError with clear message if invalid
   - Store as `EXPECTED_API_KEY = _load_api_key()`

3. **FastAPI Security Dependency**
   - Import: `from fastapi.security import APIKeyHeader` and `from fastapi import Depends`
   - Create `api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)`
   - Create `require_api_key()` function that validates provided key using `secrets.compare_digest()`
   - Return `HTTPException(401)` if key is missing or invalid

4. **Swagger UI Integration (OpenAPI)**
   - Import `from fastapi.openapi.utils import get_openapi`
   - Create `custom_openapi()` function that:
     - Adds security scheme to OpenAPI spec with type="apiKey", in="header", name="X-API-Key"
     - Applies security requirement to all endpoints EXCEPT `/health`
   - Set `app.openapi = custom_openapi`

5. **Protect Endpoints**
   - Add `_auth: None = Depends(require_api_key)` parameter to each protected endpoint
   - Leave `/health` endpoint unprotected for health checks

6. **Update FastAPI App Init**
   - Add `description` parameter to `FastAPI()` constructor
   - Add `description` parameter to `APIKeyHeader()` constructor

Result: API will require `X-API-Key` header, Swagger UI will show "Authorize" button, and key will load from `.env` or environment.
```

---

## Detailed Checklist (For Reference)

When implementing, verify:

- [ ] `python-dotenv` added to requirements.txt
- [ ] `.env.example` file created with placeholder
- [ ] `.env` added to `.gitignore`
- [ ] Imports: `os`, `secrets`, `dotenv.load_dotenv`, `fastapi.security.APIKeyHeader`, `fastapi.openapi.utils.get_openapi`
- [ ] `load_dotenv(project_root / ".env")` called early in app initialization
- [ ] `_load_api_key()` function created with error handling
- [ ] `APIKeyHeader` object created with description
- [ ] `require_api_key()` dependency function uses `secrets.compare_digest()`
- [ ] `custom_openapi()` function defines security scheme
- [ ] Security scheme applied to all endpoints except `/health`
- [ ] Protected endpoints have `_auth: None = Depends(require_api_key)` parameter
- [ ] Application starts without errors
- [ ] Swagger UI at `/docs` shows "Authorize" button
- [ ] Requesting protected endpoint without header returns 401
- [ ] Requesting with wrong key returns 401
- [ ] Requesting with correct key succeeds

---

## Files to Create/Modify

| File | Action | What to Add |
|------|--------|-----------|
| `requirements.txt` | Add to end | `python-dotenv==1.1.1` |
| `.env.example` | Create | `API_KEY_NAME=CHANGE_ME` (adjust name) |
| `.gitignore` | Update | `.env` (if not already present) |
| `your_api.py` | Modify | Imports, load_dotenv(), _load_api_key(), require_api_key(), custom_openapi(), endpoint decorators |

---

## Quick Start Commands (For Users)

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Setup local .env
cp .env.example .env
python3 -c "import secrets; print(secrets.token_urlsafe(32))" # Generate random key
# Edit .env and paste the key value

# 3. Start API
uvicorn your_api:app --reload

# 4. Test with curl
curl -X GET "http://127.0.0.1:8000/docs" \
  -H "X-API-Key: <your_key_here>"

# 5. Use Swagger UI
# Open http://127.0.0.1:8000/docs
# Click "Authorize"
# Enter API key
```

---

## Security Best Practices

- Use `secrets.compare_digest()` (timing-attack resistant)
- Never print API key to logs
- Validate key on app startup (fail fast)
- Keep `/health` unprotected for load balancers
- For production: inject from cloud secret manager, not `.env`
- Rotate keys periodically
- Use separate keys per environment (dev/staging/prod)

