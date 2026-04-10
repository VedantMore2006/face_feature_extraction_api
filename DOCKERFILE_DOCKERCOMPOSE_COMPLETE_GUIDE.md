# Easy Guide: Add Dockerfile and docker-compose.yml to Any Ready API Repo

This guide is for teams that already have:

- API code ready
- requirements.txt ready

But do not have these 2 files yet:

1. Dockerfile
2. docker-compose.yml

This is a practical copy-paste guide with very small edits.

---

## 1. What You Need Before Starting

Minimum required:

1. Your API entry file (example: scoring_api.py)
2. requirements.txt
3. The command you use to run your API (example: uvicorn scoring_api:app --port 8011)
4. Port number your API should expose (here: 8011)

Optional but recommended:

- .env.example
- .dockerignore
- /health endpoint for health check

---

## 2. 10-Minute Workflow

1. Create Dockerfile in repo root.
2. Create docker-compose.yml in repo root.
3. Replace only 3 values:
   - APP_MODULE (example: scoring_api:app)
   - PORT (example: 8011)
   - ENV vars (if needed)
4. Run:

```bash
docker compose build
docker compose up -d
```

5. Test:

```bash
curl http://localhost:8011/health
```

---

## 3. Dockerfile (Copy As Is)

You can copy this as-is for most FastAPI projects.

Change only:

- APP MODULE in the last line
- PORT if not 8011

```dockerfile
FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy full project (simple and safe for most repos)
COPY . .

ENV PYTHONUNBUFFERED=1

EXPOSE 8011

HEALTHCHECK --interval=30s --timeout=10s --start-period=20s --retries=3 \
  CMD curl -f http://localhost:8011/health || exit 1

# Change APP_MODULE if needed, example: main:app or api:app
CMD ["uvicorn", "scoring_api:app", "--host", "0.0.0.0", "--port", "8011"]
```

---

## 4. docker-compose.yml (Copy As Is)

You can copy this as-is for most single API services.

Change only:

- Service name (optional)
- APP port if not 8011
- Environment variable names/values as needed

```yaml
version: '3.8'

services:
  api:
    build:
      context: .
      dockerfile: Dockerfile
    image: my-api:latest
    container_name: my-api
    ports:
      - "8011:8011"
    environment:
      - PYTHONUNBUFFERED=1
      - SCORING_API_KEY=${SCORING_API_KEY}
      - MODEL_DIR=${MODEL_DIR:-reports/model_training/run_20260324_171117}
      - LABEL_COL=${LABEL_COL:-condition_label}
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8011/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 20s
    logging:
      driver: "json-file"
      options:
        max-size: "10m"
        max-file: "3"
```

---

## 5. What Parts Are Mandatory vs Optional

### Mandatory in Dockerfile

- FROM
- WORKDIR
- COPY requirements.txt + pip install
- COPY project files
- CMD to start API

### Optional but Recommended in Dockerfile

- HEALTHCHECK
- EXPOSE
- apt cleanup for smaller image

### Mandatory in docker-compose.yml

- services
- build or image
- ports

### Optional but Recommended in docker-compose.yml

- environment
- restart policy
- healthcheck
- logging limits

---

## 6. Exactly What To Edit in the Templates

If your project is different, edit only these lines.

### In Dockerfile

1. Uvicorn module:

```dockerfile
CMD ["uvicorn", "YOUR_FILE_NAME:app", "--host", "0.0.0.0", "--port", "YOUR_PORT"]
```

Example:

```dockerfile
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

2. Port:

```dockerfile
EXPOSE YOUR_PORT
```

### In docker-compose.yml

1. Port mapping:

```yaml
ports:
  - "YOUR_PORT:YOUR_PORT"
```

2. Optional environment variables:

```yaml
environment:
  - YOUR_ENV_VAR=${YOUR_ENV_VAR}
```

---

## 7. Local Test Commands (Quick)

```bash
docker compose build
docker compose up -d
docker compose ps
docker compose logs -f
```

Stop:

```bash
docker compose down
```

---

## 8. Hostinger Docker Manager (Compose URL)

1. Push repo to GitHub.
2. Use raw compose URL:

```text
https://raw.githubusercontent.com/<org>/<repo>/main/docker-compose.yml
```

3. In Hostinger panel, set env vars (for example SCORING_API_KEY).
4. Deploy.
5. Open API docs:

```text
http://<vps-ip>:8011/docs
```

---

## 9. Common Issues and Fast Fixes

Issue: Container starts but API not reachable

- Check ports in compose and EXPOSE/CMD port in Dockerfile

Issue: API key error at startup

- Set real SCORING_API_KEY (not CHANGE_ME)

Issue: Build fails due to missing files

- Ensure requirements.txt is in root
- Ensure COPY path matches your repo layout

Issue: Healthcheck fails

- Verify /health endpoint exists
- Or remove HEALTHCHECK temporarily

---

## 10. Minimal Final Checklist

Before sharing with team:

- Dockerfile exists
- docker-compose.yml exists
- Port mapping is correct
- Startup command is correct
- docker compose up -d works
- API reachable on expected port

---

This guide is intentionally simple: if your API is already ready and requirements.txt exists, these two files are usually enough to containerize quickly.
