# Privacy-Safe Behavioral Analysis System

A **privacy-first** real-time facial behavior analysis system that extracts high-level behavioral features without storing identifiable biometric data. The system uses baseline normalization, temporal smoothing, and event-based response tracking to provide behavioral insights suitable for mental health, user experience research, and human-computer interaction studies.

---

## 🎯 Core Specialties

### **1. Privacy-by-Design Architecture**
- ✅ **NO raw landmark coordinates stored** — only behavioral abstractions
- ✅ **NO face geometry recorded** — prevents identity reconstruction
- ✅ **NO video/image persistence** — all processing is real-time
- ✅ **Baseline-relative scaling** — each person is their own reference point
- ✅ **Clipped & rounded features** — all values normalized to [0, 1] range

**Result:** Behavioral fingerprints that reveal engagement patterns WITHOUT compromising privacy.

---

### **2. Personal Session Baseline (PSB) Normalization**
Instead of comparing individuals to population norms, the system:
- Collects **30-second baseline** at session start (neutral state)
- Computes **per-person statistics** (mean, standard deviation)
- Scales **all subsequent features** relative to this baseline
- Ensures **cross-session consistency** and individual sensitivity

**Benefit:** A naturally expressive person won't be flagged as "abnormal" — deviations are relative to their own baseline.

---

### **3. Six Behavioral Feature Extraction**

| Feature | What It Measures | Indicator Of |
|---------|------------------|--------------|
| **AU12 (Smile)** | Lip corner elevation | Positive affect, engagement, genuine emotion |
| **Expressivity** | Total facial movement | Animation, emotional engagement vs. flat affect |
| **Head Velocity** | Horizontal head rotation speed | Restlessness, scanning behavior, attention shifts |
| **Eye Contact** | Frontal gaze maintenance | Social engagement, attention, avoidance |
| **Blink Rate** | Blinks per minute (BPM) | Cognitive load, stress, anxiety, concentration |
| **Response Latency** | Time from stimulus to mouth opening | Processing speed, hesitation, cognitive effort |

---

### **4. Event-Based Response Latency Detection**
Traditional systems log every frame. This system:
- **Manual stimulus trigger** — press "s" when question ends
- **Automatic mouth opening detection** — finds response start
- **Latency calculation** — time between stimulus and response
- **Baseline-scaled latency** — normalized for individual speech patterns

**Use Case:** Interview analysis, cognitive testing, conversational AI evaluation.

---

### **5. Real-Time Processing Pipeline**
- **MediaPipe Face Mesh** — 468 landmark detection at 15 FPS
- **Temporal smoothing** — 5-frame moving average reduces noise
- **Per-frame feature computation** — raw → smoothed → scaled
- **Bounds validation** — assertion ensures all features in [0, 1]
- **Dual CSV logging** — production features + validation raw values

---

### **6. Flexible Frame Source Architecture**
- **Abstraction layer** — Pipeline is source-agnostic via `FrameSource` interface
- **Live webcam support** — `CameraSource` for real-time analysis
- **Video file support** — `VideoFileSource` for offline processing
- **Deterministic timestamps** — Frame-based timing for reproducible video analysis
- **Easy extensibility** — Add RTSP streams, image sequences, or network cameras

**Use Cases:**
- Live monitoring during interviews or therapy sessions
- Batch processing of archived video recordings
- Validation and testing with controlled video inputs
- Research with reproducible timestamps

---

## 📁 Project Structure

- Facial_analysis/
- [extra/main.py](extra/main.py) — Archived CLI entrypoint with argparse (webcam/video selection)
- [config.py](config.py) — Tunable parameters (FPS, baseline duration, etc.)
- [app.py](app.py) — FastAPI wrapper entrypoint for the API service
- [Project_extract.py](Project_extract.py) — Project extraction utility
- [data/](data/) — Privacy-safe feature logs
- [src/](src/) — Core runtime modules
- [src/pipeline.py](src/pipeline.py) — Main processing orchestrator
- [src/frame_source.py](src/frame_source.py) — **NEW:** Frame source abstraction (CameraSource/VideoFileSource)
- [src/camera.py](src/camera.py) — Legacy webcam capture (now wrapped by frame_source)
- [src/face_mesh.py](src/face_mesh.py) — MediaPipe landmark detection
- [src/landmark_processor.py](src/landmark_processor.py) — Extract subset of key landmarks
- [src/baseline.py](src/baseline.py) — Personal baseline collection & normalization
- [src/scaler.py](src/scaler.py) — Z-score scaling with sigma floor
- [src/smoothing.py](src/smoothing.py) — Moving average temporal filter
- [src/feature_vector.py](src/feature_vector.py) — Clip, round, and construct final vector
- [src/feature_logger.py](src/feature_logger.py) — Dual CSV writer (production + validation)
- [src/logger.py](src/logger.py) — Optional raw landmark logger (debug)
- [src/feature_engine/](src/feature_engine/) — Feature computation engines
- [src/feature_engine/au12.py](src/feature_engine/au12.py) — Smile intensity (lip corner distance)
- [src/feature_engine/expressivity.py](src/feature_engine/expressivity.py) — Total facial movement magnitude
- [src/feature_engine/head_velocity.py](src/feature_engine/head_velocity.py) — Yaw angle velocity computation
- [src/feature_engine/head_pose.py](src/feature_engine/head_pose.py) — Head orientation estimation
- [src/feature_engine/eye_contact.py](src/feature_engine/eye_contact.py) — Gaze engagement ratio
- [src/feature_engine/blink.py](src/feature_engine/blink.py) — Blink detection & rate calculation
- [src/feature_engine/response_latency.py](src/feature_engine/response_latency.py) — Event-based latency tracker
- [docs/](docs/) — Documentation (includes video processing guide)

---

## 🔬 Technical Architecture

### **Data Flow:**

```
Frame Source (Webcam OR Video File)
    ↓
Frame Acquisition (CameraSource / VideoFileSource)
    ↓
MediaPipe Face Mesh (468 landmarks)
    ↓
Landmark Subset Extraction (17 key points)
    ↓
Raw Feature Computation (AU12, Expressivity, Head Yaw, EAR, etc.)
    ↓
Temporal Smoothing (5-frame moving average)
    ↓
Baseline Collection (first 30 seconds)
    ↓
--- BASELINE LOCK ---
    ↓
Baseline-Relative Scaling (z-score normalization)
    ↓
Feature Vector Construction [6 values, clipped to [0,1]]
    ↓
Bounds Assertion (ensures valid range)
    ↓
Dual CSV Logging:
  • Privacy-Safe Features (data/features_*.csv)
  • Validation Raw Values (data/validation_raw_session_*.csv)
    ↓
API Entry Point (app.py)
```

---

## 🚀 Getting Started

### **Prerequisites:**
```bash
pip install opencv-python mediapipe numpy streamlit plotly pandas
```

### **1. Run Real-Time Analysis:**

#### **Live Webcam Mode (Default):**
```bash
python extra/main.py
```

#### **Offline Video File Mode:**
```bash
python extra/main.py --video path/to/video.mp4
```

**Examples:**
```bash
# Process a recorded interview
python extra/main.py --video recordings/interview_session.mp4

# Batch process multiple videos
for video in recordings/*.mp4; do
    python extra/main.py --video "$video"
done
```

**Controls:**
- Press **"q"** to quit session
- Press **"s"** to trigger stimulus (for response latency)

**Output:**
- Production features: `data/features_{timestamp}.csv`
- Validation raw values: `data/validation_raw_session_{timestamp}.csv`
- Real-time video feed with landmarks & phase indicator

---

### **2. Run the API Service:**
```bash
uvicorn app:app --host 0.0.0.0 --port 8010 --reload
```

**API endpoints:**
- `GET /health` — service health check
- `POST /extract/video` — upload a video and generate a session vector
- `GET /extract/session/{session_id}/vector` — fetch a saved session vector

---

## API Reference (Current Codebase)

The current workspace exposes one FastAPI service through `app.py`, which imports the API instance from `extraction_api.py`.

- Base URL (local): `http://127.0.0.1:8010`
- Swagger UI: `http://127.0.0.1:8010/docs`
- OpenAPI JSON: `http://127.0.0.1:8010/openapi.json`

### Authentication

All extraction endpoints require API key authentication via header `X-API-Key`.

1. Create `.env` in project root:

```bash
cp example.env .env
```

2. Set a real key:

```bash
EXTRACTION_API_KEY=replace_with_a_long_random_secret
```

3. Start the API:

```bash
uvicorn app:app --host 0.0.0.0 --port 8010 --reload
```

### Endpoint Summary

| Method | Route | Auth | What It Does |
|---|---|---|---|
| `GET` | `/health` | No | Health check for service status |
| `POST` | `/extract/video` | Yes | Uploads a video and generates one session-level ML vector |
| `GET` | `/extract/session/{session_id}/vector` | Yes | Returns saved vector for a given session |

### 1) Health Check

- Method: `GET`
- Route: `/health`
- Input: none
- Output:

```json
{
  "status": "ok",
  "service": "Facial Extraction API"
}
```

curl:

```bash
curl -X GET "http://127.0.0.1:8010/health"
```

### How to Use

```python
import requests

url = "http://127.0.0.1:8010/extract/video"

payload = {
  "mode": "balanced",
  "frame_stride": 0,
  "min_duration_seconds": 150.0,
  "allow_short": True,
  "model_dir": "reports/model_training/run_20260324_171117",
  "training_report": "",
  "label_col": "condition_label",
}

headers = {
  "x-api-key": "your_api_key"
}

with open("extra/test.mp4", "rb") as video_file:
  files = {
    "video": video_file
  }
  response = requests.post(url, params=payload, headers=headers, files=files)

print(response.json())
```

### 2) Extract Vector From Video

- Method: `POST`
- Route: `/extract/video`
- Content type: `multipart/form-data`
- Required form field:
  - `video`: video file upload

Query parameters:

- `mode` (string): `accurate` | `balanced` | `fast`, default `balanced`
- `frame_stride` (int): `>=0`, default `0` (uses mode default)
- `min_duration_seconds` (float): `>0`, default `150.0`
- `allow_short` (bool): default `false`
- `model_dir` (string): default `reports/model_training/run_20260324_171117`
- `training_report` (string): optional path, default empty (auto-resolve)
- `label_col` (string): default `condition_label`

Headers:

- `X-API-Key: <your_api_key>`

Success response:

```json
{
  "session_id": "8d0c5f7e62d14e2cbe64b70842d4f4da",
  "vector_feature_count": 63
}
```

Server-side artifacts created for this request:

- Full session payload: `reports/api_sessions/{session_id}.json`
- Vector-only payload: `reports/api_vectors/{session_id}.json`
- Raw extraction CSV (timestamp + raw base features): `reports/api_raw_features/api_raw_features_{session_id}.csv`

Video persistence policy:

- Uploaded videos are processed through a temporary file and deleted after extraction.
- Original uploaded video content is **not** stored permanently.

curl:

```bash
curl -X POST "http://127.0.0.1:8010/extract/video?mode=balanced&allow_short=true" \
  -H "X-API-Key: replace_with_a_long_random_secret" \
  -F "video=@extra/test.mp4"
```

### 3) Get Saved Vector By Session ID

- Method: `GET`
- Route: `/extract/session/{session_id}/vector`
- Path parameter:
  - `session_id` (string)
- Headers:
  - `X-API-Key: <your_api_key>`

Success response shape:

```json
{
  "session_id": "8d0c5f7e62d14e2cbe64b70842d4f4da",
  "vector": {
    "au12_mean_amplitude__mean": 0.000123,
    "au12_mean_amplitude__std": 0.000045
  }
}
```

curl:

```bash
curl -X GET "http://127.0.0.1:8010/extract/session/8d0c5f7e62d14e2cbe64b70842d4f4da/vector" \
  -H "X-API-Key: replace_with_a_long_random_secret"
```

### End-to-End curl Workflow

1. Upload video and capture session ID:

```bash
SESSION_ID=$(curl -s -X POST "http://127.0.0.1:8010/extract/video?mode=balanced&allow_short=true" \
  -H "X-API-Key: replace_with_a_long_random_secret" \
  -F "video=@extra/test.mp4" \
  | python3 -c "import sys,json; print(json.load(sys.stdin)['session_id'])")

echo "$SESSION_ID"
```

2. Fetch vector:

```bash
curl -s "http://127.0.0.1:8010/extract/session/$SESSION_ID/vector" \
  -H "X-API-Key: replace_with_a_long_random_secret"
```

### Common API Errors

- `401 Unauthorized`: missing or invalid `X-API-Key`
- `400 Bad Request`: invalid query/input, or video too short when `allow_short=false`
- `404 Not Found`: missing session, training report, or model artifact path
- `500 Internal Server Error`: extraction runtime failure

---

## 🧬 Feature Modules Explained

### **AU12 (Action Unit 12 — Smile Intensity)**
- **File:** `src/feature_engine/au12.py`
- **Method:** Euclidean distance between left/right lip corners
- **Scaling:** Baseline-relative (personal baseline smile width)
- **Interpretation:** 
  - **High (>0.6):** Active smiling, positive affect
  - **Low (<0.4):** Neutral or suppressed expression

---

### **Expressivity (Facial Animation)**
- **File:** `src/feature_engine/expressivity.py`
- **Method:** Sum of per-landmark movement velocities
- **Scaling:** Baseline-relative (personal baseline animation level)
- **Interpretation:**
  - **High (>0.6):** Animated, emotionally engaged
  - **Low (<0.4):** Flat affect, emotional suppression

---

### **Head Velocity (Scanning Behavior)**
- **File:** `src/feature_engine/head_velocity.py`
- **Method:** Yaw angle change per second (°/s)
- **Scaling:** Baseline-relative (personal baseline head movement)
- **Interpretation:**
  - **High (>0.6):** Active scanning, restlessness
  - **Low (<0.4):** Head fixation, sustained attention

---

### **Eye Contact (Gaze Engagement)**
- **File:** `src/feature_engine/eye_contact.py`
- **Method:** Proportion of time with yaw angle < threshold (frontal gaze)
- **Scaling:** Baseline-relative (personal baseline gaze patterns)
- **Interpretation:**
  - **High (>0.6):** Strong social engagement
  - **Low (<0.4):** Gaze aversion, distraction

---

### **Blink Rate (Cognitive Load)**
- **File:** `src/feature_engine/blink.py`
- **Method:** Eye Aspect Ratio (EAR) drops below threshold → blink count → BPM
- **Scaling:** Baseline-relative (personal baseline blink frequency)
- **Interpretation:**
  - **High (>0.6):** Elevated stress, anxiety, fatigue
  - **Low (<0.4):** Concentration or discomfort (staring)

---

### **Response Latency (Processing Speed)**
- **File:** `src/feature_engine/response_latency.py`
- **Method:** 
  1. Press "s" key when stimulus ends
  2. Detect mouth opening (above baseline threshold)
  3. Calculate time difference
- **Scaling:** Baseline-relative (personal baseline speech onset speed)
- **Interpretation:**
  - **High (>0.6):** Quick response, low hesitation
  - **Low (<0.4):** Delayed response, processing difficulty

---

## 🔐 Privacy Guarantees

### **What IS Stored:**
✅ Scaled behavioral features (6 numbers per frame)  
✅ Timestamp (for temporal analysis)  
✅ Session metadata (FPS, baseline duration)

### **What IS NOT Stored:**
❌ Raw landmark coordinates (x, y, z)  
❌ Video frames or images  
❌ Face geometry or mesh structure  
❌ Identifiable biometric data  
❌ Reconstructable facial information

**Compliance:** Suitable for GDPR, HIPAA, and privacy-sensitive applications.

---

## 📊 Output Format

### **Production Features: features_{timestamp}.csv**

| Column | Description | Range |
|--------|-------------|-------|
| `S_AU12` | Scaled smile intensity | 0.0 - 1.0 |
| `S_AUVar` | Scaled expressivity (facial animation) | 0.0 - 1.0 |
| `S_HeadVelocity` | Scaled head rotation speed | 0.0 - 1.0 |
| `S_EyeContact` | Scaled gaze engagement ratio | 0.0 - 1.0 |
| `S_BlinkRate` | Scaled blink frequency | 0.0 - 1.0 |
| `S_ResponseLatency` | Scaled response timing | 0.0 - 1.0 |

**Interpretation:**
- **0.5** = Baseline (neutral state)
- **>0.5** = Elevated relative to baseline
- **<0.5** = Suppressed relative to baseline

---

### **Validation Raw Values: validation_raw_session_{timestamp}.csv**

| Column | Description | Notes |
|--------|-------------|-------|
| `frame_index` | Frame number in session | Sequential counter |
| `timestamp_ms` | Milliseconds since session start | Deterministic for video files |
| `au12_raw` | Raw smile intensity | **NO smoothing, NO scaling, NO baseline** |
| `expressivity_raw` | Raw facial movement | **NO smoothing, NO scaling, NO baseline** |
| `head_velocity_raw` | Raw head rotation speed | **NO smoothing, NO scaling, NO baseline** |
| `blink_rate_raw` | Raw blink frequency | **NO smoothing, NO scaling, NO baseline** |
| `ear_raw` | Raw Eye Aspect Ratio | **NO smoothing, NO scaling, NO baseline** |
| `yaw_raw` | Raw head yaw angle | **NO smoothing, NO scaling, NO baseline** |

**Purpose:** Validation artifact for verifying feature computation correctness

---

## ⚙️ Configuration

Edit `config.py` to customize:

```python
class CameraConfig:
    DEVICE_ID = 0      # Webcam selection
    WIDTH = 640        # Frame width
    HEIGHT = 480       # Frame height
    FPS = 15           # Target frame rate

class BaselineConfig:
    ENABLE_BASELINE = True
    DURATION_SECONDS = 30    # Baseline collection time
    SIGMA_FLOOR = 1e-6       # Minimum std dev

class DebugConfig:
    SHOW_LANDMARKS = True    # Draw landmarks on video
    SHOW_FPS = True          # Display FPS counter
```

---

## 🧪 Use Cases

### **1. Mental Health Monitoring**
- Track affect patterns over therapy sessions
- Detect flat affect, emotional suppression
- Monitor engagement during video consultations

### **2. User Experience Research**
- Measure engagement during product demos
- Detect confusion or frustration (elevated blink, low smile)
- Assess response confidence (latency tracking)

### **3. Interview Analysis**
- Quantify hesitation patterns (response latency)
- Detect stress indicators (blink rate, expressivity)
- Compare candidate behavioral profiles

### **4. Educational Technology**
- Monitor student attention (eye contact, head velocity)
- Detect confusion or cognitive overload (blink rate)
- Assess engagement during lessons (expressivity, smile)

### **5. Conversational AI Evaluation**
- Measure user frustration during chatbot interactions
- Detect when explanation is insufficient (scanning behavior)
- Optimize response timing based on latency data

---

## 🛡️ Ethical Considerations

### **Informed Consent:**
Always obtain explicit consent before recording or analyzing facial behavior.

### **Purpose Limitation:**
Use data only for stated purposes (e.g., research, UX testing).

### **Data Minimization:**
This system already implements privacy-by-design — no excess data is collected.

### **Transparency:**
Inform users that behavioral patterns are being analyzed, and explain what features are extracted.

### **Bias Mitigation:**
Baseline normalization reduces cross-individual bias, but always validate on diverse populations.

---

## 🔧 Extending the System

### **Add New Features:**
1. Create feature engine in `src/feature_engine/your_feature.py`
2. Implement computation logic (raw value from landmarks)
3. Add to pipeline in `src/pipeline.py`
4. Update feature vector in `src/feature_vector.py`
5. Update the README and response schema examples if the API output changes

### **Add New Frame Sources:**
1. Create new class inheriting from `FrameSource` in `src/frame_source.py`
2. Implement required methods: `read()`, `release()`, `get_fps()`, `is_realtime()`
3. Add argparse option in `extra/main.py`
4. Example: RTSP streams, image sequences, network cameras

### **Custom Baselines:**
- Modify `src/baseline.py` to persist baselines across sessions
- Store baseline stats in database for longitudinal studies

### **Real-Time Alerts:**
- Add thresholds in pipeline for extreme deviations
- Trigger alerts when multiple features exceed limits

---

## 📚 References & Documentation

### **Technical References:**
- **MediaPipe Face Mesh:** [Google MediaPipe](https://mediapipe.dev/)
- **Action Units (FACS):** Ekman & Friesen facial coding system
- **Eye Aspect Ratio:** Soukupová & Čech (2016)
- **Privacy-Preserving CV:** Behavioral abstraction techniques

### **Project Documentation:**
- **[Video Processing Guide](docs/VIDEO_PROCESSING_GUIDE.md)** — How to use video files instead of webcam
- **[Frame Source Refactoring](docs/FRAME_SOURCE_REFACTORING.md)** — Technical details of abstraction layer

---

## 📝 License

This project is for research and educational purposes. Ensure compliance with local privacy regulations when deploying.

---

## 🤝 Contributing

Contributions are welcome! Focus areas:
- Additional behavioral features (e.g., gaze tracking, micro-expressions)
- Cross-session baseline persistence
- Real-time alert system
- Multi-face tracking

---

## 📧 Contact
- Email: [vedantmoremain@gamil.com](mailto:vedantmoremain@gmail.com)
- For questions, issues, or collaboration inquiries, please open a GitHub issue or contact the project maintainer.

---

**Built with privacy, powered by behavior analysis.** 🔐📊

