# 🧠 Facial Feature Extraction API — End-to-End Workflow

> A plain-language, function-level guide to how data flows through the entire system —
> from the moment a video is uploaded to the moment a feature vector is saved.

---

## 📁 Project at a Glance

```
face_feature_extraction_api/
│
├── app.py                  ← Entry point. Starts the server.
├── extraction_api.py       ← All API routes live here (upload, fetch).
├── predict_video_risk.py   ← Core processing engine (video → features).
│
└── src/
    ├── config.py           ← All tunable settings (FPS, thresholds, etc.)
    ├── contracts.py        ← Loads & validates the feature definition file.
    ├── regions.py          ← Maps face into named zones (brow, lips, jaw…).
    ├── landmark_stream.py  ← Opens video/webcam, runs MediaPipe, yields frames.
    ├── feature_engine.py   ← Computes all 34 raw behavioral features per frame.
    ├── normalization.py    ← Collects baseline, then scales values to [0–1].
    ├── pipeline.py         ← Live/webcam mode orchestrator (with on-screen UI).
    └── csv_logger.py       ← Writes feature rows to a CSV file.
```

---

## 🚀 Step 0 — Starting the Server

**File:** `app.py`

```python
from extraction_api import app
```

That single line is the entire file.
When you run `uvicorn app:app --port 5100`, Python imports `extraction_api.py`,
which builds the FastAPI app and registers all routes.
The server is now live and waiting for requests.

**What happens at import time inside `extraction_api.py`:**

| Action | What it does |
|---|---|
| `load_dotenv(...)` | Reads the `.env` file and loads the API key into the environment. |
| `_load_api_key()` | Reads `EXTRACTION_API_KEY` from the environment. Crashes loudly if it's missing or still set to `CHANGE_ME`. |
| `sessions_dir.mkdir(...)` | Creates `reports/api_sessions/` if it doesn't exist. |
| `vectors_dir.mkdir(...)` | Creates `reports/api_vectors/`. |
| `raw_features_dir.mkdir(...)` | Creates `reports/api_raw_features/`. |
| `custom_openapi()` | Registers the API key security scheme in the Swagger UI (`/docs`). |

---

## 🌐 The Three API Endpoints

| Method | Route | Auth needed? | What it does |
|---|---|---|---|
| `GET` | `/health` | No | Quick check that the server is running. |
| `POST` | `/extract/video` | Yes (`X-API-Key` header) | Upload a video → get a session ID back. |
| `GET` | `/extract/session/{session_id}/vector` | Yes | Retrieve the saved feature vector for a session. |

---

## 🔐 How Authentication Works

**Function:** `require_api_key()` — `extraction_api.py`

Every protected endpoint has `_auth: None = Depends(require_api_key)` in its signature.
FastAPI runs this function before the endpoint logic.

```
Incoming request
      │
      ▼
require_api_key()
      │
      ├─ No key provided?   → 401 Unauthorized
      ├─ Wrong key?         → 401 Unauthorized
      └─ Key matches?       → ✅ Continue to endpoint
```

---

## 📤 Main Flow — `POST /extract/video`

**Function:** `extract_from_video()` — `extraction_api.py`

This is the big one. Here is every step in order.

---

### Stage 1 — Validate Inputs

```
extract_from_video() called
          │
          ├─→ RuntimeConfig()                     [src/config.py]
          │     Creates a settings object with defaults:
          │     FPS=15, baseline=20s, smoothing window=5 frames, etc.
          │
          ├─→ validate_runtime_config(cfg)         [src/config.py]
          │     Checks every setting is a sane number.
          │     Raises ValueError if anything is zero or negative.
          │
          ├─→ validate_runtime_contract()          [src/regions.py]
          │     Confirms that the 248 landmark IDs are all valid,
          │     unique, and within MediaPipe's 0–477 range.
          │     Runs at import time too — so it fails fast.
          │
          └─→ load_feature_contract(project_root) [src/contracts.py]
                Reads docs/feature_definitions_v1.json.
                Returns a FeatureContract object that lists the
                34 feature names in the exact order the model expects.
```

---

### Stage 2 — Save the Video Temporarily

```
extract_from_video()
          │
          ├─ Generate session_id  (random UUID hex, e.g. "8d0c5f7e...")
          │
          ├─ Read the uploaded file bytes from memory
          │
          └─ Write to a temp file on disk  (e.g. /tmp/api_8d0c5f7e_upload.mp4)
               This is so OpenCV can open it by file path.
               The temp file is deleted at the end (in the finally block).
```

---

### Stage 3 — Resolve Feature Schema

**Function:** `_expected_schema()` — `extraction_api.py`
which calls → `resolve_expected_feature_order()` — `predict_video_risk.py`

```
_expected_schema(model_dir, training_report, label_col)
          │
          └─→ resolve_expected_feature_order(project_root, training_report_path, label_col)
                    │
                    ├─ Opens training_report.json (model metadata file).
                    │
                    ├─ Finds the path to the original training CSV.
                    │
                    ├─ Reads the CSV header row (just the column names, no data).
                    │
                    └─ Returns the list of 63 engineered feature names
                       in the exact order the ML model was trained on.

          Returns: expected_features (list of 63 names)
```

> **Why 63?** Each of the 34 raw features gets 6 statistical summaries
> (mean, std, min, max, range, slope) = 204 candidates.
> After feature pruning during training, 63 were kept.

---

### Stage 4 — Extract Raw Features from Video

**Function:** `extract_raw_timeseries()` — `predict_video_risk.py`

This is the core processing loop. It turns the video into a table of numbers.

```
extract_raw_timeseries(video_file, cfg, feature_order, frame_stride)
          │
          ├─→ _build_engine_cfg(cfg)
          │     Converts the RuntimeConfig settings into a FeatureEngineConfig
          │     (just moves values from one dataclass to another).
          │
          ├─→ FeatureEngine(feature_order, cfg)   [src/feature_engine.py]
          │     Creates the stateful feature calculator.
          │     Initialises smoothing buffers (one deque per feature).
          │     Initialises event queues (blink timestamps, smile timestamps, etc.)
          │
          └─→ LandmarkStream(cfg, video_file_path) [src/landmark_stream.py]
                Opens the video file.
                Starts MediaPipe FaceMesh (478-point model).
```

**Inside the frame loop:**

```
for each frame in the video:
          │
          ├─ frame_stride skip check
          │     e.g. stride=2 means process every 2nd frame (balanced mode).
          │     Stride=1 = accurate, Stride=3 = fast.
          │
          ├─→ engine.compute(frame_data)           [src/feature_engine.py]
          │     (see Feature Engine section below)
          │     Returns: dict of 34 raw feature values for this frame.
          │
          └─ Append {timestamp, feature1, feature2, ...} to rows list.

After loop:
          └─ Returns: DataFrame (rows × 35 columns) + extraction stats
```

---

### Stage 4a — Inside `LandmarkStream` (frame-by-frame)

**Class:** `LandmarkStream` — `src/landmark_stream.py`

```
LandmarkStream.__enter__()
          │
          ├─ cv2.VideoCapture(video_file_path)   Opens the video file.
          │
          └─ mp.solutions.face_mesh.FaceMesh(...)
                Starts MediaPipe with:
                  refine_landmarks=True  (enables iris tracking, adds pts 468–477)
                  max_num_faces=1
                  min_detection_confidence=0.5

LandmarkStream.frames()  ← generator, yields one frame at a time
          │
          ├─ capture.read()         Reads next BGR frame from video.
          │
          ├─ cv2.cvtColor(BGR→RGB)  MediaPipe needs RGB.
          │
          ├─ face_mesh.process(rgb) Runs face detection + landmark tracking.
          │                         Returns 478 (x,y,z) points if face found.
          │
          ├─ _extract_landmark_tuples(face_landmarks)
          │     Converts MediaPipe's landmark objects into a plain Python list
          │     of (x, y, z) tuples. All coordinates are 0.0–1.0 (normalised).
          │
          ├─ Timestamp from video position (not wall clock)
          │     Uses CAP_PROP_POS_MSEC so time is tied to video, not CPU speed.
          │
          └─ yield FrameData(timestamp, landmarks), raw_frame
```

**`_extract_landmark_tuples(face_landmarks)`**
Converts MediaPipe's internal format into a simple list like:
`[(0.52, 0.34, -0.01), (0.53, 0.35, -0.01), ...]`
478 tuples total. Each is (x, y, z) in 0–1 space.

---

### Stage 4b — Inside `FeatureEngine.compute()` (one frame)

**Class:** `FeatureEngine` — `src/feature_engine.py`

This is where the actual face reading happens. Called once per frame.

```
FeatureEngine.compute(frame_data)
          │
          ├─→ extract_regions(frame_data.landmarks)   [src/regions.py]
          │     Groups the 478 landmarks into named zones:
          │     forehead, left_eyebrow, right_eyebrow, left_eye, right_eye,
          │     nose_tip, lips (upper/lower), cheeks, chin, jaw, etc.
          │     Returns: dict of region_name → list of (x,y,z) points.
          │
          ├─ Geometric primitives (distances between region centroids):
          │     mouth_width    = distance(left_lip_corner, right_lip_corner)
          │     mouth_open     = distance(upper_lip, lower_lip)
          │     brow_gap_ratio = vertical gap brow↔glabella / face_height
          │     left_ear       = EAR formula on left eye landmarks
          │     right_ear      = EAR formula on right eye landmarks
          │     eye_openness   = min(left_ear, right_ear)   [more robust]
          │
          ├─ Motion primitives (need previous frame):
          │     head_velocity            = nose_tip displacement / dt
          │     motion_energy            = mean squared displacement of all tracked points
          │     micro_motion_energy      = same, but only cheek/moustache points
          │     landmark_displacement_mean = mean of all tracked point displacements
          │
          ├─ Gaze:
          │     gaze_horizontal = iris_x - eye_center_x  (normalised by eye distance)
          │     gaze_vertical   = iris_y - eye_center_y
          │     frontal_contact = soft score: 1.0 when looking straight, drops toward 0
          │
          ├─ Event detection (fires once per event, not every frame):
          │     smile onset        → appends timestamp to smile_events queue
          │     lip compress onset → appends to lip_compression_events
          │     AU20 (lip stretch) → appends to au20_events
          │     blink onset/offset → validates with EAR drop rate, appends to blink_events
          │     gaze shift         → large iris jump, appends to gaze_shift_events
          │     downward gaze      → iris drops below threshold, appends to downward_gaze_events
          │     head nod           → chin vertical velocity spike, appends to gesture_events
          │     speech onset/pause → mouth crosses speaking threshold, tracks latencies
          │
          ├─ Feature assembly (34 values):
          │     au12_mean_amplitude      = mouth_width (smile proxy)
          │     au12_variance            = variance of recent expression scores × 100
          │     au12_activation_frequency= smile events / 60 seconds
          │     au15_mean_amplitude      = lip corner to lower lip distance (sadness)
          │     au4_mean_activation      = 1 - brow_gap_ratio (brow lowering)
          │     au4_duration_ratio       = mean of brow tension history
          │     au1_au2_peak_intensity   = max of recent brow raise values
          │     au20_activation_rate     = lip-stretch events / 60 s
          │     lip_compression_frequency= lip-press events / 60 s
          │     overall_au_variance      = combined mouth+brow variance × 100
          │     facial_emotional_range   = max - min of expression history
          │     facial_transition_frequency = expression jumps / 10 s window
          │     near_zero_au_activation_ratio= fraction of frames with no expression change
          │     mean_head_velocity       = average head speed (recent frames)
          │     head_velocity_peak       = max head speed (recent frames)
          │     head_motion_energy       = average squared landmark displacement × 10000
          │     motion_energy_floor_score= fraction of frames where motion ≈ zero
          │     landmark_displacement_mean= mean displacement of all tracked points
          │     gesture_frequency        = head nods / 10 s window
          │     posture_rigidity_index   = 1 / (1 + variance of head velocity)
          │     micro_motion_energy      = cheek/moustache micro-movement energy
          │     blink_rate               = blinks per minute (last 60 s window)
          │     blink_duration           = average duration of a blink (seconds)
          │     blink_cluster_density    = burst blinks (gap < 2 s) / 60 s
          │     eye_contact_ratio        = mean of frontal contact scores
          │     downward_gaze_frequency  = downward gaze events / 60 s
          │     gaze_shift_frequency     = gaze saccades / 10 s window
          │     baseline_eye_openness    = current median EAR value
          │     response_latency_mean    = mean pause-to-speech latency
          │     speech_onset_delay       = most recent latency
          │     nod_onset_latency        = mean nod lag after speech onset
          │     pause_duration_mean      = mean pause length in seconds
          │     extended_silence_ratio   = fraction of pauses > 2 seconds
          │     reaction_time_instability_index = std dev of response latencies
          │
          └─→ _smooth(feature_name, value)   (called for every feature)
                    Maintains a sliding window (5 frames) per feature.
                    Returns the moving average — reduces jitter.
                    Returns: dict of 34 smoothed raw feature values.
```

**Helper functions inside `feature_engine.py`:**

| Function | What it does |
|---|---|
| `_distance(a, b)` | 3D Euclidean distance between two (x,y,z) points. |
| `_distance_xy(a, b)` | 2D distance (ignores z). Used for EAR and mouth measurements. |
| `_centroid(points)` | Average position of a list of points. Used to get the "centre" of a region. |
| `_ear(landmarks, eye_map)` | Eye Aspect Ratio — the ratio of vertical to horizontal eye opening. Below ~0.21 = blink. |
| `_eye_horizontal_span(...)` | Width of the eye from outer to inner corner. Used to detect eye size asymmetry. |
| `_event_rate(events, now, window)` | Counts how many timestamps in the deque fall within the last `window` seconds. |
| `_all_displacements(curr, prev, ids)` | For each tracked landmark ID, how far did it move since last frame? |
| `_safe_mean(values)` | Mean with a guard for empty lists (returns 0.0 instead of crashing). |
| `_safe_variance(values)` | Variance with a guard for < 2 values. |

---

### Stage 5 — Engineer the Session Vector

**Function:** `engineer_session_vector()` — `predict_video_risk.py`

```
engineer_session_vector(raw_df, feature_cols)
          │
          ├─ Input:  DataFrame of ~2250 rows × 35 columns (timestamp + 34 features)
          │
          ├─ For each of the 34 features, compute 6 summary statistics:
          │     feat__mean    = average value across all frames
          │     feat__std     = how much it varied
          │     feat__min     = lowest value seen
          │     feat__max     = highest value seen
          │     feat__range   = max - min
          │     feat__slope   = linear trend over time (was it going up or down?)
          │
          └─ Returns: dict of 34 × 6 = 204 engineered feature values
                      (the ML model then uses the 63 it was trained on)
```

---

### Stage 6 — Build the Final ML Vector

Back in `extract_from_video()`:

```
engineered = engineer_session_vector(...)   ← 204 values

ml_vector = {
    name: float(engineered[name])
    for name in expected_features          ← pick only the 63 the model needs
}
```

This is the vector that gets saved and later passed to the scoring service.

---

### Stage 7 — Save Everything

**Function:** `_save_session_outputs()` — `extraction_api.py`

```
_save_session_outputs(session_id, payload)
          │
          ├─ Write to reports/api_sessions/{session_id}.json
          │     Full payload: metadata + all 63 feature values + stats.
          │
          └─ Write to reports/api_vectors/{session_id}.json
                Slim payload: just session_id + the 63-value vector.
```

Also saves the raw per-frame CSV:
```
raw_df.to_csv(reports/api_raw_features/api_raw_features_{session_id}.csv)
```

---

### Stage 8 — Return the Response

```
extract_from_video() returns:
{
    "session_id": "8d0c5f7e...",
    "vector_feature_count": 63
}
```

The caller uses the `session_id` to fetch the full vector later.

---

## 📥 Secondary Flow — `GET /extract/session/{session_id}/vector`

**Function:** `get_session_vector()` — `extraction_api.py`

```
GET /extract/session/8d0c5f7e.../vector
          │
          ├─ Look for reports/api_sessions/8d0c5f7e....json
          │
          ├─ Not found? → 404 Not Found
          │
          └─ Found? → Return:
                {
                    "session_id": "8d0c5f7e...",
                    "vector": { "au12_mean_amplitude__mean": 0.4821, ... }
                }
```

---

## 🗺️ Full Data Flow — One Diagram

```
CLIENT sends video file
          │
          ▼
[extraction_api.py]  extract_from_video()
          │
          ├── Validate API key          require_api_key()
          │
          ├── Load settings             RuntimeConfig()
          │                             validate_runtime_config()
          │
          ├── Validate face map         validate_runtime_contract()  [regions.py]
          │
          ├── Load feature names        load_feature_contract()      [contracts.py]
          │
          ├── Resolve model schema      _expected_schema()
          │                             └─ resolve_expected_feature_order()
          │
          ├── Save video to temp file
          │
          ├── Extract raw features      extract_raw_timeseries()     [predict_video_risk.py]
          │       │
          │       ├─ Open video         LandmarkStream.__enter__()   [landmark_stream.py]
          │       │                     └─ cv2.VideoCapture()
          │       │                     └─ mp.FaceMesh()
          │       │
          │       └─ Per-frame loop     LandmarkStream.frames()
          │               │
          │               ├─ Read frame        capture.read()
          │               ├─ Detect face       face_mesh.process()
          │               ├─ Extract points    _extract_landmark_tuples()
          │               │
          │               └─ Compute features  FeatureEngine.compute()   [feature_engine.py]
          │                       │
          │                       ├─ Group landmarks    extract_regions()  [regions.py]
          │                       ├─ Measure distances  _distance(), _ear(), etc.
          │                       ├─ Detect events      smile, blink, gaze, nod…
          │                       ├─ Assemble 34 values
          │                       └─ Smooth values      _smooth()
          │
          ├── Engineer session vector   engineer_session_vector()    [predict_video_risk.py]
          │       └─ 34 features × 6 stats = 204 candidates
          │
          ├── Select 63 model features
          │
          ├── Save outputs              _save_session_outputs()
          │       ├─ api_sessions/{id}.json   (full)
          │       ├─ api_vectors/{id}.json    (vector only)
          │       └─ api_raw_features/{id}.csv
          │
          └── Return { session_id, vector_feature_count: 63 }
                          │
                          ▼
                      CLIENT
```

---

## ⚙️ Supporting Modules — Quick Reference

### `src/config.py`

| Function / Class | What it does |
|---|---|
| `RuntimeConfig` | Dataclass holding all runtime settings: FPS, baseline duration, frame size, all detection thresholds. |
| `FeatureThresholdConfig` | Nested dataclass with values like `blink_ear_threshold=0.21`, `smile_threshold=0.06`. |
| `validate_runtime_config(cfg)` | Checks every value is positive and within a sensible range. Raises `ValueError` if not. |
| `cfg.baseline_min_frames` | Property: how many frames the baseline window covers (baseline_seconds × FPS). |
| `cfg.output_path(project_root)` | Property: resolves and creates the output directory for CSV files. |

---

### `src/contracts.py`

| Function / Class | What it does |
|---|---|
| `load_feature_contract(project_root)` | Reads `docs/feature_definitions_v1.json`, validates it, and returns a `FeatureContract`. |
| `FeatureContract` | Frozen dataclass: holds `feature_order` (list of 34 names), `feature_count`, `schema_version`. |
| `_read_json(path)` | Safe JSON reader — raises `ContractError` on missing file or invalid JSON. |

---

### `src/regions.py`

| Function / Class | What it does |
|---|---|
| `FACIAL_REGIONS` | The master dictionary: 32 named zones, each with a list of MediaPipe landmark IDs. |
| `validate_runtime_contract()` | Called at import time. Checks all 248 IDs are valid, unique, and in range. |
| `validate_region_mapping(mapping)` | Validates any region dict for duplicates and out-of-range IDs. |
| `validate_ear_points(left, right)` | Checks that the 6 required EAR point keys exist for each eye. |
| `extract_regions(landmarks)` | Given 478 raw points, returns a dict of `region_name → [(x,y,z), …]` tuples. |
| `get_region_points(landmarks, name)` | Returns the (x,y,z) tuples for one specific region by name. |
| `flatten_region_ids()` | Returns a flat sorted list of all 248 unique landmark IDs used across all regions. |

---

### `src/landmark_stream.py`

| Function / Class | What it does |
|---|---|
| `LandmarkStream.__init__()` | Stores config and file path. No I/O yet. |
| `LandmarkStream.__enter__()` | Opens video file (or webcam). Starts MediaPipe FaceMesh. |
| `LandmarkStream.__exit__()` | Closes MediaPipe and video capture. Destroys any OpenCV windows. |
| `LandmarkStream.frames()` | Generator. Reads each frame, runs face detection, yields `(FrameData, raw_frame)`. Stops at end of file. |
| `_extract_landmark_tuples(face_landmarks)` | Converts MediaPipe landmark objects into a list of `(x, y, z)` float tuples. |
| `draw_face_landmarks(frame, landmarks)` | Optional visual helper — draws the face mesh on a frame for debugging. |

---

### `src/feature_engine.py`

| Function / Class | What it does |
|---|---|
| `FeatureEngine.__init__()` | Creates smoothing buffers and event queues for all 34 features. |
| `FeatureEngine.compute(frame_data)` | Main method. Takes one frame's landmarks, returns 34 smoothed raw feature values. |
| `FeatureEngine._smooth(name, value)` | Pushes a value into the feature's sliding window, returns the moving average. |
| `FeatureEngineState` | Dataclass holding all per-session memory: event queues, previous frame landmarks, blink state, etc. |
| `FeatureEngineConfig` | Frozen dataclass with all detection thresholds (smile, blink, gaze, etc.). |

---

### `src/normalization.py`

> **Note:** Used in live/webcam mode (`pipeline.py`). Not used in the API video flow — the API does its own statistical engineering instead.

| Function / Class | What it does |
|---|---|
| `BaselineNormalizer` | Collects feature values during the first 20 seconds (baseline window). After that, normalises new values against the baseline. |
| `update_and_maybe_lock(timestamp, raw)` | Adds values to the baseline buffer. Returns `True` the moment the baseline window completes. |
| `normalize(raw_features)` | Applies z-score then sigmoid to map each value to [0–1]. Returns a dict of normalised values. |
| `_compute_stats()` | Calculates mean and std for each feature from the collected baseline samples. |
| `_sigmoid(z)` | `1 / (1 + e^-z)`. Maps any z-score to a 0–1 range. 0.5 = exactly at baseline. |
| `_clip01(value)` | Clamps a value to [0.0, 1.0]. Prevents floating point overflow. |
| `_std(values)` | Standard deviation with a sigma floor of 0.01 — prevents division-by-zero for flat signals. |

---

### `src/csv_logger.py`

| Function / Class | What it does |
|---|---|
| `FeatureCsvLogger.__post_init__()` | Creates the output directory and opens a new timestamped CSV file for writing. |
| `FeatureCsvLogger.write_row(timestamp, features)` | Writes one row: timestamp + all feature values. Validates that all feature names are present. |
| `FeatureCsvLogger.close()` | Flushes and closes the file handle. |

---

### `src/pipeline.py` (Live / Webcam Mode)

This file is for **live webcam sessions**, not the API. It runs the full pipeline with a real-time on-screen display.

| Function | What it does |
|---|---|
| `run_raw_extraction_pipeline(cfg, feature_order, project_root)` | Opens webcam (or video), runs feature extraction, writes raw CSV, shows live window. |
| `_get_protocol_step(elapsed_s)` | Returns the current instruction from the behavioural protocol (e.g. "Look at camera for 10s"). |
| `_build_feature_panel(...)` | Renders all 34 live feature values as text on a side panel for monitoring. |
| `_draw_mapped_landmark_ids(frame, landmarks, ids, view)` | Draws yellow dots and ID numbers on the face for each of the 248 tracked landmarks. |
| `_draw_connections_from_tuples(frame, landmarks, connections, ...)` | Draws lines between landmark pairs to show the face mesh. |
| `_get_connection_catalog()` | Loads MediaPipe's built-in edge lists (tessellation, contours, irises) for drawing. |

---

## 📂 Output Files

After a successful `/extract/video` call, three files are created:

| File | Location | Contents |
|---|---|---|
| Raw frame CSV | `reports/api_raw_features/api_raw_features_{id}.csv` | One row per processed frame. 34 raw feature columns + timestamp. |
| Full session JSON | `reports/api_sessions/{id}.json` | Metadata + 63 engineered feature values + extraction stats. |
| Vector-only JSON | `reports/api_vectors/{id}.json` | Just `session_id` + the 63 ML-ready feature values. |

---

## 🔢 The 34 Raw Features (what `FeatureEngine` produces)

| # | Feature Name | What it measures |
|---|---|---|
| 1 | `au12_mean_amplitude` | How wide the smile is (lip corner distance) |
| 2 | `au12_variance` | How much the smile fluctuates |
| 3 | `au12_activation_frequency` | How often a smile starts |
| 4 | `au15_mean_amplitude` | Lip corner drooping (sadness proxy) |
| 5 | `au4_mean_activation` | Brow lowering / furrowing |
| 6 | `au4_duration_ratio` | How long brows stay tense |
| 7 | `au1_au2_peak_intensity` | Brow raising peak (surprise / fear) |
| 8 | `au20_activation_rate` | Lip stretching sideways (fear grimace) |
| 9 | `lip_compression_frequency` | Deliberate lip pressing (suppressed emotion) |
| 10 | `overall_au_variance` | Whole-face emotional variability |
| 11 | `facial_emotional_range` | Gap between most and least expressive frames |
| 12 | `facial_transition_frequency` | How often expression changes dramatically |
| 13 | `near_zero_au_activation_ratio` | Fraction of frames with no expression change (blunted affect) |
| 14 | `mean_head_velocity` | Average head movement speed |
| 15 | `head_velocity_peak` | Fastest head movement in recent frames |
| 16 | `head_motion_energy` | Overall physical activity |
| 17 | `motion_energy_floor_score` | Fraction of frames where person is nearly still |
| 18 | `landmark_displacement_mean` | Average point-to-point movement across face |
| 19 | `gesture_frequency` | Head nod rate |
| 20 | `posture_rigidity_index` | How stiff / unchanging the head movement is |
| 21 | `micro_motion_energy` | Tiny cheek / upper lip tremor |
| 22 | `blink_rate` | Blinks per minute |
| 23 | `blink_duration` | How long each blink lasts |
| 24 | `blink_cluster_density` | Rapid-fire bursts of blinking |
| 25 | `eye_contact_ratio` | How consistently the person faces the camera |
| 26 | `downward_gaze_frequency` | How often gaze drops significantly downward |
| 27 | `gaze_shift_frequency` | How often the eyes dart sideways |
| 28 | `baseline_eye_openness` | Current eye opening (EAR value) |
| 29 | `response_latency_mean` | Average pause before speaking |
| 30 | `speech_onset_delay` | Most recent pause-to-speech gap |
| 31 | `nod_onset_latency` | How quickly person nods after speaking |
| 32 | `pause_duration_mean` | Average silence duration |
| 33 | `extended_silence_ratio` | Fraction of pauses longer than 2 seconds |
| 34 | `reaction_time_instability_index` | How inconsistent the response timing is |

---

*End of document.*