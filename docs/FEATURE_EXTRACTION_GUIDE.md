# Comprehensive Feature Extraction Guide

**Document Version:** 1.0  
**Last Updated:** 2026-06-10  
**Scope:** Facial behavioral feature extraction pipeline for mental health screening

---

## Table of Contents

1. [Overview & Architecture](#overview--architecture)
2. [The 34 Primary Raw Features](#the-34-primary-raw-features)
3. [Feature Extraction Methods](#feature-extraction-methods)
4. [Normalization & Z-Score Process](#normalization--z-score-process)
5. [The 63 Engineered Features Sent to Model](#the-63-engineered-features-sent-to-model)
6. [Feature Importance Rankings](#feature-importance-rankings)
7. [Quick Reference by Clinical Domain](#quick-reference-by-clinical-domain)

---

## Overview & Architecture

### What Is This Pipeline?

This pipeline transforms **real-time facial landmarks from MediaPipe** (478 3D points tracked on a face) into **34 high-level behavioral features** (computed per frame), then **aggregates those over a session** to produce **63 engineered features** that feed into a machine learning classifier.

### The Three-Stage Flow

```
Stage 1: Per-Frame Raw Extraction
┌─────────────────────────────────────────────┐
│ MediaPipe Face Landmarks (478 points)       │
│ + Geometric calculations & event detection  │
│ + Temporal smoothing                        │
└─────────────┬───────────────────────────────┘
              │
              ↓
         [34 Raw Features per frame]
              │
    (e.g., au12_mean_amplitude=0.048,
            blink_rate=15 events/min, ...)
              │
     ┌────────┴────────┐
     │ Multiple frames │
     │ (150 sec session)
     └────────┬────────┘
              │
              ↓
Stage 2: Session Aggregation (Engineer Session Vector)
┌─────────────────────────────────────────────┐
│ For each of 34 raw features, compute:        │
│  - __mean (average over 150 sec)             │
│  - __std (variability)                       │
│  - __min (minimum value seen)                │
│  - __max (maximum value seen)                │
│  - __range (max - min)                       │
│  - __slope (trend over time)                 │
└─────────────┬───────────────────────────────┘
              │
              ↓
         [204 Aggregated Features]
              │
Stage 3: Normalization & Selection
┌─────────────────────────────────────────────┐
│ Per-feature baseline collection (first 30s) │
│ Compute mean & std from baseline            │
│ Apply z-score normalization → sigmoid scale │
│ Select 63 most discriminative features      │
└─────────────┬───────────────────────────────┘
              │
              ↓
    [63 Normalized Features to Model]
              │
              ↓
         [Risk Classification]
         (anxiety, depression, bipolar, etc.)
```

---

## The 34 Primary Raw Features

### Feature Computation Principles

Every raw feature is computed **per frame** (every video frame sent to the feature engine):

- **Geometric distances** → differences between facial landmarks (e.g., mouth width = distance between left/right lip corners).
- **Event detection** → binary state changes (e.g., did a blink occur in this frame?).
- **Rates** → events per time window (e.g., blinks per minute).
- **Temporal smoothing** → raw feature values are smoothed with a 5-frame rolling average to reduce noise.

### Group 1: Facial Expression (AU / AU = Action Unit from FACS)

**Action Units (AUs)** are standardized facial muscle movements defined in the Facial Action Coding System (FACS). These 13 features track distinct expression patterns.

#### AU12 - Smile / Lip Corner Puller

| Attribute | Value |
|-----------|-------|
| **Feature Name** | `au12_mean_amplitude` |
| **Extracted From** | Lip corner landmarks (left & right corners, upper & lower lip border) |
| **Computation** | Distance between left and right mouth corners; normalized by face height |
| **Signal Type** | Continuous distance (0.0 → 1.0 scale within face) |
| **What It Means** | Strength of smile/lip-corner pulling upward |
| **Clinical Significance** | Reduced AU12 may indicate affective flattening (depression, schizophrenia); elevated in hypomanic or manic states |
| **Raw Value Range** | ~0.02–0.08 |

| Feature Name | Extraction Method | Significance |
|---|---|---|
| `au12_variance` | Frame-to-frame variability of lip-corner distance | Measure of smile stability. High variance = unstable smile; low variance = flat or fixed smile |
| `au12_activation_frequency` | Count of frames where AU12 > smile_threshold (0.06) per 60-second window | Frequency of smiling. Low frequency may suggest reduced positive affect |

#### AU15 - Sad Mouth / Lip Corner Depressor

| Feature Name | Extraction Method | Significance |
|---|---|---|
| `au15_mean_amplitude` | Distance from lip corners down to lower lip; normalized by face height | Downward mouth corners indicate sadness or contempt. Elevated in depression |

#### AU4 - Brow Lowerer / Furrow

| Feature Name | Extraction Method | Significance |
|---|---|---|
| `au4_mean_activation` | Vertical distance from brow centroid to glabella (between-eyebrows), normalized by face height; inverted so 1.0 = fully lowered | Brow furrowing indicates worry, anger, or concentration. Elevated in anxiety, OCD, and depression |
| `au4_duration_ratio` | Fraction of frames where AU4 > threshold | Persistence of furrowing. High ratio = sustained tension |

#### AU1 & AU2 - Brow Raiser

| Feature Name | Extraction Method | Significance |
|---|---|---|
| `au1_au2_peak_intensity` | Peak co-activation of inner + outer brow raise (measured as upward movement + vertical distance to forehead) | Sudden alert/surprise spike. Elevated during acute stress, fear, or startled response |

#### AU20 - Lip Stretch / Fear Mouth

| Feature Name | Extraction Method | Significance |
|---|---|---|
| `au20_activation_rate` | Count of frames where upper-lower lip horizontal stretch exceeds threshold (0.055) per 60-second window | Fear/tension mouth response. Rare in neutral; elevated in anxiety and panic |

#### Lip Compression

| Feature Name | Extraction Method | Significance |
|---|---|---|
| `lip_compression_frequency` | Count of frames where lip opening < 0.010 (very tight compression) per 60-second window | Emotional suppression or anger control. Elevated in individuals suppressing feelings |

#### Overall Expression Variance

| Feature Name | Extraction Method | Significance |
|---|---|---|
| `overall_au_variance` | Aggregate variance across all tracked AU proxies (AU12, AU15, AU4, AU1, AU2) | Global facial dynamism. High variance = expressive face; low variance = flat/blunted affect (seen in depression, autism spectrum) |
| `facial_emotional_range` | Max-minus-min span of expression intensities within the session | Emotional range breadth. Narrow range = restricted affect |
| `facial_transition_frequency` | Rate of expression state transitions (frames where expression intensity changes > 0.12) per 60-second window | Expression flexibility. Low frequency = rigid/fixed expressions; high = rapid mood shifts |
| `near_zero_au_activation_ratio` | Fraction of frames where all AU proxies are < 0.02 (nearly expressionless) | Affective flattening indicator. High ratio characteristic of depression, negative symptoms in psychosis |

---

### Group 2: Head Motion & Posture

These 8 features track gross motor behavior, which often reflects psychomotor changes in mental health conditions.

| Feature Name | Extraction Method | Significance |
|---|---|---|
| `mean_head_velocity` | Mean speed of head movement (nose_tip, chin landmarks) frame-to-frame, per 60-sec window | General motor tempo. Low velocity = psychomotor slowing (depression); high = agitation (mania, anxiety) |
| `head_velocity_peak` | Maximum instantaneous head movement speed observed in session | Agitation spike marker. Single sudden head movement |
| `head_motion_energy` | Summed magnitude of head displacement across all frames (energy-like metric) | Overall motor activation. High energy = hypermotor, low = psychomotor retardation |
| `motion_energy_floor_score` | Proportion of time spent in near-zero motion baseline | Prolonged stillness. High score = rigid/withdrawn posture (depression); low = restless |
| `landmark_displacement_mean` | Mean frame-to-frame displacement across key landmark indices | Subtle facial micro-movement level. Reflects fidgeting, anxiety-related tension |
| `gesture_frequency` | Count of detected head/chin gesture-like events (nod-like motion) per 60-second window | Conversational motor engagement. Low in withdrawn/depressed states; high in engaged/hypomanic |
| `posture_rigidity_index` | Inverse of postural anchor variability (stiffness/rigidity marker) | How fixed or rigid posture appears. High = tension/guardedness; low = relaxed/natural |
| `micro_motion_energy` | Energy of very small movement components (< 0.01 units per frame) | Restlessness/fidgeting signature. Elevated in anxiety, ADHD, substance use |

---

### Group 3: Eye & Gaze Behavior

These 7 features encode eye contact, blinking, and gaze patterns—key indicators of social engagement, fatigue, and cognitive load.

#### Blink Dynamics

| Feature Name | Extraction Method | Significance |
|---|---|---|
| `blink_rate` | Count of blink events per 60-second window using Eye Aspect Ratio (EAR < 0.21) dynamics | Autonomic/cognitive load proxy. Elevated blink rate indicates stress or cognitive overload; depressed in certain mood states |
| `blink_duration` | Average closure duration of each blink event (milliseconds converted to seconds) | Brief vs. heavy blink tendency. Prolonged blinks indicate fatigue or dissociation |
| `blink_cluster_density` | Density of rapid blink bursts (inter-blink interval < 2.0 seconds) per 60-second window | Stress/arousal burst pattern. Rapid clustered blinking seen in anxiety attacks, cognitive overload |

#### Eye Contact & Gaze

| Feature Name | Extraction Method | Significance |
|---|---|---|
| `eye_contact_ratio` | Proportion of frames where gaze direction is frontal (iris displacement < 0.15 IPD units from center) | Social engagement indicator. Low eye contact = withdrawal (depression, anxiety, autism); high = engaged |
| `downward_gaze_frequency` | Count of frames where gaze is notably downward (> 0.06 IPD units below center) per 60-second window | Withdrawal/avoidance tendency. Elevated in shame, depression, social anxiety |
| `gaze_shift_frequency` | Rate of significant directional gaze shifts per 60-second window | Scanning/hypervigilance. Low = withdrawn; high = anxious/hypervigilant scanning |
| `baseline_eye_openness` | Mean Eye Aspect Ratio (EAR) from baseline dynamics; normalized by median | Alertness/fatigue proxy. High EAR = alert; low = droopier/fatigued eyes |

---

### Group 4: Temporal Response & Timing

These 6 features capture reaction speed, speech initiation delays, and conversational pauses—markers of cognitive/emotional processing.

| Feature Name | Extraction Method | Significance |
|---|---|---|
| `response_latency_mean` | Mean delay from conversation prompt/stimulus to visible response initiation (milliseconds → seconds) | Processing hesitation marker. Elevated latency = slow processing, depression, dissociation |
| `speech_onset_delay` | Delay from stimulus to mouth-opening/speech initiation | Speech initiation speed. Delayed onset indicates anxiety, avoidance, or psychomotor slowing |
| `nod_onset_latency` | Delay to first acknowledgment nod-like motion | Social feedback speed. Delayed nods = disengagement or depressive withdrawal |
| `pause_duration_mean` | Average conversational pause duration (inter-speech gaps) | Conversational tempo. Long pauses indicate depression, stress, thought disorder |
| `extended_silence_ratio` | Fraction of session time spent in pauses > 2.0 seconds | Severe slowing/withdrawal. High ratio = severe depression or catatonia |
| `reaction_time_instability_index` | Variability (std) of response latencies across repeated stimuli | Timing instability marker. High instability = inconsistent cognitive engagement, possible dissociation or bipolar mixed state |

---

## Feature Extraction Methods

### 1. Geometric Computations

#### Distance Calculation

```
distance(point_A, point_B) = √[(x_A - x_B)² + (y_A - y_B)² + (z_A - z_B)²]
```

Used for: AU amplitudes, mouth width, brow movements, iris positions.

#### 2D Distance (XY Plane Only)

```
distance_xy(point_A, point_B) = √[(x_A - x_B)² + (y_A - y_B)²]
```

Used for: Eye aspect ratio (EAR) calculations, frontal gaze direction.

#### Centroid (Average Position)

```
centroid(points) = (mean(x₁..xₙ), mean(y₁..yₙ), mean(z₁..zₙ))
```

Used for: Region centers (lips, brows, iris, etc.).

### 2. Event Detection (State Machines)

Each event is tracked as a time-series of timestamps. When an event crosses a threshold, it is recorded.

**Example: Smile Detection**
```
if au12_amplitude > smile_threshold (0.06):
    event_state = ON
    record_timestamp(now)
else:
    event_state = OFF
```

**Example: Blink Detection (EAR-based)**
```
if EAR < 0.21 for ≥ 2 consecutive frames:
    blink_state = CLOSING
    record_start_timestamp(now)
if EAR rises back > 0.21 after closure:
    blink_state = OPEN
    record_end_timestamp(now)
    compute_blink_duration(end - start)
```

### 3. Rate Calculations

For event-based features, a sliding time window is maintained:

```
event_rate = count_of_events_within_window / window_duration_seconds
```

**Example: Blink Rate**
```
blink_events = [t₁, t₂, t₃, ...] timestamps of blink starts
window = current_timestamp - 60 seconds
blink_rate = count(events where t > window) / 60.0
```

### 4. Temporal Smoothing

Every raw feature output is smoothed using a rolling average:

```
smoothed_value = mean(buffer[last_N_frames])
```
where N = `smoothing_window` (default: 5 frames).

**Why smooth?** Reduces frame-to-frame noise while preserving true behavioral trends.

### 5. Normalization by Face Dimensions

To make features invariant to camera distance, geometric measurements are normalized by face height:

```
normalized_feature = raw_distance / face_height
```

**Example: AU4 (Brow Lowerer)**
```
brow_vertical_gap = |brow_y - glabella_y|
face_height = |chin_y - forehead_y|
au4_activation = brow_vertical_gap / face_height
```

This ensures a person 1 meter away has similar AU4 values as a person 2 meters away.

---

## Normalization & Z-Score Process

### Stage 1: Baseline Collection (First ~30 Seconds)

During the first 30 seconds of a session, raw feature values are collected without any modeling or inference.

```
At each frame:
  raw_features = compute_frame_features()
  baseline_buffer[feature_name].append(raw_features[feature_name])
  
After 30 seconds:
  For each feature:
    baseline_mean[feature] = mean(baseline_buffer[feature])
    baseline_std[feature] = std(baseline_buffer[feature], ddof=1)
```

### Stage 2: Z-Score Normalization

After baseline is locked, incoming raw features are converted to **z-scores**:

```
z_score = (raw_value - baseline_mean) / max(baseline_std, sigma_floor)
```

**What does this do?**
- Centers each feature around its personal baseline (mean = 0).
- Scales by variability (std); features with low natural variation get amplified.
- `sigma_floor = 0.01` prevents division by zero for ultra-stable features.

**Example:**
```
raw_au12_amplitude = 0.055
baseline_mean_au12 = 0.050
baseline_std_au12 = 0.008

z_score = (0.055 - 0.050) / 0.008 = 0.625
```

### Stage 3: Sigmoid Scaling to [0, 1]

Z-scores are then mapped to a probability-like [0, 1] range using the sigmoid function:

```
sigmoid(z) = 1 / (1 + e^(-z))
```

**Properties:**
- z = 0 → sigmoid(0) ≈ 0.5 (neutral/baseline)
- z > 0 → sigmoid(z) > 0.5 (above baseline)
- z < 0 → sigmoid(z) < 0.5 (below baseline)
- Asymptotes at 0 and 1, never exceeding bounds

**Example (continued):**
```
sigmoid(0.625) ≈ 0.651
```

### Stage 4: Final Clamping & Precision

```
final_value = round(clamp(sigmoid(z), 0.0, 1.0), 8 decimals)
```

**Result:** Each feature normalized to [0, 1] with 8 decimal precision.

### Why This Approach?

1. **Personalization:** Each person's baseline is unique. Z-score centers on their own baseline.
2. **Scale-invariance:** Accounts for natural person-to-person differences.
3. **Probability interpretation:** Sigmoid output can be interpreted as "how unusual is this relative to baseline?" (0=normal, 1=extreme).
4. **Stability:** Sigma floor prevents numerical instability for zero-variance features.

---

## The 63 Engineered Features Sent to Model

### Aggregation Strategy

After a full session (~150 seconds), the 34 per-frame raw features are aggregated using 6 statistics per feature:

```
For each raw feature F:
  F__mean  = mean(F across all frames)
  F__std   = std(F across all frames, ddof=1)
  F__min   = min(F across all frames)
  F__max   = max(F across all frames)
  F__range = max(F) - min(F)
  F__slope = linear regression slope (frames as x-axis)
```

**34 raw features × 6 aggregations = 204 candidate features**

**Model selection:** The training process selected the **63 most discriminative features** for the final model.

### The 63 Engineered Features (Ranked by Importance)

| Rank | Feature Name | Importance | Clinical Interpretation |
|------|---|---|---|
| 1 | `nod_onset_latency__mean` | 0.2206 | **Most Important:** Average delay to acknowledgment nodding. Elevated in depression, psychomotor retardation |
| 2 | `reaction_time_instability_index__range` | 0.1624 | Range of reaction time instability; high = inconsistent engagement, dissociation |
| 3 | `speech_onset_delay__min` | 0.1244 | Minimum (fastest) speech onset delay; low indicates intact processing speed |
| 4 | `pause_duration_mean__range` | 0.0809 | Variability of pause lengths; unstable = mood shifts or thought disorder |
| 5 | `baseline_eye_openness__max` | 0.0602 | Peak eye openness; low = fatigue or depression |
| 6 | `reaction_time_instability_index__slope` | 0.0437 | Trend in instability over session; increasing = decompensation |
| 7 | `extended_silence_ratio__max` | 0.0351 | Proportion of extended silences; high = severe withdrawal |
| 8 | `au4_mean_activation__slope` | 0.0341 | Trend in brow tension; increasing = escalating worry |
| 9 | `au4_duration_ratio__std` | 0.0305 | Variability in brow tension duration; unstable = mood lability |
| 10 | `nod_onset_latency__min` | 0.0259 | Fastest nod response; slow = disengagement |
| 11 | `facial_transition_frequency__mean` | 0.0248 | Average expression transition rate; low = flat, high = labile |
| 12 | `response_latency_mean__min` | 0.0247 | Fastest response latency; slow = cognitive slowing |
| 13 | `gaze_shift_frequency__range` | 0.0227 | Variability in gaze shift rate; unstable = anxiety/hypervigilance |
| 14 | `extended_silence_ratio__slope` | 0.0217 | Trend in silence; increasing = worsening withdrawal |
| 15 | `blink_cluster_density__mean` | 0.0170 | Average clustering of blinks; high = stress/load |
| 16–63 | [47 additional features] | 0.0149 – 0.0001 | Progressively decreasing importance |

*(For complete list see [Feature Importance Rankings](#feature-importance-rankings) section below.)*

---

## Feature Importance Rankings

### Full 63-Feature Importance Table (Ranked)

| Rank | Feature | Importance | Group | Significance |
|------|---------|-----------|-------|---|
| 1 | nod_onset_latency__mean | 0.2206 | Temporal | Very High |
| 2 | reaction_time_instability_index__range | 0.1624 | Temporal | Very High |
| 3 | speech_onset_delay__min | 0.1244 | Temporal | Very High |
| 4 | pause_duration_mean__range | 0.0809 | Temporal | High |
| 5 | baseline_eye_openness__max | 0.0602 | Eye/Gaze | High |
| 6 | reaction_time_instability_index__slope | 0.0437 | Temporal | High |
| 7 | extended_silence_ratio__max | 0.0351 | Temporal | High |
| 8 | au4_mean_activation__slope | 0.0341 | Expression | High |
| 9 | au4_duration_ratio__std | 0.0305 | Expression | High |
| 10 | nod_onset_latency__min | 0.0259 | Temporal | Moderate-High |
| 11 | facial_transition_frequency__mean | 0.0248 | Expression | Moderate-High |
| 12 | response_latency_mean__min | 0.0247 | Temporal | Moderate-High |
| 13 | gaze_shift_frequency__range | 0.0227 | Eye/Gaze | Moderate-High |
| 14 | extended_silence_ratio__slope | 0.0217 | Temporal | Moderate-High |
| 15 | blink_cluster_density__mean | 0.0170 | Eye/Gaze | Moderate |
| 16 | au12_variance__min | 0.0149 | Expression | Moderate |
| 17 | speech_onset_delay__mean | 0.0102 | Temporal | Moderate |
| 18 | speech_onset_delay__max | 0.0076 | Temporal | Moderate |
| 19 | head_motion_energy__min | 0.0044 | Motion | Moderate |
| 20 | au4_duration_ratio__mean | 0.0043 | Expression | Moderate |
| 21 | head_velocity_peak__min | 0.0043 | Motion | Moderate |
| 22 | extended_silence_ratio__mean | 0.0039 | Temporal | Low-Moderate |
| 23 | blink_rate__range | 0.0034 | Eye/Gaze | Low-Moderate |
| 24 | landmark_displacement_mean__max | 0.0024 | Motion | Low-Moderate |
| 25 | au12_activation_frequency__mean | 0.0021 | Expression | Low-Moderate |
| 26 | response_latency_mean__slope | 0.0020 | Temporal | Low |
| 27 | micro_motion_energy__max | 0.0018 | Motion | Low |
| 28 | landmark_displacement_mean__std | 0.0017 | Motion | Low |
| 29 | speech_onset_delay__slope | 0.0013 | Temporal | Low |
| 30 | lip_compression_frequency__slope | 0.0011 | Expression | Low |
| 31 | au20_activation_rate__slope | 0.0011 | Expression | Low |
| 32 | nod_onset_latency__slope | 0.0008 | Temporal | Low |
| 33 | overall_au_variance__slope | 0.0007 | Expression | Low |
| 34 | baseline_eye_openness__std | 0.0006 | Eye/Gaze | Low |
| 35 | micro_motion_energy__std | 0.0006 | Motion | Low |
| 36 | motion_energy_floor_score__slope | 0.0002 | Motion | Low |
| 37 | au12_variance__slope | 0.0002 | Expression | Low |
| 38 | pause_duration_mean__slope | 0.0002 | Temporal | Low |
| 39 | head_motion_energy__slope | 0.0002 | Motion | Low |
| 40 | extended_silence_ratio__min | 0.0002 | Temporal | Low |
| 41 | au4_duration_ratio__slope | 0.0002 | Expression | Low |
| 42 | micro_motion_energy__slope | 0.0001 | Motion | Low |
| 43 | au12_mean_amplitude__slope | 0.0001 | Expression | Low |
| 44 | au12_activation_frequency__slope | 0.0001 | Expression | Very Low |
| 45 | posture_rigidity_index__slope | 0.0001 | Motion | Very Low |
| 46 | gesture_frequency__slope | 0.0001 | Motion | Very Low |
| 47 | blink_cluster_density__slope | 0.0001 | Eye/Gaze | Very Low |
| 48 | facial_transition_frequency__slope | 0.0001 | Expression | Very Low |
| 49 | facial_transition_frequency__min | 0.0001 | Expression | Very Low |
| 50 | blink_rate__slope | 0.0001 | Eye/Gaze | Very Low |
| 51 | landmark_displacement_mean__slope | 0.0001 | Motion | Very Low |
| 52 | mean_head_velocity__slope | 0.0001 | Motion | Very Low |
| 53 | near_zero_au_activation_ratio__slope | 0.0001 | Expression | Very Low |
| 54 | downward_gaze_frequency__slope | 0.0001 | Eye/Gaze | Very Low |
| 55 | facial_emotional_range__slope | 0.0001 | Expression | Very Low |
| 56 | blink_duration__slope | 0.0001 | Eye/Gaze | Very Low |
| 57 | eye_contact_ratio__slope | 0.0001 | Eye/Gaze | Very Low |
| 58 | head_velocity_peak__slope | 0.0001 | Motion | Very Low |
| 59 | au1_au2_peak_intensity__slope | 0.0001 | Expression | Very Low |
| 60 | au15_mean_amplitude__slope | 0.0001 | Expression | Very Low |
| 61 | gaze_shift_frequency__slope | 0.0001 | Eye/Gaze | Very Low |
| 62 | blink_rate__std | 0.0001 | Eye/Gaze | Very Low |
| 63 | baseline_eye_openness__slope | 0.0001 | Eye/Gaze | Very Low |

---

## Quick Reference by Clinical Domain

### Depression Screening

**Most Relevant Features:**
- `pause_duration_mean` (elevated pauses)
- `extended_silence_ratio` (high silence)
- `au4_mean_activation` (furrowing/worry)
- `response_latency_mean` (slow responses)
- `nod_onset_latency` (delayed acknowledgment)
- `baseline_eye_openness` (droopy eyes / fatigue)

**Pattern:** Slow, sparse speech; furrowed brow; low engagement.

### Anxiety Screening

**Most Relevant Features:**
- `blink_cluster_density` (blink bursts)
- `gaze_shift_frequency` (scanning/hypervigilance)
- `au20_activation_rate` (lip stretch/fear)
- `head_velocity_peak` (sudden movements)
- `micro_motion_energy` (fidgeting)
- `au4_mean_activation` (sustained tension)

**Pattern:** Rapid eye movements; fidgeting; lip tension; frequent blinks.

### Bipolar Disorder (Manic/Hypomanic)

**Most Relevant Features:**
- `facial_transition_frequency` (rapid expression changes)
- `gesture_frequency` (increased motor activity)
- `au12_mean_amplitude` (elevated smiling)
- `gaze_shift_frequency` (quick gaze changes)
- `speech_onset_delay__min` (fast response)
- `reaction_time_instability_index` (erratic timing)

**Pattern:** Rapid mood shifts; expressive face; increased motor activity; fast responses.

### Stress/General Arousal

**Most Relevant Features:**
- `blink_cluster_density` (stress blinks)
- `micro_motion_energy` (restlessness)
- `head_motion_energy` (increased movement)
- `speech_onset_delay` (hesitation)
- `au4_mean_activation` (sustained tension)

**Pattern:** Increased overall motor activity; fidgeting; tension.

---

## Implementation Notes

### Sampling Strategy

- **Default frame stride:** 2 (process every 2nd frame) for balanced accuracy/speed.
- **Baseline window:** First 30 seconds of session.
- **Session duration:** Recommended ≥150 seconds for stable feature estimates.

### Feature Order

Features are always processed in the exact order specified in `docs/feature_definitions_v1.json`. This ensures:
- CSV headers match model input order.
- Reproducible aggregation.
- Compatibility with pre-trained models.

### Thresholds & Configuration

All extraction thresholds (smile_threshold, blink_ear_threshold, etc.) are defined in `src/feature_engine.py` as `FeatureEngineConfig` defaults. These can be tuned per deployment:
- Lower threshold = more sensitive detection.
- Higher threshold = fewer false positives.

### Validation & Quality Checks

Each session is validated for:
1. Minimum duration (≥150 seconds, configurable).
2. Minimum face frames detected (≥70% of frames).
3. No NaN values in engineered features.
4. Feature bounds (after sigmoid, all in [0, 1]).

---

## Frequently Asked Questions

### Q: Why 34 raw features and not more/fewer?

**A:** The 34 features were selected to balance:
- **Coverage:** All major behavioral domains (expression, motion, gaze, timing).
- **Interpretability:** Each feature has clear facial/behavioral meaning.
- **Computational efficiency:** Real-time extraction on standard hardware.
- **Clinical relevance:** Alignment with established behavioral markers in mental health.

### Q: Why aggregate into 63 features instead of using raw features directly?

**A:** Raw per-frame features are highly temporal and sensitive to noise. Aggregation:
- Reduces session-level noise through averaging.
- Captures temporal trends (slope).
- Enables stable baseline-to-model comparison.
- Improves model generalization.

### Q: What if someone's baseline is unusual (e.g., naturally very low blink rate)?

**A:** Z-score normalization handles this:
- Each person's baseline becomes 0.5 (neutral).
- Deviations from their own baseline are what matter.
- A "low" blink rate for one person might be "high" for another—but both are normalized relative to their own baseline.

### Q: Can I use these features for real-time inference?

**A:** Not until baseline is locked (~30 seconds). After baseline lock:
- Yes, features are normalized and ready per-frame.
- However, aggregated features (mean, slope, etc.) require ≥150 seconds.

### Q: What's the difference between "raw" and "engineered" features?

**Raw (34 per frame):**
- Computed immediately at each frame.
- Specific to that instant (e.g., "smile strength now").
- Smoothed with 5-frame buffer to reduce noise.

**Engineered (63 session-level):**
- Computed after full session is complete.
- Aggregated statistics (mean, std, slope) across 150 seconds.
- Sent to ML model for classification.

### Q: How are trends ("slope") computed?

**A:** Linear regression with frame index as x-axis:
```
slope = Σ[(t - mean(t)) * (value - mean(value))] / Σ[(t - mean(t))²]
```

Positive slope = increasing over session; negative = decreasing.

---

## References & Further Reading

1. **FACS (Facial Action Coding System):** Ekman, P., & Friesen, W. V. (2002). Facial Action Coding System.
2. **Eye Aspect Ratio (EAR):** Terriberry, T. T., et al. (2017). Real-time eye tracking and gaze detection.
3. **MediaPipe Face Mesh:** https://google.github.io/mediapipe/solutions/face_mesh.html
4. **Z-Score Normalization:** Standard statistical technique for feature scaling.
5. **Sigmoid Function:** Commonly used in ML for probability mapping.

---

## Document History

| Date | Version | Changes |
|------|---------|---------|
| 2026-06-10 | 1.0 | Initial comprehensive guide. Includes all 34 raw + 63 engineered features. Normalization process documented. Feature importance rankings included. |

---

**Questions or feedback?** Review the source code files:
- `src/feature_engine.py` — Raw feature computation.
- `src/normalization.py` — Baseline & z-score logic.
- `predict_video_risk.py` — Session aggregation (engineer_session_vector).
- `docs/feature_definitions_v1.json` — Canonical feature order.
