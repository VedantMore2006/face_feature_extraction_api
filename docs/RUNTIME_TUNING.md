# Runtime Tuning And Usage

This document captures the final operating flow for the current runtime pipeline.

## Run Commands

Use your conda environment:

1. `conda activate face_env`
2. `python main.py --check-only`
3. `python main.py`

## Current Pipeline Behavior

1. Webcam stream starts.
2. MediaPipe landmarks are extracted each frame.
3. 33 raw features are computed.
4. First 30 seconds are used for baseline collection.
5. After baseline lock, features are normalized by:
   - z-score
   - sigmoid scaling
6. Normalized rows are written to CSV in data directory.

Output CSV naming pattern:

- `data/features_YYYYMMDD_HHMMSS.csv`

## Runtime Controls

During live run:

- Press `q` to quit.
- Press `b` to reset baseline and begin baseline collection again.

## Where To Tune Constants

Tune feature sensitivity in:

- `src/config.py`

Specifically in:

- `FeatureThresholdConfig`

Available tuning fields:

- `event_window_seconds`
- `smile_threshold`
- `mouth_open_threshold`
- `lip_compression_threshold`
- `brow_tension_threshold`
- `blink_ear_threshold`
- `gaze_shift_threshold`
- `downward_gaze_threshold`
- `motion_transition_threshold`
- `nod_velocity_threshold`
- `extended_silence_threshold`

## Suggested Tuning Workflow

1. Run for 60 to 90 seconds in neutral expression.
2. Watch baseline phase complete and CSV begin.
3. Test one behavior at a time:
   - Smile
   - Blink repeatedly
   - Nods
   - Gaze shifts
4. Inspect resulting CSV trends.
5. Adjust one threshold at a time and rerun.

## Safety Checks Already Enforced

1. Config validation blocks invalid thresholds.
2. Region mapping validation runs at startup.
3. Missing feature keys are rejected before normalization and CSV write.
4. CSV logger rejects values outside [0, 1].
5. All normalized values are rounded to 8 decimals.
