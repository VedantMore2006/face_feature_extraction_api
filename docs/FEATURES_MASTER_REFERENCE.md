# Facial Behavioral Features Master Reference

This document is the unified reference for:

- All behavioral features currently used by the pipeline.
- Basic meanings of each feature (computational, behavioral, and practical interpretation).
- MediaPipe landmark-to-region organization used in this project.
- Feature-to-region mapping (which features depend on which facial regions).

## 0. Project Synopsis, Intent, And Screening Decisions

### Project Overview

This project is building a privacy-safe facial behavioral analysis system that transforms real-time MediaPipe facial landmarks into high-level behavioral features. The output is a structured, normalized feature stream that can be used for behavioral monitoring, conversational analysis, and research-grade signal tracking without storing raw video or identity-reconstructable face geometry.

### Core Intent

The intent is to estimate behavioral state patterns, not to perform identity recognition and not to claim medical diagnosis from a single session. The platform is designed to:

- Quantify expression, motion, gaze, and timing behavior in a reproducible way.
- Compare behavior against each person\'s own session baseline rather than fixed population assumptions.
- Support explainable feature-level interpretation through explicit region and signal mapping.
- Enable screening-oriented triage signals that can prompt follow-up assessment.

### Business Type And Product Positioning

- Business type: Behavioral and mental health screening support.
- Primary product role: Early risk-screening and triage decision support from facial behavioral signals.
- Deployment style: Privacy-safe digital screening workflow for intake, monitoring, and follow-up prioritization.

### Mental Health Conditions In Screening Scope

This screening framework is intended to support early detection signals for the following condition categories:

- Depression
- Anxiety
- Stress-related burden
- Bipolar disorder spectrum
- Substance dependency risk patterns
- Phobia-related response patterns

Note: The system performs screening support, not final diagnosis. Diagnostic decisions remain clinical.

### What Is Being Screened

The system screens for behavioral tendencies and risk indicators that can support human decision-making:

- Reduced expressivity or affective flattening.
- Psychomotor slowing, rigidity, or unusually low movement energy.
- Agitation, restlessness, and abrupt motion spikes.
- Gaze withdrawal patterns, reduced eye contact, or excessive gaze shifts.
- Blink dynamics associated with stress, fatigue, or cognitive load.
- Delayed response initiation, longer pauses, and unstable reaction timing.

### Decision-Support Questions This Pipeline Helps Answer

During session review, feature outputs are intended to support the following screening decisions:

- Is current behavior broadly within the person\'s baseline range, or significantly shifted?
- Is the dominant pattern flattening/withdrawal, or agitation/hyperarousal?
- Are temporal-response signals showing hesitation, slowing, or instability?
- Are gaze and blink signals consistent with engagement, fatigue, or overload?
- Does the current session warrant follow-up observation, escalation, or detailed clinical review?

### Decision Boundaries (Important)

- This is a behavioral screening and monitoring system, not a standalone diagnostic engine.
- Feature values should be interpreted longitudinally and contextually, not in isolation.
- Any clinical conclusion must involve qualified professionals and multi-source evidence.

---

## 1. Scope And Canonical Sources

This reference aligns with the frozen runtime contracts in:

- `docs/feature_definitions_v1.json` (canonical feature order and definitions)
- `docs/region_mapping_v1.json` (canonical facial region mapping)
- `src/regions.py` (runtime validation and extraction contract)

Current frozen contract snapshot:

- Schema version: 1.0.0
- Frozen on: 2026-03-20
- Total features: 34
- Landmark index range: 0 to 477
- Expected unique mapped landmarks: 248
- Normalization phase after baseline lock: z-score then sigmoid scaling to [0,1]

---

## 2. Feature Groups (High-Level)

### AU / Expression Features
- au12_mean_amplitude
- au12_variance
- au12_activation_frequency
- au15_mean_amplitude
- au4_mean_activation
- au4_duration_ratio
- au1_au2_peak_intensity
- au20_activation_rate
- lip_compression_frequency
- overall_au_variance
- facial_emotional_range
- facial_transition_frequency
- near_zero_au_activation_ratio

### Head Motion Features
- mean_head_velocity
- head_velocity_peak
- head_motion_energy
- motion_energy_floor_score
- landmark_displacement_mean
- gesture_frequency
- posture_rigidity_index
- micro_motion_energy

### Eye / Gaze Features
- blink_rate
- blink_duration
- blink_cluster_density
- eye_contact_ratio
- downward_gaze_frequency
- gaze_shift_frequency
- baseline_eye_openness

### Temporal Response Features
- response_latency_mean
- speech_onset_delay
- nod_onset_latency
- pause_duration_mean
- extended_silence_ratio
- reaction_time_instability_index

---

## 3. Detailed Feature Meanings

| Feature | Group | Signal Type | Basic Computational Meaning | Behavioral Meaning | Human Interpretation |
|---|---|---|---|---|---|
| au12_mean_amplitude | au_expression | distance | Mean AU12 intensity over frames. | Average smile strength. | How strong the smile appears overall. |
| au12_variance | au_expression | variance | Variability of AU12 over time. | Smile stability vs fluctuation. | Whether smile expression changes a lot or stays flat. |
| au12_activation_frequency | au_expression | event_rate | Number of AU12 activations per time window. | How often smile events occur. | How frequently the person smiles. |
| au15_mean_amplitude | au_expression | distance | Mean activation of lip-corner depressor pattern. | Sad-mouth tendency. | Frequency/intensity of downward mouth expression. |
| au4_mean_activation | au_expression | distance | Mean brow-furrow activation intensity. | Tension/worry expression. | Strength of worried or frowning look. |
| au4_duration_ratio | au_expression | ratio | Fraction of time AU4 is active. | Persistence of brow tension. | Whether facial tension is brief or sustained. |
| au1_au2_peak_intensity | au_expression | peak | Peak intensity of inner+outer brow raise co-activation. | Acute alert/fear-like spike. | Maximum surprise/alert brow response. |
| au20_activation_rate | au_expression | event_rate | Frequency of horizontal lip stretch events. | Fear/tension mouth events. | How often tense lip stretch appears. |
| lip_compression_frequency | au_expression | event_rate | Frequency of lip compression events. | Emotional suppression pattern. | How often lips are pressed tightly. |
| overall_au_variance | au_expression | variance | Global variance across tracked AU proxies. | Overall facial dynamism. | Flat vs dynamic face behavior. |
| facial_emotional_range | au_expression | range | Max-minus-min span across expression intensities. | Emotional range breadth. | Narrow vs wide range of expressions. |
| facial_transition_frequency | au_expression | event_rate | Rate of expression state transitions. | Expression flexibility. | How often facial expressions shift. |
| near_zero_au_activation_ratio | au_expression | ratio | Fraction of frames with very low AU activity. | Flattened affect tendency. | How often face stays nearly expressionless. |
| mean_head_velocity | head_motion | velocity | Mean head movement speed over time. | General motor tempo. | Slow/sluggish vs energetic movement style. |
| head_velocity_peak | head_motion | peak | Maximum observed head speed. | Agitation spike marker. | Strongest sudden head movement in session. |
| head_motion_energy | head_motion | energy | Aggregate movement magnitude (energy-like). | Overall motor activation. | How physically active the head/face was. |
| motion_energy_floor_score | head_motion | floor | Time/score near minimal motion baseline. | Prolonged stillness tendency. | How much time is spent almost motionless. |
| landmark_displacement_mean | head_motion | mean_displacement | Mean frame-to-frame landmark displacement. | Subtle facial motion level. | Average amount of micro movement. |
| gesture_frequency | head_motion | event_rate | Frequency of detected movement gestures. | Conversational motor engagement. | How often head/face gesture events happen. |
| posture_rigidity_index | head_motion | index | Inverse variability of posture anchors. | Stiffness/rigidity marker. | How fixed or rigid posture appears. |
| micro_motion_energy | head_motion | energy | Energy of very small movement components. | Restlessness/fidgeting signature. | Tiny continuous movements indicating restlessness. |
| blink_rate | eye_gaze | event_rate | Blink events per window using EAR dynamics. | Autonomic/cognitive load proxy. | How often the person blinks. |
| blink_duration | eye_gaze | duration | Average blink closure duration. | Brief vs heavy blink tendency. | Whether blinks are short or prolonged. |
| blink_cluster_density | eye_gaze | density | Density of rapid blink bursts. | Overload/arousal burst pattern. | Rapid grouped blinking under stress or load. |
| eye_contact_ratio | eye_gaze | ratio | Proportion of time gaze aligns with frontal target. | Social gaze engagement. | Looking toward camera/person vs avoiding gaze. |
| downward_gaze_frequency | eye_gaze | event_rate | Frequency of downward gaze events. | Withdrawal/downcast tendency. | How often gaze drops downward. |
| gaze_shift_frequency | eye_gaze | event_rate | Rate of directional gaze shifts. | Scanning/hypervigilance tendency. | How frequently eyes move between targets. |
| baseline_eye_openness | eye_gaze | ear | Mean eye openness from EAR baseline dynamics. | Alertness/fatigue proxy. | Open alert eyes vs droopier fatigued eyes. |
| response_latency_mean | temporal_response | latency | Mean delay from prompt to response initiation. | Processing hesitation marker. | How long the person takes to start responding. |
| speech_onset_delay | temporal_response | latency | Delay from prompt to speech-related mouth onset. | Speech initiation speed. | Time before starting to talk. |
| nod_onset_latency | temporal_response | latency | Delay to first acknowledgment nod-like motion. | Social feedback delay. | How quickly acknowledgment movement appears. |
| pause_duration_mean | temporal_response | duration | Mean conversational pause duration. | Conversational tempo slowdown. | Typical silence length between responses. |
| extended_silence_ratio | temporal_response | ratio | Fraction of time in long pauses above threshold. | Severe slowing/withdrawal indicator. | How much of session has long silences. |
| reaction_time_instability_index | temporal_response | index | Variability of response timing over session. | Timing instability marker. | Consistency vs unpredictability in response speed. |

---

## 4. Feature To Region Mapping (Canonical)

The table below shows exactly which semantic regions each feature consumes.

| Feature | Regions Used |
|---|---|
| au12_mean_amplitude | left_lip_corner, right_lip_corner, upperlip_border, lowerlip_border |
| au12_variance | left_lip_corner, right_lip_corner |
| au12_activation_frequency | left_lip_corner, right_lip_corner |
| au15_mean_amplitude | left_lip_corner, right_lip_corner, lowerlip_border |
| au4_mean_activation | left_eyebrow, right_eyebrow, between_eyebrows |
| au4_duration_ratio | left_eyebrow, right_eyebrow |
| au1_au2_peak_intensity | left_eyebrow, right_eyebrow, forehead |
| au20_activation_rate | upperlip_border, lowerlip_border |
| lip_compression_frequency | upperlip_border, lowerlip_border, moustache |
| overall_au_variance | left_eye, right_eye, left_eyebrow, right_eyebrow, upperlip_border, lowerlip_border |
| facial_emotional_range | left_eye, right_eye, left_eyebrow, right_eyebrow, upperlip_border, lowerlip_border |
| facial_transition_frequency | all |
| near_zero_au_activation_ratio | all |
| mean_head_velocity | nose_tip, nose_bridge, chin |
| head_velocity_peak | nose_tip, chin |
| head_motion_energy | nose_tip, jawline_left, jawline_right |
| motion_energy_floor_score | all |
| landmark_displacement_mean | all |
| gesture_frequency | chin, jawline_left, jawline_right |
| posture_rigidity_index | nose_bridge, between_eyebrows, chin |
| micro_motion_energy | left_cheek, right_cheek, moustache |
| blink_rate | left_eye, right_eye |
| blink_duration | left_eye, right_eye |
| blink_cluster_density | left_eye, right_eye |
| eye_contact_ratio | left_iris, right_iris, nose_bridge |
| downward_gaze_frequency | left_iris, right_iris |
| gaze_shift_frequency | left_iris, right_iris |
| baseline_eye_openness | left_eye, right_eye |
| response_latency_mean | upperlip_border, lowerlip_border, chin |
| speech_onset_delay | lowerlip_border, upper_lip_intersection |
| nod_onset_latency | nose_tip, chin |
| pause_duration_mean | all |
| extended_silence_ratio | all |
| reaction_time_instability_index | all |

---

## 5. MediaPipe Landmark Region Map (Facial Region Map)

This project uses MediaPipe Face Mesh landmark indexing and groups landmarks into semantic regions for stable feature engineering.

### Core Region Dictionary

| Region | Landmark IDs |
|---|---|
| forehead | 10, 21, 54, 67, 69, 71, 103, 109, 151, 162, 251, 284, 297, 299, 301, 332, 338 |
| left_eyebrow | 46, 52, 53, 55, 63, 65, 66, 70, 105, 107 |
| right_eyebrow | 276, 282, 283, 285, 293, 295, 296, 300, 334, 336 |
| left_upper_eye_region | 27, 28 |
| right_upper_eye_region | 257, 258 |
| left_eye | 33, 246, 161, 160, 159, 158, 157, 173, 133, 155, 154, 153, 145, 144, 163, 7 |
| left_iris | 468 |
| right_iris | 473 |
| right_eye | 263, 466, 388, 387, 386, 385, 384, 398, 362, 382, 381, 380, 374, 373, 390, 249 |
| between_eyebrows | 8, 9 |
| nose_bridge | 1, 2, 5, 6, 195, 197 |
| nose_tip | 4 |
| nose_borders | 19, 49, 64, 94, 98, 99, 122, 125, 129, 168, 174, 196, 209, 217, 236, 279, 294, 327, 328, 351, 354, 358, 399, 419, 429, 437, 456 |
| left_nostril | 218, 219, 235, 237, 240, 241, 242 |
| right_nostril | 438, 439, 455, 457, 460, 461, 462 |
| left_eye_corner_to_border | 34, 35, 124, 127, 130, 143, 226 |
| right_eye_corner_to_border | 264, 265, 353, 356, 359, 372, 446 |
| jawline_left | 58, 136, 138, 150, 172, 215 |
| jawline_right | 288, 365, 367, 379, 397, 435 |
| left_cheek | 50, 100, 101, 111, 117, 121, 123, 137, 187, 192, 205 |
| right_cheek | 280, 329, 330, 340, 346, 350, 352, 366, 411, 416, 425 |
| moustache | 57, 92, 164, 165, 212, 216, 287, 322, 391, 432, 436 |
| upperlip_border | 0, 37, 39, 40, 185, 267, 269, 270, 409 |
| lowerlip_border | 17, 84, 91, 146, 181, 314, 321, 375, 405 |
| upper_lip_intersection | 11, 12, 13 |
| lower_lip_intersection | 14, 15, 16 |
| left_lip_corner | 61 |
| right_lip_corner | 291 |
| chin | 18, 152, 175, 182, 199, 200, 208, 406, 428 |
| right_lip_corner_to_border | 394, 422, 430 |
| left_lip_corner_to_border | 169, 202, 210 |
| low_importance_region | 23, 24, 93, 128, 132, 148, 149, 176, 222, 234, 245, 253, 254, 323, 357, 361, 377, 378, 389, 400, 442, 454, 465 |

### Eye EAR Landmark Contract

These landmarks are used for robust blink and openness metrics.

Left eye EAR points:

- p1_outer: 33
- p2_upper1: 160
- p3_upper2: 158
- p4_inner: 133
- p5_lower1: 153
- p6_lower2: 145

Right eye EAR points:

- p1_outer: 263
- p2_upper1: 385
- p3_upper2: 387
- p4_inner: 362
- p5_lower1: 380
- p6_lower2: 373

---

## 6. Practical Interpretation Notes

- Features in AU/Expression group indicate emotional expressivity patterns, not diagnostic labels by themselves.
- Head-motion features capture psychomotor tempo and movement variability.
- Eye/gaze features are sensitive to fatigue, attention, stress, and camera setup.
- Temporal response features require reliable prompt and speaking/nodding event timing.
- Any clinical interpretation must be contextual, longitudinal, and reviewed by qualified experts.

---

## 7. Usage Guidance For This Repo

- Keep feature order fixed to preserve CSV column consistency and model input order.
- If region mapping changes, re-validate feature contracts and retrain downstream models.
- If thresholds are tuned, document parameter updates in runtime docs and experiment logs.
- Treat this file as the human-readable companion to the machine-readable JSON contracts.

---

## 8. Quick Index: All 34 Features In Canonical Order

1. au12_mean_amplitude
2. au12_variance
3. au12_activation_frequency
4. au15_mean_amplitude
5. au4_mean_activation
6. au4_duration_ratio
7. au1_au2_peak_intensity
8. au20_activation_rate
9. lip_compression_frequency
10. overall_au_variance
11. facial_emotional_range
12. facial_transition_frequency
13. near_zero_au_activation_ratio
14. mean_head_velocity
15. head_velocity_peak
16. head_motion_energy
17. motion_energy_floor_score
18. landmark_displacement_mean
19. gesture_frequency
20. posture_rigidity_index
21. micro_motion_energy
22. blink_rate
23. blink_duration
24. blink_cluster_density
25. eye_contact_ratio
26. downward_gaze_frequency
27. gaze_shift_frequency
28. baseline_eye_openness
29. response_latency_mean
30. speech_onset_delay
31. nod_onset_latency
32. pause_duration_mean
33. extended_silence_ratio
34. reaction_time_instability_index
