# Graph Report - .  (2026-04-25)

## Corpus Check
- Corpus is ~32,584 words - fits in a single context window. You may not need a graph.

## Summary
- 311 nodes · 625 edges · 27 communities detected
- Extraction: 74% EXTRACTED · 26% INFERRED · 0% AMBIGUOUS · INFERRED: 164 edges (avg confidence: 0.6)
- Token cost: 18,500 input · 4,200 output

## Community Hubs (Navigation)
- [[_COMMUNITY_Core Feature Extraction Engine|Core Feature Extraction Engine]]
- [[_COMMUNITY_Mental Health Behavioral Profiles|Mental Health Behavioral Profiles]]
- [[_COMMUNITY_API Contracts and Auth|API Contracts and Auth]]
- [[_COMMUNITY_Multimodal Live API Client|Multimodal Live API Client]]
- [[_COMMUNITY_Video-to-Feature Pipeline|Video-to-Feature Pipeline]]
- [[_COMMUNITY_API Services and Deployment|API Services and Deployment]]
- [[_COMMUNITY_Configuration and Validation|Configuration and Validation]]
- [[_COMMUNITY_Feature Computation Math|Feature Computation Math]]
- [[_COMMUNITY_Facial Region Mapping|Facial Region Mapping]]
- [[_COMMUNITY_Baseline Normalization|Baseline Normalization]]
- [[_COMMUNITY_API Submission Compliance|API Submission Compliance]]
- [[_COMMUNITY_Session Vector Export|Session Vector Export]]
- [[_COMMUNITY_Runtime Type Definitions|Runtime Type Definitions]]
- [[_COMMUNITY_Package Init|Package Init]]
- [[_COMMUNITY_FrameSource Abstraction|FrameSource Abstraction]]
- [[_COMMUNITY_Privacy by Design|Privacy by Design]]
- [[_COMMUNITY_Landmark Types|Landmark Types]]
- [[_COMMUNITY_pandas|pandas]]
- [[_COMMUNITY_joblib|joblib]]
- [[_COMMUNITY_scikit-learn|scikit-learn]]
- [[_COMMUNITY_scipy|scipy]]
- [[_COMMUNITY_pydantic|pydantic]]
- [[_COMMUNITY_multipart|multipart]]
- [[_COMMUNITY_requests|requests]]
- [[_COMMUNITY_Event-Based Latency|Event-Based Latency]]
- [[_COMMUNITY_SAD Condition|SAD Condition]]
- [[_COMMUNITY_Melancholic Condition|Melancholic Condition]]

## God Nodes (most connected - your core abstractions)
1. `RuntimeConfig` - 26 edges
2. `run_raw_extraction_pipeline()` - 24 edges
3. `extract_from_video()` - 19 edges
4. `FeatureEngine` - 19 edges
5. `FrameData` - 19 edges
6. `LandmarkStream` - 18 edges
7. `BaselineNormalizer` - 18 edges
8. `RegionMappingError` - 17 edges
9. `FeatureEngineError` - 16 edges
10. `FeatureCsvLogger` - 15 edges

## Surprising Connections (you probably didn't know these)
- `process_face() (live_api_client copy)` --semantically_similar_to--> `process_face_extraction_only()`  [INFERRED] [semantically similar]
  extra/live_api_client copy.py → /home/vedant/face_feature_extraction_api/extra/video_to_feature_live.py
- `extra/main.py main()` --calls--> `run_raw_extraction_pipeline()`  [EXTRACTED]
  extra/main.py → /home/vedant/face_feature_extraction_api/src/pipeline.py
- `validate_runtime_contract()` --references--> `LEFT_EYE_EAR constant`  [EXTRACTED]
  /home/vedant/face_feature_extraction_api/src/regions.py → src/regions.py
- `validate_runtime_contract()` --references--> `RIGHT_EYE_EAR constant`  [EXTRACTED]
  /home/vedant/face_feature_extraction_api/src/regions.py → src/regions.py
- `FastAPI API Key Authentication Implementation Guide` --references--> `python-dotenv==1.1.1`  [EXTRACTED]
  docs/FASTAPI_AUTH_IMPLEMENTATION.md → requirements.txt

## Hyperedges (group relationships)
- **End-to-End Video Inference Pipeline** — landmark_stream_landmarkstream, feature_engine_featureengine, predict_video_risk_extract_raw_timeseries, predict_video_risk_engineer_session_vector, predict_video_risk_main [EXTRACTED 0.95]
- **API Video Extraction and Session Storage Flow** — extraction_api_extract_from_video, extraction_api_save_session_outputs, extraction_api_get_session_vector [EXTRACTED 0.95]
- **Runtime Contract and Configuration Validation** — config_validate_runtime_config, regions_validate_runtime_contract, contracts_load_feature_contract [EXTRACTED 0.92]
- **End-to-End Extraction Pipeline: MediaPipe → Features → Baseline → Vector → API** — requirements_mediapipe, readme_realtime_pipeline, readme_psb_normalization, features_34_canonical, readme_extraction_api [EXTRACTED 0.95]
- **API Authentication Security Triad: load_dotenv + _load_api_key + require_api_key** — fastapi_auth_load_api_key, fastapi_auth_require_api_key, requirements_dotenv [EXTRACTED 1.00]
- **Behavioral Screening System: Features + Conditions + AI Reports** — features_34_canonical, condition_behaviour_doc, ai_report_20260323_030822, features_screening_intent [INFERRED 0.85]

## Communities

### Community 0 - "Core Feature Extraction Engine"
Cohesion: 0.09
Nodes (47): RuntimeConfig, CsvLoggerError, FeatureCsvLogger, CSV output logger for normalized feature vectors., Raised when CSV output cannot be initialized or written., FeatureEngine, FeatureEngineConfig, FeatureEngineError (+39 more)

### Community 1 - "Mental Health Behavioral Profiles"
Cohesion: 0.08
Nodes (36): Behavioral AI Screening Report 2026-03-23 03:08 (qwen3.5), Behavioral AI Screening Report 2026-03-23 03:42 (qwen3.5), Primary Screening: Bipolar Disorder (57% confidence), Condition and Behaviour Reference Document, Depression in Bipolar Disorder Behavioral Profile, Burnout Behavioral Profile, Chronic Stress Behavioral Profile, Cyclothymic Disorder Behavioral Profile (+28 more)

### Community 2 - "API Contracts and Auth"
Cohesion: 0.13
Nodes (24): ContractError, FeatureContract, load_feature_contract(), Runtime contract loaders for frozen docs definitions.  These helpers keep JSON c, Raised when a project contract file is missing or malformed., _read_json(), custom_openapi(), _expected_schema() (+16 more)

### Community 3 - "Multimodal Live API Client"
Cohesion: 0.15
Nodes (28): _debug_terminal(), fuse_multimodal(), LiveAPIError, _log_request(), _optional_env(), _pick_feature_vector(), _preview_mapping(), process_face() (+20 more)

### Community 4 - "Video-to-Feature Pipeline"
Cohesion: 0.13
Nodes (17): app.py re-export of FastAPI app, FastAPI extraction_api app, get_session_vector(), fuse_multimodal(), process_face() (live_api_client copy), LiveAPIError, _log_api_call(), process_face() (+9 more)

### Community 5 - "API Services and Deployment"
Cohesion: 0.13
Nodes (21): API User Guide (Two-Service Architecture), Hostinger VPS Docker Deployment, Extraction API Service (port 8010), Scoring API Service (port 8011), End-to-End Test Pipeline Script (api/test_pipeline.py), Rationale: Split extraction and scoring into two services, Dockerfile and docker-compose.yml Complete Guide, Docker HEALTHCHECK via /health endpoint (+13 more)

### Community 6 - "Configuration and Validation"
Cohesion: 0.14
Nodes (14): FeatureThresholdConfig, Configuration values for the facial analysis runtime., validate_runtime_config(), extra/main.py bootstrap(), extra/main.py main(), bootstrap(), main(), parse_args() (+6 more)

### Community 7 - "Feature Computation Math"
Cohesion: 0.21
Nodes (14): _all_displacements(), _centroid(), FeatureEngine.compute(), _distance(), _distance_xy(), _ear(), _event_rate(), _eye_horizontal_span() (+6 more)

### Community 8 - "Facial Region Mapping"
Cohesion: 0.2
Nodes (17): extract_regions(), FACIAL_REGIONS landmark map, get_region_points(), Region mapping and runtime region extraction utilities.  This module freezes the, Validate a region mapping contract and return summary statistics., Validate EAR point contracts for both eyes., Validate all region-related runtime contracts., Return point tuples for one semantic region.      Args:         landmarks: Full- (+9 more)

### Community 9 - "Baseline Normalization"
Cohesion: 0.19
Nodes (7): BaselineStats, _clip01(), _mean(), Baseline collection and normalization utilities.  Pipeline normalization logic:, Normalize raw features into [0,1] with 8 decimal precision., _sigmoid(), stats()

### Community 10 - "API Submission Compliance"
Cohesion: 0.17
Nodes (12): API Formatting Compliance Report, Environment Variables Naming Convention, example.env File Requirement, output.json Requirement, payload.json Requirement, README File Requirement, run.py Ready-to-Run Script Requirement, API Submission Requirements (Saraswati College) (+4 more)

### Community 11 - "Session Vector Export"
Cohesion: 0.52
Nodes (6): _find_latest_session_file(), _load_session_payload(), main(), parse_args(), _process_one(), _write_vector_file()

### Community 12 - "Runtime Type Definitions"
Cohesion: 0.67
Nodes (2): PipelineStatus, Common runtime data structures shared by pipeline modules.

### Community 13 - "Package Init"
Cohesion: 1.0
Nodes (1): Core runtime package for the facial analysis pipeline.

### Community 14 - "FrameSource Abstraction"
Cohesion: 1.0
Nodes (2): Flexible Frame Source Architecture, Rationale: FrameSource abstraction for source-agnostic pipeline

### Community 15 - "Privacy by Design"
Cohesion: 1.0
Nodes (2): Privacy-by-Design Architecture, Rationale: No raw landmark or video storage for privacy compliance

### Community 17 - "Landmark Types"
Cohesion: 1.0
Nodes (1): LandmarkPoint type alias

### Community 18 - "pandas"
Cohesion: 1.0
Nodes (1): pandas==2.3.3

### Community 19 - "joblib"
Cohesion: 1.0
Nodes (1): joblib==1.5.3

### Community 20 - "scikit-learn"
Cohesion: 1.0
Nodes (1): scikit-learn==1.7.2

### Community 21 - "scipy"
Cohesion: 1.0
Nodes (1): scipy==1.15.3

### Community 22 - "pydantic"
Cohesion: 1.0
Nodes (1): pydantic==2.12.5

### Community 23 - "multipart"
Cohesion: 1.0
Nodes (1): python-multipart==0.0.22

### Community 24 - "requests"
Cohesion: 1.0
Nodes (1): requests==2.32.5

### Community 25 - "Event-Based Latency"
Cohesion: 1.0
Nodes (1): Event-Based Response Latency Detection

### Community 26 - "SAD Condition"
Cohesion: 1.0
Nodes (1): Seasonal Affective Disorder (SAD) Behavioral Profile

### Community 27 - "Melancholic Condition"
Cohesion: 1.0
Nodes (1): Melancholic Depression Behavioral Profile

## Knowledge Gaps
- **81 isolated node(s):** `Core runtime package for the facial analysis pipeline.`, `Configuration values for the facial analysis runtime.`, `Runtime contract loaders for frozen docs definitions.  These helpers keep JSON c`, `Raised when a project contract file is missing or malformed.`, `CSV output logger for normalized feature vectors.` (+76 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **Thin community `Runtime Type Definitions`** (3 nodes): `PipelineStatus`, `Common runtime data structures shared by pipeline modules.`, `runtime_types.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Package Init`** (2 nodes): `Core runtime package for the facial analysis pipeline.`, `__init__.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `FrameSource Abstraction`** (2 nodes): `Flexible Frame Source Architecture`, `Rationale: FrameSource abstraction for source-agnostic pipeline`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Privacy by Design`** (2 nodes): `Privacy-by-Design Architecture`, `Rationale: No raw landmark or video storage for privacy compliance`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Landmark Types`** (1 nodes): `LandmarkPoint type alias`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `pandas`** (1 nodes): `pandas==2.3.3`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `joblib`** (1 nodes): `joblib==1.5.3`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `scikit-learn`** (1 nodes): `scikit-learn==1.7.2`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `scipy`** (1 nodes): `scipy==1.15.3`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `pydantic`** (1 nodes): `pydantic==2.12.5`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `multipart`** (1 nodes): `python-multipart==0.0.22`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `requests`** (1 nodes): `requests==2.32.5`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Event-Based Latency`** (1 nodes): `Event-Based Response Latency Detection`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `SAD Condition`** (1 nodes): `Seasonal Affective Disorder (SAD) Behavioral Profile`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Melancholic Condition`** (1 nodes): `Melancholic Depression Behavioral Profile`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `extract_from_video()` connect `API Contracts and Auth` to `Core Feature Extraction Engine`, `Facial Region Mapping`, `Video-to-Feature Pipeline`, `Configuration and Validation`?**
  _High betweenness centrality (0.121) - this node is a cross-community bridge._
- **Why does `LiveAPIError` connect `Multimodal Live API Client` to `Core Feature Extraction Engine`?**
  _High betweenness centrality (0.115) - this node is a cross-community bridge._
- **Why does `RuntimeConfig` connect `Core Feature Extraction Engine` to `API Contracts and Auth`, `Configuration and Validation`?**
  _High betweenness centrality (0.088) - this node is a cross-community bridge._
- **Are the 21 inferred relationships involving `RuntimeConfig` (e.g. with `Validate API key from X-API-Key header.` and `Add security scheme to OpenAPI spec for Swagger UI display.`) actually correct?**
  _`RuntimeConfig` has 21 INFERRED edges - model-reasoned connections that need verification._
- **Are the 7 inferred relationships involving `run_raw_extraction_pipeline()` (e.g. with `FeatureEngineConfig` and `.output_path()`) actually correct?**
  _`run_raw_extraction_pipeline()` has 7 INFERRED edges - model-reasoned connections that need verification._
- **Are the 6 inferred relationships involving `extract_from_video()` (e.g. with `RuntimeConfig` and `ValueError`) actually correct?**
  _`extract_from_video()` has 6 INFERRED edges - model-reasoned connections that need verification._
- **Are the 9 inferred relationships involving `FeatureEngine` (e.g. with `FrameData` and `PipelineError`) actually correct?**
  _`FeatureEngine` has 9 INFERRED edges - model-reasoned connections that need verification._