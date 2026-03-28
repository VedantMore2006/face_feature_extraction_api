"""Region mapping and runtime region extraction utilities.

This module freezes the region-to-landmark contract and provides validated
helpers to transform full-frame landmarks into semantic facial regions.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, MutableMapping, Sequence, Tuple

MIN_LANDMARK_ID = 0
MAX_LANDMARK_ID = 477
EXPECTED_UNIQUE_LANDMARKS = 248

# Canonical region mapping (frozen contract v1).
FACIAL_REGIONS: Dict[str, List[int]] = {
    "forehead": [10, 21, 54, 67, 69, 71, 103, 109, 151, 162, 251, 284, 297, 299, 301, 332, 338],
    "left_eyebrow": [46, 52, 53, 55, 63, 65, 66, 70, 105, 107],
    "right_eyebrow": [276, 282, 283, 285, 293, 295, 296, 300, 334, 336],
    "left_upper_eye_region": [27, 28],
    "right_upper_eye_region": [257, 258],
    "left_eye": [33, 246, 161, 160, 159, 158, 157, 173, 133, 155, 154, 153, 145, 144, 163, 7],
    "left_iris": [468],
    "right_iris": [473],
    "right_eye": [263, 466, 388, 387, 386, 385, 384, 398, 362, 382, 381, 380, 374, 373, 390, 249],
    "between_eyebrows": [8, 9],
    "nose_bridge": [1, 2, 5, 6, 195, 197],
    "nose_tip": [4],
    "nose_borders": [19, 49, 64, 94, 98, 99, 122, 125, 129, 168, 174, 196, 209, 217, 236, 279, 294, 327, 328, 351, 354, 358, 399, 419, 429, 437, 456],
    "left_nostril": [218, 219, 235, 237, 240, 241, 242],
    "right_nostril": [438, 439, 455, 457, 460, 461, 462],
    "left_eye_corner_to_border": [34, 35, 124, 127, 130, 143, 226],
    "right_eye_corner_to_border": [264, 265, 353, 356, 359, 372, 446],
    "jawline_left": [58, 136, 138, 150, 172, 215],
    "jawline_right": [288, 365, 367, 379, 397, 435],
    "left_cheek": [50, 100, 101, 111, 117, 121, 123, 137, 187, 192, 205],
    "right_cheek": [280, 329, 330, 340, 346, 350, 352, 366, 411, 416, 425],
    "moustache": [57, 92, 164, 165, 212, 216, 287, 322, 391, 432, 436],
    "upperlip_border": [0, 37, 39, 40, 185, 267, 269, 270, 409],
    "lowerlip_border": [17, 84, 91, 146, 181, 314, 321, 375, 405],
    "upper_lip_intersection": [11, 12, 13],
    "lower_lip_intersection": [14, 15, 16],
    "left_lip_corner": [61],
    "right_lip_corner": [291],
    "chin": [18, 152, 175, 182, 199, 200, 208, 406, 428],
    "right_lip_corner_to_border": [394, 422, 430],
    "left_lip_corner_to_border": [169, 202, 210],
    "low_importance_region": [23, 24, 93, 128, 132, 148, 149, 176, 222, 234, 245, 253, 254, 323, 357, 361, 377, 378, 389, 400, 442, 454, 465],
}

LEFT_EYE_STRUCTURE: Dict[str, object] = {
    "anchors": {
        "inner_corner": [133],
        "outer_corner": [33],
        "center": [468],
    },
    "contour_full": [33, 246, 161, 160, 159, 158, 157, 173, 133, 155, 154, 153, 145, 144, 163, 7],
    "ear_points": {
        "p1_outer": 33,
        "p2_upper1": 160,
        "p3_upper2": 158,
        "p4_inner": 133,
        "p5_lower1": 153,
        "p6_lower2": 145,
    },
    "corners": [33, 133],
}

RIGHT_EYE_STRUCTURE: Dict[str, object] = {
    "anchors": {
        "inner_corner": [362],
        "outer_corner": [263],
        "center": [473],
    },
    "contour_full": [263, 466, 388, 387, 386, 385, 384, 398, 362, 382, 381, 380, 374, 373, 390, 249],
    "ear_points": {
        "p1_outer": 263,
        "p2_upper1": 385,
        "p3_upper2": 387,
        "p4_inner": 362,
        "p5_lower1": 380,
        "p6_lower2": 373,
    },
    "corners": [362, 263],
}

LEFT_EYE_EAR: Dict[str, int] = dict(LEFT_EYE_STRUCTURE["ear_points"])
RIGHT_EYE_EAR: Dict[str, int] = dict(RIGHT_EYE_STRUCTURE["ear_points"])


LandmarkPoint = Tuple[float, float, float]


class RegionMappingError(ValueError):
    """Raised when region definitions or extraction inputs are invalid."""


@dataclass(frozen=True)
class RegionValidationReport:
    region_count: int
    total_assigned: int
    unique_assigned: int
    duplicate_count: int
    out_of_range_count: int


def _validate_region_name(name: str) -> None:
    if not isinstance(name, str) or not name.strip():
        raise RegionMappingError("Region name must be a non-empty string.")


def _validate_region_ids(name: str, ids: Sequence[int]) -> None:
    if not isinstance(ids, Sequence) or len(ids) == 0:
        raise RegionMappingError(f"Region '{name}' must contain at least one landmark id.")

    for value in ids:
        if not isinstance(value, int):
            raise RegionMappingError(f"Region '{name}' includes non-integer landmark id: {value!r}")
        if value < MIN_LANDMARK_ID or value > MAX_LANDMARK_ID:
            raise RegionMappingError(
                f"Region '{name}' includes out-of-range landmark id {value}; "
                f"expected {MIN_LANDMARK_ID}..{MAX_LANDMARK_ID}."
            )

    if len(set(ids)) != len(ids):
        raise RegionMappingError(f"Region '{name}' contains duplicate landmark ids.")


def validate_region_mapping(mapping: Mapping[str, Sequence[int]]) -> RegionValidationReport:
    """Validate a region mapping contract and return summary statistics."""
    if not mapping:
        raise RegionMappingError("Region mapping is empty.")

    all_ids: List[int] = []
    for name, ids in mapping.items():
        _validate_region_name(name)
        _validate_region_ids(name, ids)
        all_ids.extend(ids)

    unique_ids = set(all_ids)
    duplicate_count = len(all_ids) - len(unique_ids)

    out_of_range_count = sum(
        1
        for idx in unique_ids
        if idx < MIN_LANDMARK_ID or idx > MAX_LANDMARK_ID
    )

    report = RegionValidationReport(
        region_count=len(mapping),
        total_assigned=len(all_ids),
        unique_assigned=len(unique_ids),
        duplicate_count=duplicate_count,
        out_of_range_count=out_of_range_count,
    )

    if report.unique_assigned != EXPECTED_UNIQUE_LANDMARKS:
        raise RegionMappingError(
            "Unique landmark count mismatch. "
            f"Expected {EXPECTED_UNIQUE_LANDMARKS}, got {report.unique_assigned}."
        )

    if report.duplicate_count != 0:
        raise RegionMappingError(
            "Cross-region duplicates detected. "
            f"Duplicate assignments: {report.duplicate_count}."
        )

    if report.out_of_range_count != 0:
        raise RegionMappingError(
            "Out-of-range landmark ids detected in region mapping."
        )

    return report


def validate_ear_points(left_eye_ear: Mapping[str, int], right_eye_ear: Mapping[str, int]) -> None:
    """Validate EAR point contracts for both eyes."""
    required = {"p1_outer", "p2_upper1", "p3_upper2", "p4_inner", "p5_lower1", "p6_lower2"}

    for eye_name, eye_map in (("left_eye_ear", left_eye_ear), ("right_eye_ear", right_eye_ear)):
        missing = sorted(required.difference(eye_map.keys()))
        if missing:
            raise RegionMappingError(f"{eye_name} missing keys: {missing}")

        for point_name in required:
            point_idx = eye_map[point_name]
            if not isinstance(point_idx, int):
                raise RegionMappingError(
                    f"{eye_name}.{point_name} must be int, got {type(point_idx).__name__}."
                )
            if point_idx < MIN_LANDMARK_ID or point_idx > MAX_LANDMARK_ID:
                raise RegionMappingError(
                    f"{eye_name}.{point_name} out of range: {point_idx}; "
                    f"expected {MIN_LANDMARK_ID}..{MAX_LANDMARK_ID}."
                )


def validate_runtime_contract() -> RegionValidationReport:
    """Validate all region-related runtime contracts."""
    report = validate_region_mapping(FACIAL_REGIONS)
    validate_ear_points(LEFT_EYE_EAR, RIGHT_EYE_EAR)
    return report


def get_region_points(
    landmarks: Sequence[LandmarkPoint],
    region_name: str,
    *,
    strict: bool = True,
) -> List[LandmarkPoint]:
    """Return point tuples for one semantic region.

    Args:
        landmarks: Full-frame landmark sequence of (x, y, z) points.
        region_name: Region key from FACIAL_REGIONS.
        strict: If True, raise on missing/invalid ids. If False, skip invalid ids.
    """
    if region_name not in FACIAL_REGIONS:
        raise RegionMappingError(
            f"Unknown region '{region_name}'. Available: {sorted(FACIAL_REGIONS.keys())}"
        )

    if not isinstance(landmarks, Sequence) or len(landmarks) == 0:
        raise RegionMappingError("Landmarks must be a non-empty sequence.")

    selected: List[LandmarkPoint] = []
    for idx in FACIAL_REGIONS[region_name]:
        if idx >= len(landmarks):
            if strict:
                raise RegionMappingError(
                    f"Landmark index {idx} for region '{region_name}' is out of bounds for "
                    f"landmark sequence length {len(landmarks)}."
                )
            continue

        point = landmarks[idx]
        if not isinstance(point, Sequence) or len(point) != 3:
            if strict:
                raise RegionMappingError(
                    f"Invalid landmark tuple at index {idx}; expected (x, y, z), got {point!r}."
                )
            continue

        selected.append((float(point[0]), float(point[1]), float(point[2])))

    if strict and not selected:
        raise RegionMappingError(f"Region '{region_name}' has no valid points in this frame.")

    return selected


def extract_regions(
    landmarks: Sequence[LandmarkPoint],
    region_names: Iterable[str] | None = None,
    *,
    strict: bool = True,
) -> Dict[str, List[LandmarkPoint]]:
    """Extract multiple regions from frame landmarks."""
    names = list(region_names) if region_names is not None else list(FACIAL_REGIONS.keys())
    if not names:
        raise RegionMappingError("region_names cannot be empty.")

    regions: Dict[str, List[LandmarkPoint]] = {}
    for name in names:
        regions[name] = get_region_points(landmarks, name, strict=strict)
    return regions


def flatten_region_ids(mapping: Mapping[str, Sequence[int]] | None = None) -> List[int]:
    """Return sorted list of unique ids present in region mapping."""
    source = mapping if mapping is not None else FACIAL_REGIONS
    unique = {idx for ids in source.values() for idx in ids}
    return sorted(unique)


# Fail fast at import time so broken mapping is detected before a session starts.
validate_runtime_contract()
