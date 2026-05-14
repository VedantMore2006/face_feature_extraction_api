#!/usr/bin/env python3
"""Generate full synthetic dataset CSV with 34 features + condition label.

Output: `video_feature_synthetic_data.csv` at project root with 10003 rows
(6 classes x 1667 rows = 10002 data rows + 1 header) and 35 columns.

The script sources per-feature statistics from `reports/synthetic_data_validation.json`
and `reports/pattern_analysis.json` (fallback). For each condition it samples
from a Gaussian around the reported mean with a conservative std fallback.
"""
from __future__ import annotations

import json
import math
import random
from pathlib import Path
import csv
from typing import Dict, List

ROOT = Path(__file__).resolve().parents[1]
SYN_JSON = ROOT / "reports" / "synthetic_data_validation.json"
PAT_JSON = ROOT / "reports" / "pattern_analysis.json"
OUT_CSV = ROOT / "video_feature_synthetic_data.csv"

TARGET_LABELS = [
    ("depression", "depression"),
    ("normal", "normal"),
    ("stress", "stress"),
    ("anxiety", "anxiety"),
    ("bipolar", "bipolar"),
    ("suicidal_tendency", "suicidal_tendency"),
]

ROWS_PER_CLASS = 1667


def load_json(path: Path) -> Dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def extract_feature_stats(syn: Dict, pat: Dict) -> Dict[str, Dict[str, Dict[str, float]]]:
    # Build structure: feature -> condition_key -> {mean, std}
    stats: Dict[str, Dict[str, Dict[str, float]]] = {}

    # Prefer 'comparison' section in synthetic validation if available
    comp = syn.get("comparison", {}) if isinstance(syn, dict) else {}
    if comp:
        for feat, conds in comp.items():
            stats.setdefault(feat, {})
            for cond_key, vals in conds.items():
                try:
                    mean_v = float(vals.get("mean", 0.0))
                except Exception:
                    mean_v = 0.0
                try:
                    std_v = float(vals.get("std", 0.0))
                except Exception:
                    std_v = 0.0
                stats[feat][cond_key] = {"mean": mean_v, "std": std_v}

    # Fallback: use pattern_analysis.statistics.*.mean
    if not stats:
        stats = {}
        stat_block = pat.get("statistics", {}) if isinstance(pat, dict) else {}
        for cond_key, cond_vals in stat_block.items():
            mean_map = cond_vals.get("mean", {}) if isinstance(cond_vals, dict) else {}
            for feat, v in mean_map.items():
                stats.setdefault(feat, {})
                try:
                    stats[feat][cond_key] = {"mean": float(v), "std": 0.0}
                except Exception:
                    continue

    return stats


def ensure_feature_list(stats: Dict) -> List[str]:
    # Prefer explicit 34-feature listing from synthetic validation clinical_data
    # if present (use keys from a sample condition's "sample_features").
    try:
        syn_clinical = load_json(SYN_JSON).get("clinical_data", {})
        if isinstance(syn_clinical, dict) and syn_clinical:
            # pick first condition present
            first = next(iter(syn_clinical.values()))
            sample_feats = first.get("sample_features", {}) if isinstance(first, dict) else {}
            if sample_feats:
                feats_list = list(sample_feats.keys())
                return feats_list
    except Exception:
        pass

    feats = list(stats.keys())
    if not feats:
        raise RuntimeError("No feature statistics available to build dataset.")
    feats_sorted = sorted(feats)
    return feats_sorted


def resolve_condition_stats_for_feat(stats: Dict, feat: str, cond_key: str, available_conds: List[str]) -> Dict[str, float]:
    # If exact cond_key stat exists use it.
    if cond_key in stats.get(feat, {}):
        m = stats[feat][cond_key].get("mean", 0.0)
        s = stats[feat][cond_key].get("std", 0.0)
        return {"mean": float(m), "std": float(s)}

    # If 'normal' not available, compute average of other conditions
    vals = []
    for c in available_conds:
        entry = stats.get(feat, {}).get(c)
        if entry is None:
            continue
        vals.append(entry.get("mean", 0.0))
    if vals:
        avg = float(sum(vals) / len(vals))
        # std fallback: use std of vals or small fraction
        std_est = float(max(0.01 * abs(avg), 0.001))
        return {"mean": avg, "std": std_est}

    # Ultimate fallback
    return {"mean": 0.0, "std": 0.01}


def sample_value(mean: float, std: float) -> float:
    if std <= 0.0:
        # small relative jitter
        std = max(abs(mean) * 0.05, 0.001)
    val = random.gauss(mean, std)
    # constrain sensible range: disallow extreme negatives
    if val < 0.0:
        val = 0.0
    return float(val)


def main() -> None:
    syn = load_json(SYN_JSON)
    pat = load_json(PAT_JSON)
    stats = extract_feature_stats(syn, pat)
    feats = ensure_feature_list(stats)

    available_conds = set()
    for f in feats:
        available_conds.update(stats.get(f, {}).keys())
    available_conds = sorted(list(available_conds))

    rows: List[Dict[str, object]] = []

    for label_display, cond_key in TARGET_LABELS:
        for i in range(ROWS_PER_CLASS):
            row: Dict[str, object] = {}
            for feat in feats:
                cond_stats = resolve_condition_stats_for_feat(stats, feat, cond_key, available_conds)
                mean_v = cond_stats.get("mean", 0.0)
                std_v = cond_stats.get("std", 0.0)
                val = sample_value(mean_v, std_v)
                row[feat] = val
            row["condition_label"] = label_display
            rows.append(row)

    # Write CSV header: feats + condition_label
    header = feats + ["condition_label"]
    OUT_CSV.unlink(missing_ok=True)
    with OUT_CSV.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=header)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

    print(f"Wrote {len(rows)} rows to {OUT_CSV} with {len(header)} columns")


if __name__ == "__main__":
    main()
