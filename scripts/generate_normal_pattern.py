#!/usr/bin/env python3
"""Generate a synthetic 'normal' session profile from existing pattern statistics.

Reads `reports/pattern_analysis.json`, averages available condition means to
produce a 'normal' mean feature vector and samples synthetic session vectors.
Writes a CSV to `reports/synthetic_normal_samples.csv` and prints a short summary.
"""
from __future__ import annotations

import json
import math
import random
from pathlib import Path
import csv
from typing import Dict, List

ROOT = Path(__file__).resolve().parents[1]
PATTERN_PATH = ROOT / "reports" / "pattern_analysis.json"
OUT_CSV = ROOT / "reports" / "synthetic_normal_samples.csv"


def load_patterns(path: Path) -> Dict:
    return json.loads(path.read_text(encoding="utf-8"))


def build_normal_profile(patterns: Dict) -> Dict[str, float]:
    stats = patterns.get("statistics", {})
    conds = [k for k in stats.keys() if k != "normal"]
    mean_acc: Dict[str, List[float]] = {}

    for c in conds:
        c_mean = stats.get(c, {}).get("mean", {})
        for k, v in c_mean.items():
            try:
                fv = float(v)
            except Exception:
                continue
            mean_acc.setdefault(k, []).append(fv)

    normal_mean: Dict[str, float] = {}
    for k, vals in mean_acc.items():
        if not vals:
            continue
        normal_mean[k] = float(sum(vals) / len(vals))

    return normal_mean


def sample_synthetic_rows(mean_profile: Dict[str, float], n: int = 50) -> List[Dict[str, float]]:
    rows: List[Dict[str, float]] = []
    for _ in range(n):
        row: Dict[str, float] = {}
        for k, m in mean_profile.items():
            # small relative noise: 5% of absolute value or a tiny floor
            scale = max(abs(m) * 0.05, 0.001)
            row[k] = float(random.gauss(m, scale))
        rows.append(row)
    return rows


def write_csv(rows: List[Dict[str, float]], out_path: Path) -> None:
    if not rows:
        return
    keys = sorted(rows[0].keys())
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=keys)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r.get(k, "") for k in keys})


def main() -> None:
    if not PATTERN_PATH.exists():
        raise FileNotFoundError(f"Missing {PATTERN_PATH}")

    patterns = load_patterns(PATTERN_PATH)
    normal_mean = build_normal_profile(patterns)
    rows = sample_synthetic_rows(normal_mean, n=1667)
    write_csv(rows, OUT_CSV)
    print(f"Wrote {len(rows)} synthetic normal samples to {OUT_CSV}")


if __name__ == "__main__":
    main()
