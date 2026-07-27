#!/usr/bin/env python3
"""Generate the zero-valued Track-2 Kaggle submission template from test_pairs.csv."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path


GRID_HEIGHT, GRID_WIDTH = 36, 64
FRAME_IDS = range(0, 49, 8)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pairs", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    with args.pairs.open(newline="") as fh:
        pairs = list(csv.DictReader(fh))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    rows = 0
    with args.out.open("w", newline="", buffering=1024 * 1024) as fh:
        writer = csv.writer(fh, lineterminator="\n")
        writer.writerow(["id", "R", "G", "B"])
        for pair in pairs:
            variant, scene, seq_root = pair["video"].split("__", 2)
            prefix = f"{variant}-{scene}-{seq_root}_0"
            for frame_id in FRAME_IDS:
                for query_id in range(GRID_HEIGHT * GRID_WIDTH):
                    writer.writerow([f"{prefix}-q{query_id:04d}-f{frame_id:03d}", 0, 0, 0])
                    rows += 1
    print(f"[kaggle-nvs] wrote {rows:,} rows -> {args.out}")


if __name__ == "__main__":
    main()
