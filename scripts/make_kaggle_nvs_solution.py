#!/usr/bin/env python3
"""Create the Kaggle metric ``solution.csv`` for the Syn4D NVS challenge.

Kaggle metric competitions score numeric CSV columns, not MP4 files directly.
This script reuses the companion tracking challenge's exact row ids, Usage split,
and valid mask, and replaces its 3D coordinates with target-view RGB values:

    id,sequence,valid,Usage,R,G,B

Rows remain one per (sequence, query pixel, scored frame).  RGB is sampled from
the target view (view 1 by default) at the query pixel supplied in queries.csv.
The output therefore has the same 2,097,152 rows and Public/Private partition
as the tracking solution.csv, while its numeric targets are suitable for a PSNR
metric.  The companion Kaggle scorer should resize/decode a participant MP4 to
the released 1280x720 canvas, then emit/predict R,G,B for these same ids.
"""
from __future__ import annotations

import argparse
import csv
import itertools
import json
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
from PIL import Image


def _load_queries(path: Path) -> Dict[str, Dict[int, Tuple[int, int]]]:
    queries: Dict[str, Dict[int, Tuple[int, int]]] = {}
    with path.open(newline="") as fh:
        for row in csv.DictReader(fh):
            sequence = row["sequence"]
            query_id = int(row["query_id"])
            # Released query coordinates are integer-valued, encoded as e.g. 689.000.
            queries.setdefault(sequence, {})[query_id] = (int(round(float(row["u"]))), int(round(float(row["v"]))))
    if not queries:
        raise ValueError(f"No query rows in {path}")
    return queries


def _parse_id(row_id: str) -> Tuple[int, int]:
    """Return (query_id, source-frame index) from ...-q000-f006."""
    try:
        query_part = row_id.rsplit("-q", 1)[1]
        query_id_s, frame_s = query_part.split("-f", 1)
        return int(query_id_s), int(frame_s)
    except (IndexError, ValueError) as exc:
        raise ValueError(f"Unexpected tracking row id: {row_id}") from exc


def _target_view_name(source_sequence: str, target_view: int) -> str:
    stem, separator, source_view = source_sequence.rpartition("_")
    if not separator or not source_view.isdigit():
        raise ValueError(f"Cannot parse source-view suffix from {source_sequence}")
    return f"{stem}_{target_view}"


class TargetFrameCache:
    def __init__(self, dataset_root: Path, sequence: str, target_view: int) -> None:
        try:
            variant, scene, source_sequence = sequence.split("/", 2)
        except ValueError as exc:
            raise ValueError(f"Expected sequence variant/scene/seq_view, got {sequence}") from exc
        self.target_sequence = _target_view_name(source_sequence, target_view)
        self.png_dir = dataset_root / variant / scene / "png" / self.target_sequence
        if not self.png_dir.is_dir():
            raise FileNotFoundError(f"Missing target-view PNG directory: {self.png_dir}")
    def load_frame(self, frame_id: int) -> np.ndarray:
        """Decode exactly one target frame; holding all 32 PNGs is needlessly large."""
        path = self.png_dir / f"{self.target_sequence}_{frame_id:04d}.png"
        if not path.is_file():
            raise FileNotFoundError(f"Missing target frame: {path}")
        with Image.open(path) as source:
            return np.asarray(source.convert("RGB"), dtype=np.uint8).copy()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tracking-solution", type=Path, required=True,
                    help="Existing tracking solution.csv; supplies id, sequence, valid, Usage.")
    ap.add_argument("--queries", type=Path, required=True,
                    help="Tracking challenge data/queries.csv.")
    ap.add_argument("--dataset-root", type=Path, required=True,
                    help="Syn4D kaggle_eval root containing <variant>/<scene>/png.")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--target-view", type=int, default=1)
    ap.add_argument("--skip-sequences", type=int, default=0,
                    help="Skip this many sequence groups; use with --append for resumable batches.")
    ap.add_argument("--limit-sequences", type=int, help="write only the first N sequences (smoke test)")
    ap.add_argument("--append", action="store_true",
                    help="Append rows to an existing CSV without writing another header.")
    ap.add_argument("--finalize", action="store_true",
                    help="Write output metadata after this (normally final) batch.")
    args = ap.parse_args()

    if args.skip_sequences < 0:
        raise SystemExit("--skip-sequences must be non-negative")
    if args.append and not args.out.is_file():
        raise SystemExit(f"--append requires an existing output: {args.out}")
    if not args.append and args.out.exists():
        raise SystemExit(f"Refusing to overwrite existing output: {args.out}")
    queries = _load_queries(args.queries)
    args.out.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = ["id", "sequence", "valid", "Usage", "R", "G", "B"]
    rows = 0
    sequence_count = 0
    seen_sequences = set()
    mode = "a" if args.append else "w"
    with args.tracking_solution.open(newline="") as source, args.out.open(mode, newline="") as destination:
        reader = csv.DictReader(source)
        required = {"id", "sequence", "valid", "Usage"}
        missing = required.difference(reader.fieldnames or ())
        if missing:
            raise ValueError(f"Tracking solution missing required columns: {sorted(missing)}")
        writer = csv.DictWriter(destination, fieldnames=fieldnames)
        if not args.append:
            writer.writeheader()
        source_sequence_index = 0
        for sequence, group in itertools.groupby(reader, key=lambda row: row["sequence"]):
            source_sequence_index += 1
            if sequence in seen_sequences:
                raise ValueError(f"Tracking solution is not grouped by sequence: {sequence} appears twice")
            seen_sequences.add(sequence)
            if source_sequence_index <= args.skip_sequences:
                continue
            if args.limit_sequences is not None and sequence_count >= args.limit_sequences:
                break
            sequence_count += 1
            if sequence not in queries:
                raise KeyError(f"No queries for tracking sequence {sequence}")
            cache = TargetFrameCache(args.dataset_root, sequence, args.target_view)
            print(f"[{sequence_count}] {sequence} -> {cache.target_sequence}")

            # Kaggle joins on id, so row order is immaterial. Grouping output by
            # frame lets us decode a 1280x720 target PNG once, rather than
            # caching 32 full-resolution PNGs (or decoding each one 512 times).
            rows_by_frame: Dict[int, list[tuple[dict, int, int]]] = {}
            for row in group:
                query_id, frame_id = _parse_id(row["id"])
                try:
                    u, v = queries[sequence][query_id]
                except KeyError as exc:
                    raise KeyError(f"No query {query_id} for {sequence}") from exc
                rows_by_frame.setdefault(frame_id, []).append((row, u, v))

            for frame_id in sorted(rows_by_frame):
                image = cache.load_frame(frame_id)
                for row, u, v in rows_by_frame[frame_id]:
                    if not (0 <= v < image.shape[0] and 0 <= u < image.shape[1]):
                        raise ValueError(
                            f"Pixel ({u}, {v}) outside {image.shape[1]}x{image.shape[0]} for {cache.target_sequence}"
                        )
                    rgb = image[v, u]
                    writer.writerow({
                        "id": row["id"],
                        "sequence": sequence,
                        "valid": row["valid"],
                        "Usage": row["Usage"],
                        "R": int(rgb[0]),
                        "G": int(rgb[1]),
                        "B": int(rgb[2]),
                    })
                    rows += 1

    if args.finalize:
        with args.out.open(newline="") as fh:
            total_rows = sum(1 for _ in fh) - 1
        meta = {
            "rows": total_rows,
            "sequences_total": source_sequence_index,
            "sequences_written_this_batch": sequence_count,
            "target_view": args.target_view,
            "tracking_solution": str(args.tracking_solution),
            "queries": str(args.queries),
            "dataset_root": str(args.dataset_root),
            "columns": fieldnames,
            "rgb_canvas": [1280, 720],
        }
        args.out.with_suffix(args.out.suffix + ".meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(f"[kaggle-nvs] wrote {rows:,} rows across {sequence_count} sequences -> {args.out}")


if __name__ == "__main__":
    main()
