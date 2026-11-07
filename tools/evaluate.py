#!/usr/bin/env python3
"""Evaluate a single, explicitly frame-aligned MOT sequence with pinned TrackEval.

Ground truth: frame,id,left,top,width,height,mark,class,visibility.
Predictions: frame,id,left,top,width,height,score,-1,-1,-1.
Coordinates are continuous pixel rectangles (no +1 pixel convention).
This adapter is for class-1 insect data, not the MOTChallenge pedestrian protocol:
it removes mark<=0 rows, applies no distractor suppression and never thresholds
prediction confidence. Every supplied valid prediction is evaluated.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import math
import os
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "third_party" / "TrackEval"))
from trackeval.metrics.hota import HOTA
from trackeval.metrics.clear import CLEAR
from trackeval.metrics.identity import Identity

TRACKEVAL_SHA = "12c8791b303e0a0b50f753af204249e622d0281a"


def guard_output_paths(inputs, outputs):
    """Reject canonical/symlink/hardlink collisions before any output is opened.

    Distinct existing generated files may be replaced intentionally. Input files
    and two aliases of the same output must never be overwritten by this run.
    """
    sources = [Path(path) for path in inputs if path is not None]
    destinations = [Path(path) for path in outputs]

    def equivalent(first, second):
        if os.path.normcase(str(first.resolve())) == os.path.normcase(str(second.resolve())):
            return True
        try:
            return first.samefile(second)
        except FileNotFoundError:
            return False

    for index, destination in enumerate(destinations):
        for source in sources:
            if equivalent(source, destination):
                raise ValueError(f"output would overwrite input: {destination}")
        for previous in destinations[:index]:
            if equivalent(previous, destination):
                raise ValueError(f"output paths refer to the same file: {destination}")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def integer(value: str, label: str) -> int:
    number = float(value)
    if not math.isfinite(number) or not number.is_integer():
        raise ValueError(f"{label} must be a finite integer: {value!r}")
    return int(number)


def read_mot(path: Path, frames: int, ground_truth: bool):
    """Read marked class-1 GT/all predictions and reject ambiguous input."""
    if frames <= 0:
        raise ValueError("frames must be positive, including trailing empty frames")
    rows = defaultdict(list)
    ignored = 0
    seen = set()
    with path.open(encoding="utf-8-sig") as stream:
        for line_number, line in enumerate(stream, 1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            fields = line.split(",") if "," in line else line.split()
            needed = 9 if ground_truth else 10
            context = f"{path}:{line_number}"
            if len(fields) < needed:
                raise ValueError(f"{context}: expected at least {needed} MOT columns")
            frame = integer(fields[0], f"{context} frame")
            identity = integer(fields[1], f"{context} id")
            if not 1 <= frame <= frames:
                raise ValueError(f"{context}: frame {frame} outside 1..{frames}")
            if ground_truth:
                mark = float(fields[6])
                if not math.isfinite(mark):
                    raise ValueError(f"{context}: mark is not finite")
                if mark <= 0:
                    ignored += 1
                    continue
                category = integer(fields[7], f"{context} class")
                if category != 1:
                    raise ValueError(f"{context}: marked class {category}; this single-class adapter requires class 1. Convert explicitly; classes are never silently dropped.")
            else:
                score = float(fields[6])
                if not math.isfinite(score):
                    raise ValueError(f"{context}: prediction score is not finite")
            if identity <= 0:
                raise ValueError(f"{context}: identity must be positive")
            key = (frame, identity)
            if key in seen:
                raise ValueError(f"{context}: duplicate frame/id {key}")
            seen.add(key)
            box = np.asarray([float(value) for value in fields[2:6]], dtype=float)
            if not np.isfinite(box).all() or box[2] <= 0 or box[3] <= 0:
                raise ValueError(f"{context}: boxes must be finite with positive width/height")
            rows[frame].append((identity, box))
    return rows, ignored


def box_iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Continuous xywh IoU, matching TrackEval bbox similarity semantics."""
    a = np.asarray(a, dtype=float).reshape((-1, 4))
    b = np.asarray(b, dtype=float).reshape((-1, 4))
    left_top = np.maximum(a[:, None, :2], b[None, :, :2])
    right_bottom = np.minimum(a[:, None, :2] + a[:, None, 2:],
                              b[None, :, :2] + b[None, :, 2:])
    wh = np.maximum(0.0, right_bottom - left_top)
    overlap = wh[..., 0] * wh[..., 1]
    union = a[:, 2, None] * a[:, 3, None] + b[None, :, 2] * b[None, :, 3] - overlap
    return np.divide(overlap, union, out=np.zeros_like(overlap), where=union > 0)


def prepare_data(gt_rows, prediction_rows, frames: int):
    gt_labels = sorted({identity for rows in gt_rows.values() for identity, _ in rows})
    pr_labels = sorted({identity for rows in prediction_rows.values() for identity, _ in rows})
    gt_map = {identity: index for index, identity in enumerate(gt_labels)}
    pr_map = {identity: index for index, identity in enumerate(pr_labels)}
    data = {"num_timesteps": frames, "num_gt_ids": len(gt_map),
            "num_tracker_ids": len(pr_map), "num_gt_dets": 0,
            "num_tracker_dets": 0, "gt_ids": [], "tracker_ids": [],
            "similarity_scores": []}
    for frame in range(1, frames + 1):
        ground = sorted(gt_rows.get(frame, []), key=lambda row: row[0])
        predictions = sorted(prediction_rows.get(frame, []), key=lambda row: row[0])
        data["gt_ids"].append(np.asarray([gt_map[row[0]] for row in ground], dtype=int))
        data["tracker_ids"].append(np.asarray([pr_map[row[0]] for row in predictions], dtype=int))
        data["num_gt_dets"] += len(ground)
        data["num_tracker_dets"] += len(predictions)
        data["similarity_scores"].append(box_iou([row[1] for row in ground], [row[1] for row in predictions]))
    return data


def evaluate_sequence(gt: Path, pred: Path, frames: int, iou_threshold: float = 0.5):
    if not 0 < iou_threshold <= 1:
        raise ValueError("IoU threshold must be in (0,1]")
    gt_rows, ignored = read_mot(gt, frames, True)
    prediction_rows, _ = read_mot(pred, frames, False)
    data = prepare_data(gt_rows, prediction_rows, frames)
    hota_metric = HOTA()
    hota = hota_metric.eval_sequence(data)
    clear = CLEAR({"THRESHOLD": iou_threshold, "PRINT_CONFIG": False}).eval_sequence(data)
    identity = Identity({"THRESHOLD": iou_threshold, "PRINT_CONFIG": False}).eval_sequence(data)
    result = {name: float(np.mean(hota[name])) for name in ("HOTA", "AssA", "DetA", "LocA")}
    result.update({name: float(identity[name]) for name in ("IDF1", "IDP", "IDR")})
    result["MOTA"] = float(clear["MOTA"])
    for output, field in (("IDSW", "IDSW"), ("Frag", "Frag"), ("FP", "CLR_FP"), ("FN", "CLR_FN"), ("TP", "CLR_TP")):
        result[output] = int(clear[field])
    result.update({"frames": frames, "gt_detections": data["num_gt_dets"],
                   "prediction_detections": data["num_tracker_dets"],
                   "gt_identities": data["num_gt_ids"], "predicted_identities": data["num_tracker_ids"]})
    result["hota_curves"] = {"iou_thresholds": hota_metric.array_labels.tolist(),
                             **{name: hota[name].tolist() for name in ("HOTA", "AssA", "DetA")}}
    result["metadata"] = {
        "implementation": "official TrackEval metric modules with a single-class MOT input adapter",
        "trackeval_commit": TRACKEVAL_SHA,
        "metric_scale": "fractions; MOTA can be negative; counts are integers",
        "hota_aggregation": "mean over official IoU thresholds 0.05 through 0.95",
        "clear_identity_iou_threshold": iou_threshold,
        "prediction_confidence_filter": "none; every supplied valid prediction is evaluated",
        "gt_preprocessing": "class 1 required; mark<=0 rows removed; no distractor/ignore-region suppression",
        "unmarked_gt_rows_removed": ignored,
        "gt_path": str(gt.resolve()), "pred_path": str(pred.resolve()),
        "gt_sha256": sha256(gt), "pred_sha256": sha256(pred),
        "limitations": ["Pinned upstream CLEAR Frag may undercount interruptions when an entire prediction frame is empty; metric code is preserved except removed NumPy scalar aliases.",
                        "This is not the MOTChallenge pedestrian distractor preprocessing protocol."]}
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--gt", type=Path, required=True)
    parser.add_argument("--pred", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--frames", type=int, required=True, help="Full source frame count, including empty/trailing frames")
    parser.add_argument("--iou-threshold", type=float, default=0.5, help="CLEAR/Identity threshold; HOTA uses its official threshold sweep")
    args = parser.parse_args(argv)
    try:
        guard_output_paths([args.gt, args.pred], [args.output])
        result = evaluate_sequence(args.gt, args.pred, args.frames, args.iou_threshold)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    except (OSError, ValueError) as error:
        parser.exit(2, f"error: {error}\n")
    print(json.dumps({key: result[key] for key in ("HOTA", "AssA", "IDF1", "IDSW", "FP", "FN")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
