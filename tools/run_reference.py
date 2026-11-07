#!/usr/bin/env python3
"""Run pinned official OC-SORT or ByteTrack association on cached detections.

This is detector-free replay, not an end-to-end speed benchmark. Every source
observation index is processed, including frames with no cached detections.
Upstream references use a fixed Kalman step per update, not physical timestamps.
"""
from __future__ import annotations

import argparse
import csv
import importlib.metadata
import json
import math
from pathlib import Path
import sys
import time
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluate import guard_output_paths, integer, sha256

SHAS = {"ocsort": "8462e7e729a93ccd3bd995c0a79a890336cb3a0b",
        "bytetrack": "d1bf0191adff59bc8fcfeaa0b33d3d1642552a99"}


def read_detections(path: Path, frames: int, class_id=None):
    by_frame = {}
    classes = set()
    timestamps = {}
    with path.open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        required = {"frame", "timestamp", "x", "y", "w", "h", "score", "class_id"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f"{path}: missing cached detection columns {sorted(required)}")
        for number, row in enumerate(reader, 2):
            frame = integer(row["frame"], f"{path}:{number} frame")
            category = integer(row["class_id"], f"{path}:{number} class_id")
            values = [float(row[key]) for key in ("timestamp", "x", "y", "w", "h", "score")]
            if not all(math.isfinite(value) for value in values):
                raise ValueError(f"{path}:{number}: nonfinite value")
            timestamp, x, y, width, height, score = values
            if not 1 <= frame <= frames or width <= 0 or height <= 0 or not 0 <= score <= 1 or timestamp < 0:
                raise ValueError(f"{path}:{number}: invalid frame, geometry, timestamp or confidence")
            if frame in timestamps and not math.isclose(timestamps[frame], timestamp, abs_tol=1e-9):
                raise ValueError(f"{path}:{number}: inconsistent timestamps within frame")
            timestamps[frame] = timestamp
            classes.add(category)
            if class_id is None or category == class_id:
                by_frame.setdefault(frame, []).append([x, y, x + width, y + height, score])
    if class_id is None and len(classes) > 1:
        raise ValueError("Reference runner is single-class: pass --class-id explicitly for a multi-class cache")
    for first, second in zip(sorted(timestamps), sorted(timestamps)[1:]):
        if timestamps[second] <= timestamps[first]:
            raise ValueError("Cached detection frame timestamps must increase strictly")
    return by_frame, sorted(classes)


def frame_time_metadata(path: Path | None, frames: int, fps: float):
    result = {"provided": path is not None, "irregular_source_sampling": None}
    if path is None:
        return result
    with path.open(newline="", encoding="utf-8-sig") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != frames:
        raise ValueError("--frame-times must contain exactly --frames data rows")
    times = []
    for frame, row in enumerate(rows, 1):
        if integer(row["frame"], "frame-times frame") != frame:
            raise ValueError("frame-times must enumerate every observation index in order")
        timestamp = float(row["timestamp"])
        if not math.isfinite(timestamp) or (times and timestamp <= times[-1]):
            raise ValueError("frame-times timestamps must be finite and strictly increasing")
        times.append(timestamp)
    deltas = np.diff(times)
    result.update({"path": str(path.resolve()), "sha256": sha256(path),
                   "irregular_source_sampling": bool(np.any(np.abs(deltas - 1 / fps) > 1e-6)),
                   "minimum_source_delta_seconds": float(deltas.min()) if len(deltas) else None,
                   "maximum_source_delta_seconds": float(deltas.max()) if len(deltas) else None})
    return result


def make_tracker(args):
    if args.algorithm == "ocsort":
        sys.path.insert(0, str(ROOT / "third_party" / "OC_SORT"))
        from trackers.ocsort_tracker.ocsort import OCSort
        return OCSort(det_thresh=args.track_thresh, max_age=args.max_age,
                      min_hits=args.min_hits, iou_threshold=args.iou_threshold,
                      delta_t=args.delta_t, inertia=args.inertia, use_byte=args.ocsort_use_byte)
    sys.path.insert(0, str(ROOT / "third_party" / "ByteTrack"))
    from yolox.tracker.basetrack import BaseTrack
    from yolox.tracker.byte_tracker import BYTETracker
    BaseTrack._count = 0
    config = SimpleNamespace(track_thresh=args.track_thresh, track_buffer=args.max_age,
                             match_thresh=args.match_thresh, mot20=False)
    return BYTETracker(config, frame_rate=args.fps)


def run(args):
    if args.frames <= 0 or args.fps <= 0 or not math.isfinite(args.fps):
        raise ValueError("frames and fps must be positive")
    if not 0.1 < args.track_thresh < 0.9:
        raise ValueError("track-thresh must be in (0.1,0.9), preserving upstream low-score range and ByteTrack birth threshold")
    if args.min_hits <= 0 or args.max_age <= 0 or args.delta_t <= 0:
        raise ValueError("min-hits, max-age and delta-t must be positive")
    if not 0 < args.iou_threshold <= 1 or not 0 < args.match_thresh <= 1:
        raise ValueError("IoU and matching thresholds must be in (0,1]")
    manifest_path = args.manifest or args.output.with_suffix(".manifest.json")
    guard_output_paths([args.detections, args.frame_times], [args.output, manifest_path])
    by_frame, classes = read_detections(args.detections, args.frames, args.class_id)
    time_metadata = frame_time_metadata(args.frame_times, args.frames, args.fps)
    tracker = make_tracker(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    measurements = []
    records = 0
    with args.output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        for frame in range(1, args.frames + 1):
            detections = np.asarray(by_frame.get(frame, []), dtype=float).reshape((-1, 5))
            started = time.perf_counter()
            # Input cache already uses original pixels; identical sizes keep the
            # upstream rescale factor equal to one without reading the images.
            tracks = tracker.update(detections, (1, 1), (1, 1))
            measurements.append(time.perf_counter() - started)
            if args.algorithm == "ocsort":
                rows = [(track[4], track[:4], 1.0) for track in tracks]
            else:
                rows = [(track.track_id, track.tlbr, float(track.score)) for track in tracks]
            for track_id, box, confidence in rows:
                x1, y1, x2, y2 = map(float, box)
                if not np.isfinite([x1, y1, x2, y2, confidence]).all() or x2 <= x1 or y2 <= y1:
                    raise ValueError(f"Upstream {args.algorithm} returned invalid geometry at frame {frame}")
                writer.writerow([frame, int(track_id), f"{x1:.8f}", f"{y1:.8f}",
                                 f"{x2 - x1:.8f}", f"{y2 - y1:.8f}", f"{confidence:.8f}", -1, -1, -1])
                records += 1
    packages = {}
    for name in ("numpy", "scipy", "filterpy", "lap"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    manifest = {
        "algorithm": args.algorithm,
        "implementation": "official OC-SORT tracking-only subset" if args.algorithm == "ocsort" else "official ByteTrack with compatibility adapters",
        "upstream_commit": SHAS[args.algorithm],
        "provenance_file": "third_party/PROVENANCE.json",
        "detections": str(args.detections.resolve()), "detections_sha256": sha256(args.detections),
        "output": str(args.output.resolve()), "output_sha256": sha256(args.output),
        "frames": args.frames, "prediction_records": records,
        "input_classes": classes, "selected_class_id": args.class_id,
        "settings": {key: value for key, value in vars(args).items() if not isinstance(value, Path)},
        "packages": packages, "source_frame_times": time_metadata,
        "time_model": "upstream fixed step: one update per available observation index, including empty detections; source timestamp gaps are not passed into its Kalman model",
        "output_semantics": "upstream active-track output only; no locally invented interpolation or lost-track boxes",
        "score_semantics": "OC-SORT output has no confidence; MOT score 1 is a placeholder, not calibrated probability" if args.algorithm == "ocsort" else "last upstream detection score",
        "timing_scope": "cached detection tracker.update calls only; excludes detector, decoding, I/O and imports; not an end-to-end performance claim",
        "replay_update_ms_median": float(np.median(measurements) * 1000),
        "replay_update_ms_p95": float(np.percentile(measurements, 95) * 1000),
        "replay_update_seconds_total": float(sum(measurements)),
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--algorithm", choices=("ocsort", "bytetrack"), required=True)
    parser.add_argument("--detections", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--frames", type=int, required=True)
    parser.add_argument("--frame-times", type=Path, help="Optional complete frame/timestamp CSV; records irregular-sampling caveats")
    parser.add_argument("--fps", type=float, default=30)
    parser.add_argument("--class-id", type=int, help="Explicit class selection; multi-class caches otherwise rejected")
    parser.add_argument("--track-thresh", type=float, default=0.3)
    parser.add_argument("--max-age", type=int, default=30, help="OC-SORT max_age / ByteTrack track_buffer (upstream scales ByteTrack buffer by fps/30)")
    parser.add_argument("--min-hits", type=int, default=3, help="OC-SORT only")
    parser.add_argument("--iou-threshold", type=float, default=0.3, help="OC-SORT only")
    parser.add_argument("--delta-t", type=int, default=3, help="OC-SORT observation direction lookback")
    parser.add_argument("--inertia", type=float, default=0.2, help="OC-SORT direction consistency weight")
    parser.add_argument("--ocsort-use-byte", action="store_true", help="Enable upstream optional low-score association; disabled by default")
    parser.add_argument("--match-thresh", type=float, default=0.8, help="ByteTrack high-score matching cost threshold")
    args = parser.parse_args(argv)
    try:
        manifest = run(args)
    except (OSError, ValueError, ImportError) as error:
        parser.exit(2, f"error: {error}\nInstall requirements-eval.txt in the project virtual environment if a dependency is missing.\n")
    print(json.dumps({key: manifest[key] for key in ("algorithm", "frames", "prediction_records", "implementation")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
