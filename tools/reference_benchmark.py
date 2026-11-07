#!/usr/bin/env python3
"""Frozen official-reference association benchmarks on the core's detection caches.

Validation/test identities are selected only through configs/splits.json. This
runner never trains or tunes thresholds. Provided native-bee boxes are an oracle
association diagnostic, not detector results. Cached replay is not live FPS.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from evaluate import evaluate_sequence, sha256
import run_reference

ROOT = Path(__file__).resolve().parents[1]
METRICS = ("HOTA", "AssA", "DetA", "IDF1", "MOTA")
COUNTS = ("IDSW", "Frag", "FP", "FN", "TP")
HIGH_THRESHOLD = 0.35
MAX_TRACK_AGE_SECONDS = 0.75


def config_for(algorithm: str, fps: float):
    # Upstream ByteTrack interprets track_buffer as a 30-FPS frame count, then
    # scales by source fps. OC-SORT directly counts update/observation indices.
    max_age = max(1, round(MAX_TRACK_AGE_SECONDS * (30 if algorithm == "bytetrack" else fps)))
    return {"algorithm": algorithm, "fps": fps, "track_thresh": HIGH_THRESHOLD,
            "max_age": max_age, "min_hits": 1, "iou_threshold": 0.1,
            "delta_t": 3, "inertia": 0.2, "ocsort_use_byte": False,
            "match_thresh": 0.8, "class_id": None}


def source_for(sequence: Path, metadata, source: str):
    if source == "onnx":
        path = ROOT / "runs/detector_cache" / metadata["name"] / "detections.csv"
        label = "cached ONNX detector predictions; inspect cache manifest for model and preprocessing"
        oracle = False
    else:
        key = "detections" if "detections" in metadata else "oracle_detections"
        if key not in metadata:
            raise ValueError(f"{sequence}: no supplied detections or oracle_detections in metadata")
        path = sequence / metadata[key]
        oracle = key == "oracle_detections"
        label = ("oracle GT boxes with identities removed; association-only diagnostic" if oracle else
                 "synthetic perturbed boxes; not neural detector results" if metadata.get("synthetic") else
                 "author/provided cached detections; provenance is in sequence metadata")
    if not path.is_file():
        raise FileNotFoundError(f"Missing {source} detections: {path}. Prepare/cache this input before reference evaluation.")
    return path, label, oracle


def run_one(sequence: Path, algorithm: str, output: Path, source: str, split_hash: str, force=False):
    metadata_path = sequence / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    fps = float(metadata["fps"])
    if not math.isfinite(fps) or fps <= 0:
        raise ValueError(f"{sequence}: invalid metadata fps")
    frames = int(metadata["num_frames"])
    detections, label, oracle = source_for(sequence, metadata, source)
    ground_truth = sequence / metadata["ground_truth"]
    frame_times = sequence / metadata["timestamp_file"]
    settings = config_for(algorithm, fps)
    output.mkdir(parents=True, exist_ok=True)
    fingerprint = {
        "sequence": metadata["name"], "sequence_path": str(sequence.resolve()),
        "algorithm": algorithm, "source": source, "source_description": label,
        "oracle_detections": oracle, "settings": settings,
        "detections_sha256": sha256(detections), "gt_sha256": sha256(ground_truth),
        "frame_times_sha256": sha256(frame_times), "metadata_sha256": sha256(metadata_path),
        "splits_sha256": split_hash,
        "reference_provenance_sha256": sha256(ROOT / "third_party/PROVENANCE.json"),
        "evaluator_sha256": sha256(ROOT / "tools/evaluate.py"),
        "reference_runner_sha256": sha256(ROOT / "tools/run_reference.py"),
        "benchmark_runner_sha256": sha256(Path(__file__)),
    }
    record_path = output / "experiment.json"
    if not force and record_path.is_file():
        previous = json.loads(record_path.read_text(encoding="utf-8"))
        if (all(previous.get(key) == value for key, value in fingerprint.items())
                and (output / "mot.txt").is_file() and (output / "metrics.json").is_file()
                and previous.get("upstream_manifest", {}).get("output_sha256") == sha256(output / "mot.txt")):
            return previous
    args = SimpleNamespace(**settings, detections=detections, output=output / "mot.txt",
                           manifest=output / "upstream.manifest.json", frames=frames, frame_times=frame_times)
    upstream = run_reference.run(args)
    metrics = evaluate_sequence(ground_truth, output / "mot.txt", frames)
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    record = {**fingerprint, "frames": frames, "metrics": metrics,
              "upstream_manifest": upstream, "sequence_metadata": metadata,
              "outputs": {"mot": str((output / "mot.txt").resolve()),
                          "metrics": str((output / "metrics.json").resolve()),
                          "manifest": str((output / "upstream.manifest.json").resolve())}}
    record_path.write_text(json.dumps(record, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return record


def aggregate(records):
    result = {}
    for algorithm in sorted({record["algorithm"] for record in records}):
        group = [record for record in records if record["algorithm"] == algorithm]
        frames = sum(record["frames"] for record in group)
        result[algorithm] = {
            "implementation": group[0]["upstream_manifest"]["implementation"],
            "sequences": len(group), "frames": frames,
            "detector_calls": frames, "detector_rate": 1.0,
            **{name: float(np.mean([record["metrics"][name] for record in group])) for name in METRICS},
            **{name: sum(record["metrics"][name] for record in group) for name in COUNTS},
            "aggregation": "unweighted sequence means for HOTA/AssA/DetA/IDF1/MOTA; NOT pooled official benchmark scores; counts summed",
            "detector_rate_definition": "all source observation indices use their available detection input; this is cached replay and does not execute a detector",
            "oracle_association_only": any(record["oracle_detections"] for record in group),
        }
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", choices=("synthetic", "native_bee", "gmot40"), default="synthetic")
    parser.add_argument("--split", choices=("validation", "test"), default="validation")
    parser.add_argument("--source", choices=("provided", "onnx"), default="provided")
    parser.add_argument("--output", type=Path, help="Defaults to runs/reference_<dataset>_<split>_<source>")
    parser.add_argument("--algorithms", nargs="+", choices=("ocsort", "bytetrack"), default=("ocsort", "bytetrack"))
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)
    splits_path = ROOT / "configs/splits.json"
    try:
        splits = json.loads(splits_path.read_text(encoding="utf-8"))
        if args.dataset not in splits or args.split not in splits[args.dataset]:
            raise ValueError(f"Dataset/split {args.dataset}/{args.split} has not been prepared in configs/splits.json")
        if not splits[args.dataset][args.split]:
            raise ValueError("Selected split contains no sequences")
        output = args.output or ROOT / "runs" / f"reference_{args.dataset}_{args.split}_{args.source}"
        split_hash = sha256(splits_path)
        records = []
        for relative in splits[args.dataset][args.split]:
            sequence = ROOT / relative
            for algorithm in dict.fromkeys(args.algorithms):
                record = run_one(sequence, algorithm, output / sequence.name / algorithm,
                                 args.source, split_hash, args.force)
                records.append(record)
                metrics = record["metrics"]
                print(f"{sequence.name:38} {algorithm:10} HOTA={metrics['HOTA']:.4f} IDF1={metrics['IDF1']:.4f} IDSW={metrics['IDSW']} source={record['source_description']}", flush=True)
        report = {
            "dataset": args.dataset, "split": args.split, "source": args.source,
            "split_policy": splits[args.dataset].get("policy"), "splits_sha256": split_hash,
            "metric_scale": "fractions; integer error counts",
            "configuration_policy": "fixed shared high threshold .35; settings declared in source before test; no per-test tuning",
            "threshold_semantics": "OC-SORT min_hits=1, IoU=.1, original direction weight=.2/lookback3; optional BYTE disabled. ByteTrack match=.8, original low=.1 and birth=high+.1 retained. Shared max track age nominally .75s converted to upstream frame-count APIs; neither reference consumes physical timestamp gaps.",
            "timing_scope": "cached detector-free association replay; no live FPS or end-to-end speedup claim",
            "input_matching": "exact same cache path and SHA-256 as the project's core benchmark; tracker-specific original output/threshold rules are retained",
            "limitations": ["Sparse annotated observations and processed bee ROI backgrounds limit deployment claims.",
                            "Native-bee provided GT boxes are explicitly oracle association diagnostics.",
                            "TrackEval Frag retains its documented upstream empty-prediction-frame limitation.",
                            "Sequence means are NOT pooled official benchmark scores."],
            "aggregate": aggregate(records),
            "per_sequence": [{"sequence": record["sequence"], "algorithm": record["algorithm"],
                              "frames": record["frames"], "source_description": record["source_description"],
                              "oracle_detections": record["oracle_detections"],
                              "detections_sha256": record["detections_sha256"],
                              "metrics": {name: record["metrics"][name] for name in METRICS + COUNTS},
                              "outputs": record["outputs"]} for record in records],
        }
        output.mkdir(parents=True, exist_ok=True)
        (output / "aggregate.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        with (output / "aggregate.csv").open("w", newline="", encoding="utf-8") as stream:
            fields = ("algorithm", "sequences", "frames", "detector_rate") + METRICS + COUNTS
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            for algorithm, values in report["aggregate"].items():
                writer.writerow({field: algorithm if field == "algorithm" else values[field] for field in fields})
    except (OSError, ValueError, ImportError) as error:
        parser.exit(2, f"error: {error}\n")
    print(json.dumps(report["aggregate"], indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
