#!/usr/bin/env python3
"""Validation selection and predeclared refinement comparisons on cached inputs.

Earlier test clips are reused exploratory evidence. Additional confirmation clips
cannot be scored by this runner before a validation-only selection is frozen.
"""
from __future__ import annotations
import argparse
import csv
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import subprocess

import numpy as np
from benchmark import environment, discovery
from evaluate import evaluate_sequence, sha256

ROOT = Path(__file__).resolve().parents[1]
SCORES = ("HOTA", "IDF1", "AssA", "DetA", "MOTA")
COUNTS = ("IDSW", "FP", "FN", "TP", "Frag")
RUN_OUTPUT_FILES = ("mot.txt", "telemetry.csv", "frames.csv", "detections.csv", "summary.json", "metrics.json", "experiment.json")
_IMAGE_INTEGRITY_CACHE = {}


def ordered_image_files(sequence, metadata):
    project = ROOT.resolve()
    directory = Path(sequence) / metadata["images"]
    resolved = directory.resolve()
    if not resolved.is_relative_to(project) or directory.is_symlink():
        raise ValueError("Image directory must remain inside the project and may not be a symlink")
    files = sorted(directory.iterdir(), key=lambda item: item.name)
    for path in files:
        if path.is_symlink() or not path.resolve().is_relative_to(project) or not path.is_file():
            raise ValueError("Image source contains a non-file, symlink or escaped path")
        if path.suffix.lower() not in (".jpg", ".jpeg", ".png"):
            raise ValueError(f"Unexpected non-image source file: {path.name}")
    if len(files) != int(metadata["num_frames"]):
        raise ValueError("Actual image file count differs from metadata num_frames")
    return files


def verify_image_content(sequence, metadata):
    """Stream original ordered bytes; memo is invalidated by file stat changes."""
    declared = metadata.get("image_content_sha256", "").lower()
    if len(declared) != 64 or any(character not in "0123456789abcdef" for character in declared):
        raise ValueError("Metadata needs a declared image_content_sha256")
    paths = ordered_image_files(sequence, metadata)
    signature = tuple((path.name, path.stat().st_size, path.stat().st_mtime_ns,
                       path.stat().st_ctime_ns, path.stat().st_ino) for path in paths)
    key = (str(Path(sequence).resolve()), metadata["images"], declared, signature)
    if key not in _IMAGE_INTEGRITY_CACHE:
        import hashlib
        digest = hashlib.sha256()
        for path in paths:
            with path.open("rb") as stream:
                for block in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(block)
        if digest.hexdigest() != declared:
            raise ValueError("Actual ordered image bytes differ from metadata image_content_sha256")
        _IMAGE_INTEGRITY_CACHE[key] = {"sha256": digest.hexdigest(), "num_frames": len(paths),
                                       "ordered_files": [path.name for path in paths],
                                       "verification": "SHA-256 of actual ordered source bytes; no image decoding"}
    return dict(_IMAGE_INTEGRITY_CACHE[key])


def guard_file_writes(write_paths, input_paths):
    """Reject resolved-path and existing hard-link aliases before any writes."""
    protected = [Path(path).resolve() for path in input_paths]
    for output in map(Path, write_paths):
        target = output.resolve()
        for source in protected:
            same_name = os.path.normcase(str(target)) == os.path.normcase(str(source))
            same_file = target.exists() and source.exists() and target.samefile(source)
            if same_name or same_file:
                raise ValueError(f"Output would overwrite protected input: {output} -> {source}")


def guard_output_paths(directory, input_paths):
    guard_file_writes([Path(directory) / name for name in RUN_OUTPUT_FILES], input_paths)


def protected_case_paths(case, include_images=False):
    sequence = ROOT / case["path"]
    meta_path = sequence / "metadata.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    source = case.get("source", "provided")
    detection = (ROOT / "runs/detector_cache" / meta["name"] / "detections.csv" if source == "onnx" else
                 sequence / meta["oracle_detections"] if source == "oracle" else
                 sequence / meta.get("detections", meta.get("oracle_detections", "detections.csv")))
    paths = [meta_path, detection, sequence / meta["ground_truth"], sequence / meta["timestamp_file"]]
    if include_images:
        paths.extend(ordered_image_files(sequence, meta))
    return paths


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def case_key(case):
    path = Path(case["path"])
    return "__".join(path.parts[1:])


def case_inputs(case):
    sequence = ROOT / case["path"]
    meta_path = sequence / "metadata.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    source = case.get("source", "provided")
    if source == "onnx":
        detections = ROOT / "runs/detector_cache" / meta["name"] / "detections.csv"
    elif source == "oracle":
        detections = sequence / meta["oracle_detections"]
    else:
        detections = sequence / meta.get("detections", meta.get("oracle_detections", "detections.csv"))
    gt = sequence / meta["ground_truth"]
    times = sequence / meta["timestamp_file"]
    fingerprint = {"metadata_sha256": sha256(meta_path), "detections_sha256": sha256(detections),
                   "gt_sha256": sha256(gt), "frame_times_sha256": sha256(times),
                   "image_content_sha256": meta.get("image_content_sha256")}
    return sequence, meta, detections, gt, times, fingerprint


def command_for(protocol, profile_name, schedule, output_policy, sequence, meta, detections, times, output):
    profile = protocol["profiles"][profile_name]
    executable = ROOT / protocol["executables"][profile["executable"]]
    if os.name != "nt" and executable.suffix == ".exe":
        executable = executable.with_suffix("")
    if not executable.is_file():
        raise FileNotFoundError(f"Executable not built/copied: {executable}")
    knobs = {**protocol["shared_tracker"], **profile["args"], **schedule, "output_policy": output_policy}
    if profile["executable"] == "baseline_v2":
        if knobs.get("scene_method", "legacy") != "legacy" or knobs.get("risk_scope", "all") != "all":
            raise ValueError("Frozen baseline_v2 cannot represent robust scene or active risk scope")
        knobs = {key: value for key, value in knobs.items()
                 if not key.startswith("motion_") and key not in ("scene_method", "risk_scope")}
    command = [str(executable), "--detections", str(detections), "--frame-times", str(times),
               "--frames", str(meta["num_frames"]), "--fps", str(meta["fps"]), "--output", str(output)]
    if knobs.get("schedule") == "adaptive":
        command.extend(["--images", str(sequence / meta["images"])])
    for key, value in knobs.items():
        command.extend(["--" + key.replace("_", "-"), str(value)])
    return command, knobs


def schedule_diagnostics(frame_log, knobs):
    with frame_log.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    threshold = float(knobs.get("risk_threshold", 0.18))
    novelty_threshold = float(knobs.get("novelty_threshold", 0.00008))
    selected = 0
    for index, row in enumerate(rows, 1):
        if int(row["frame"]) != index:
            raise ValueError("Nonconsecutive output frame indices")
        selected += int(row["detector_ran"])
        if knobs["schedule"] in ("adaptive", "risk", "covariance"):
            limit = math.floor(1 + float(knobs.get("target_rate", 1)) * (index - 1) + 1e-9)
            if selected > limit:
                raise ValueError(f"Hard detector-call prefix budget violated at frame {index}")
    result = {"frames": len(rows), "calls": selected,
              "uncertainty_threshold_exceeded_fraction": float(np.mean([float(row["uncertainty"]) >= threshold for row in rows])),
              "ambiguity_threshold_exceeded_fraction": float(np.mean([float(row["ambiguity"]) >= threshold for row in rows])),
              "novelty_threshold_exceeded_fraction": float(np.mean([float(row["novelty"]) >= novelty_threshold for row in rows]))}
    for column in ("scene_global_change", "scene_reset", "scene_warming"):
        if rows and column in rows[0]:
            result[column + "_mean"] = float(np.mean([float(row[column]) for row in rows]))
    return result


def run_one(protocol, case, profile_name, schedule_name, schedule, output_policy, output, phase, force=False, protocol_path=None):
    sequence, meta, detections, gt, times, inputs = case_inputs(case)
    command, knobs = command_for(protocol, profile_name, schedule, output_policy,
                                 sequence, meta, detections, times, output)
    protected = protected_case_paths(case, schedule.get("schedule") == "adaptive") + [Path(command[0])]
    if protocol_path is not None:
        protected.append(protocol_path)
    guard_output_paths(output, protected)
    output.mkdir(parents=True, exist_ok=True)
    fingerprint = {"case": case, "profile": profile_name, "schedule_name": schedule_name,
                   "knobs": knobs, "command": command, "phase": phase,
                   "output_policy": output_policy, "inputs": inputs,
                   "executable_sha256": sha256(Path(command[0])),
                   "evaluator_sha256": sha256(ROOT / "tools/evaluate.py"),
                   "runner_sha256": sha256(Path(__file__))}
    record_path = output / "experiment.json"
    if not force and record_path.is_file():
        previous = json.loads(record_path.read_text(encoding="utf-8"))
        if (all(previous.get(key) == value for key, value in fingerprint.items())
                and (output / "mot.txt").is_file()
                and previous.get("mot_sha256") == sha256(output / "mot.txt")):
            return previous
    process = subprocess.run(command, env=environment(), capture_output=True, text=True)
    if process.returncode:
        raise RuntimeError(f"{case_key(case)}/{profile_name}: {process.stderr or process.stdout}")
    summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    if summary["frames"] != meta["num_frames"]:
        raise ValueError("Executable processed a different number of observations from the declared case")
    metrics = evaluate_sequence(gt, output / "mot.txt", meta["num_frames"])
    found = discovery(gt, detections, output / "frames.csv", meta["num_frames"], times)
    found["interpretation"] = "Detector-box availability at score>=.08/IoU>=.5; not high-confidence track initialization. First annotated visibility, not verified physical birth."
    diagnostics = schedule_diagnostics(output / "frames.csv", knobs)
    record = {**fingerprint, "metadata": meta, "metrics": metrics, "summary": summary,
              "discovery": found, "schedule_diagnostics": diagnostics,
              "mot_sha256": sha256(output / "mot.txt"), "output": str(output.resolve()),
              "evidence_label": "additional predeclared confirmation" if phase == "confirmation" else
                                "reused prior test: exploratory" if phase == "exploratory" else
                                "oracle/processed-data diagnostic" if phase == "diagnostic" else "validation development"}
    write_json(output / "metrics.json", metrics)
    write_json(record_path, record)
    print(f"{phase:12} {case_key(case):40} {profile_name:24} {schedule_name:28} {output_policy:6} HOTA={metrics['HOTA']:.4f} IDF1={metrics['IDF1']:.4f} calls={summary['detector_rate']:.4f}", flush=True)
    return record


def summarize(records):
    groups = {}
    for record in records:
        key = "/".join((record["output_policy"], record["profile"], record["schedule_name"]))
        groups.setdefault(key, []).append(record)
    result = {}
    for key, group in groups.items():
        frames = sum(row["summary"]["frames"] for row in group)
        calls = sum(row["summary"]["detector_calls"] for row in group)
        result[key] = {"sequences": len(group), "frames": frames, "detector_calls": calls,
                       "detector_rate": calls / frames,
                       **{name: float(np.mean([row["metrics"][name] for row in group])) for name in SCORES},
                       **{name: sum(row["metrics"][name] for row in group) for name in COUNTS},
                       **{name: sum(row["discovery"][name] for row in group)
                          for name in ("targets", "missed_targets", "short_targets", "missed_short_targets")},
                       "aggregation": "Unweighted sequence means, not official pooled scores; error/discovery counts summed, call rate frame-weighted."}
    return result


def select_validation(protocol, records, schedule_name="every"):
    eligible = [name for name, profile in protocol["profiles"].items() if profile.get("selection_eligible")]
    primary = protocol["primary_output_policy"]
    selected_records = [row for row in records if row["output_policy"] == primary and row["schedule_name"] == schedule_name]
    outcomes = {}
    for profile in eligible:
        per_group = {}
        for group_name, group in protocol["validation_groups"].items():
            required = {case_key(case) for case in group["cases"]}
            found = [row for row in selected_records if row["profile"] == profile and case_key(row["case"]) in required]
            if {case_key(row["case"]) for row in found} != required:
                raise ValueError(f"Cannot select from incomplete validation: {profile}/{group_name}")
            per_group[group_name] = {name: float(np.mean([row["metrics"][name] for row in found])) for name in SCORES}
        weighted = {metric: sum(protocol["validation_groups"][group]["weight"] * value[metric]
                                for group, value in per_group.items()) for metric in SCORES}
        outcomes[profile] = {"groups": per_group, "source_balanced": weighted}
    guard_group = protocol["selection"]["raw_guard_group"]
    baseline = protocol["selection"]["reference_profile"]
    minimum = outcomes[baseline]["groups"][guard_group]["HOTA"] - protocol["selection"]["max_raw_hota_drop"]
    allowed = [name for name in eligible if outcomes[name]["groups"][guard_group]["HOTA"] + 1e-12 >= minimum]
    winner = max(allowed, key=lambda name: (outcomes[name]["source_balanced"]["HOTA"],
                                          outcomes[name]["source_balanced"]["IDF1"], name == baseline))
    regressions = []
    old = {case_key(row["case"]): row for row in selected_records if row["profile"] == "baseline_v2"}
    new = {case_key(row["case"]): row for row in selected_records if row["profile"] == "two_stage"}
    for key in sorted(set(old) & set(new)):
        if old[key]["inputs"] != new[key]["inputs"]:
            raise ValueError("Frozen baseline/current control did not use identical case inputs")
        regressions.append({"case": key, "mot_bytes_identical": old[key]["mot_sha256"] == new[key]["mot_sha256"],
                            "deltas_new_minus_baseline": {metric: new[key]["metrics"][metric] - old[key]["metrics"][metric] for metric in SCORES + COUNTS}})
    return {"selected_profile": winner, "selection": {**protocol["selection"], "schedule": schedule_name}, "outcomes": outcomes,
            "raw_hota_minimum": minimum, "raw_guard_accepted_profiles": allowed,
            "baseline_v2_regression_checks": regressions, "frozen": False}


def freeze(protocol, protocol_path, output):
    selection_path = output / "validation/selection.json"
    protected = [protocol_path, selection_path, Path(__file__), ROOT / "tools/evaluate.py"]
    for group in protocol["validation_groups"].values():
        for case in group["cases"]:
            protected.extend(protected_case_paths(case, True))
    for case in protocol["confirmation_cases"]:
        protected.extend(protected_case_paths(case, True))
    protected.extend(ROOT / path for path in protocol["executables"].values())
    recommendation_path = ROOT / "configs/recommended_profile.json"
    if recommendation_path.exists():
        protected.append(recommendation_path)
    guard_file_writes([output / "freeze.json"], protected)
    if not selection_path.is_file():
        raise ValueError("Run complete validation before freezing")
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    if selection["protocol_sha256"] != sha256(protocol_path):
        raise ValueError("Protocol changed after validation; rerun validation")
    if selection["runner_sha256"] != sha256(Path(__file__)) or selection["evaluator_sha256"] != sha256(ROOT / "tools/evaluate.py"):
        raise ValueError("Runner/evaluator changed after validation; rerun validation")
    for group in protocol["validation_groups"].values():
        for case in group["cases"]:
            if case_inputs(case)[-1] != selection["validation_inputs"][case_key(case)]:
                raise ValueError("Validation inputs changed after selection; rerun validation")
    frozen_path = output / "freeze.json"
    if frozen_path.exists():
        raise ValueError("A freeze already exists; use a new --output for a new predeclared study rather than silently replacing it")
    confirmation_inputs = {case_key(case): case_inputs(case)[-1] for case in protocol["confirmation_cases"]}
    image_integrity = {}
    for case in [case for group in protocol["validation_groups"].values() for case in group["cases"]] + protocol["confirmation_cases"]:
        sequence, metadata = case_inputs(case)[:2]
        image_integrity[case_key(case)] = verify_image_content(sequence, metadata)
    binaries = {name: sha256(ROOT / path) for name, path in protocol["executables"].items()}
    if binaries != selection["executable_hashes"]:
        raise ValueError("An executable changed after validation; rerun validation")
    artifact = {"frozen": True, "frozen_utc": datetime.now(timezone.utc).isoformat(),
                "protocol_sha256": sha256(protocol_path), "runner_sha256": sha256(Path(__file__)),
                "evaluator_sha256": sha256(ROOT / "tools/evaluate.py"), "executable_hashes": binaries,
                "selection_sha256": sha256(selection_path), "selected_profile": selection["selected_profile"],
                "confirmation_inputs": confirmation_inputs, "image_integrity": image_integrity, "protocol": protocol,
                "scope": "Prior test data reused exploratory; additional confirmation clips unscored by this runner before freeze. No post-confirmation tuning."}
    if recommendation_path.exists():
        recommendation = json.loads(recommendation_path.read_text(encoding="utf-8"))
        if recommendation.get("budget_profile") is not None and recommendation["budget_profile"] not in protocol["profiles"]:
            raise ValueError("Recommended budget_profile must be a profile declared before confirmation")
        artifact["recommended_profile"] = {"path": "configs/recommended_profile.json", "sha256": sha256(recommendation_path),
                                           "content": recommendation}
    write_json(frozen_path, artifact)
    print(f"Frozen validation-selected profile {artifact['selected_profile']}: {frozen_path}")
    return artifact


def check_freeze(protocol_path, output):
    path = output / "freeze.json"
    if not path.is_file():
        raise ValueError("Confirmation requires an explicit validation-only --phase freeze")
    frozen = json.loads(path.read_text(encoding="utf-8"))
    if not frozen.get("frozen") or frozen["protocol_sha256"] != sha256(protocol_path):
        raise ValueError("Protocol differs from the frozen confirmation plan")
    if frozen["runner_sha256"] != sha256(Path(__file__)) or frozen["evaluator_sha256"] != sha256(ROOT / "tools/evaluate.py"):
        raise ValueError("Runner/evaluator changed after freeze")
    for name, expected in frozen["executable_hashes"].items():
        if sha256(ROOT / frozen["protocol"]["executables"][name]) != expected:
            raise ValueError("An executable changed after freeze")
    for case in frozen["protocol"]["confirmation_cases"]:
        if case_inputs(case)[-1] != frozen["confirmation_inputs"][case_key(case)]:
            raise ValueError("Confirmation inputs changed after freeze")
    for case in [case for group in frozen["protocol"]["validation_groups"].values() for case in group["cases"]] + frozen["protocol"]["confirmation_cases"]:
        sequence, metadata = case_inputs(case)[:2]
        if verify_image_content(sequence, metadata) != frozen["image_integrity"][case_key(case)]:
            raise ValueError("Verified image sequence differs from the frozen source bytes/order")
    recommendation = frozen.get("recommended_profile")
    if recommendation and sha256(ROOT / recommendation["path"]) != recommendation["sha256"]:
        raise ValueError("Validation-selected recommended profile changed after freeze")
    return frozen


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--protocol", type=Path, default=ROOT / "configs/refinement_protocol.json")
    parser.add_argument("--output", type=Path, default=ROOT / "runs/refinement")
    parser.add_argument("--phase", choices=("validation", "freeze", "exploratory", "confirmation", "diagnostic"), required=True)
    parser.add_argument("--profiles", nargs="+", help="Declared profile names; default all profiles")
    parser.add_argument("--schedules", nargs="+", default=["every"], help="Declared schedule names; adaptive treatments automatically get an exact matched control")
    parser.add_argument("--paired-output", action="store_true", help="Run active and all separately; selection remains primary active only")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)
    try:
        protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
        if not math.isclose(sum(group["weight"] for group in protocol["validation_groups"].values()), 1):
            raise ValueError("Validation source-group weights must sum to one")
        if args.phase == "freeze":
            freeze(protocol, args.protocol, args.output)
            return 0
        frozen = check_freeze(args.protocol, args.output) if args.phase == "confirmation" else None
        if args.phase == "validation" and (args.output / "freeze.json").exists():
            raise ValueError("Study already frozen; do not tune validation after confirmation in the same study folder")
        confirmation_profiles = None
        if frozen:
            permitted = ["baseline_v2", "two_stage", frozen["selected_profile"]]
            budget_profile = frozen.get("recommended_profile", {}).get("content", {}).get("budget_profile")
            if budget_profile is not None:
                if budget_profile not in protocol["profiles"]:
                    raise ValueError("Frozen budget profile was not predeclared")
                permitted.append(budget_profile)
            confirmation_profiles = list(dict.fromkeys(permitted))
        profiles = args.profiles or confirmation_profiles or list(protocol["profiles"])
        if frozen and any(name not in confirmation_profiles for name in profiles):
            raise ValueError("Fresh confirmation permits only validation-selected quality/budget profiles and baseline_v2/two_stage controls")
        if any(name not in protocol["profiles"] for name in profiles) or any(name not in protocol["schedules"] for name in args.schedules):
            raise ValueError("Unknown profile or schedule; declare it before scoring")
        cases = ([case for group in protocol["validation_groups"].values() for case in group["cases"]]
                 if args.phase == "validation" else protocol[args.phase + "_cases"])
        protected = [args.protocol, Path(__file__), ROOT / "tools/evaluate.py"]
        protected.extend(ROOT / path for path in protocol["executables"].values())
        for case in cases:
            protected.extend(protected_case_paths(case, any(protocol["schedules"][name].get("schedule") == "adaptive" for name in args.schedules)))
        writes = [args.output / args.phase / "aggregate.json"]
        if args.phase == "validation":
            writes.append(args.output / "validation/selection.json")
        guard_file_writes(writes, protected)
        policies = [protocol["primary_output_policy"]]
        if args.paired_output:
            policies.append(protocol["diagnostic_output_policy"])
        records = []
        for case in cases:
            for profile in profiles:
                for schedule_name in args.schedules:
                    schedule = protocol["schedules"][schedule_name]
                    if profile == "baseline_v2" and (schedule.get("scene_method", "legacy") != "legacy" or schedule.get("risk_scope", "all") != "all"):
                        continue  # Unsupported treatments are explicit, never silently downgraded.
                    for policy in policies:
                        directory = args.output / args.phase / case_key(case) / profile / schedule_name / policy
                        record = run_one(protocol, case, profile, schedule_name, schedule, policy, directory, args.phase, args.force, args.protocol)
                        records.append(record)
                        if schedule["schedule"] == "adaptive":
                            matched = {**schedule, "schedule": "uniform", "uniform_calls": record["summary"]["detector_calls"]}
                            matched_name = "matched__" + schedule_name
                            matched_directory = args.output / args.phase / case_key(case) / profile / matched_name / policy
                            control = run_one(protocol, case, profile, matched_name, matched, policy, matched_directory, args.phase, args.force, args.protocol)
                            if control["summary"]["detector_calls"] != record["summary"]["detector_calls"]:
                                raise ValueError("Matched control did not preserve the exact per-sequence call count")
                            treatment_knobs = {key: value for key, value in record["knobs"].items() if key not in ("schedule", "uniform_calls")}
                            control_knobs = {key: value for key, value in control["knobs"].items() if key not in ("schedule", "uniform_calls")}
                            if treatment_knobs != control_knobs:
                                raise ValueError("Matched control changed an association/output/signal knob")
                            records.append(control)
        report = {"phase": args.phase, "protocol_sha256": sha256(args.protocol),
                  "selected_profile": frozen["selected_profile"] if frozen else None,
                  "evidence_scope": "additional predeclared confirmation" if args.phase == "confirmation" else
                                    "previously inspected test data: exploratory" if args.phase == "exploratory" else args.phase,
                  "aggregate": summarize(records),
                  "experiments": [str(Path(record["output"]) / "experiment.json") for record in records]}
        write_json(args.output / args.phase / "aggregate.json", report)
        if args.phase == "validation":
            selection = select_validation(protocol, records)
            selection.update({"protocol_sha256": sha256(args.protocol),
                              "runner_sha256": sha256(Path(__file__)),
                              "evaluator_sha256": sha256(ROOT / "tools/evaluate.py"),
                              "validation_inputs": {case_key(case): case_inputs(case)[-1] for group in protocol["validation_groups"].values() for case in group["cases"]},
                              "executable_hashes": {name: sha256(ROOT / path) for name, path in protocol["executables"].items()},
                              "experiment_paths": report["experiments"]})
            write_json(args.output / "validation/selection.json", selection)
            print(json.dumps({"selected_profile": selection["selected_profile"], "outcomes": selection["outcomes"]}, indent=2))
        else:
            print(json.dumps(report["aggregate"], indent=2))
    except (OSError, ValueError, KeyError, RuntimeError) as error:
        parser.exit(2, f"error: {error}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
