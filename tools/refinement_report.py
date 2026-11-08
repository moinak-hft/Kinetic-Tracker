#!/usr/bin/env python3
"""Create an offline evidence report from an already executed refinement study.

This tool never invokes a detector, changes a protocol or selects a new method.
Its final report requires the existing validation freeze and confirmation outputs.
"""
from __future__ import annotations

import argparse
import base64
from collections import defaultdict
import csv
import hashlib
import html
import json
import math
import os
from pathlib import Path
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SCORES = ("HOTA", "IDF1", "AssA", "DetA", "MOTA")
COUNTS = ("IDSW", "FP", "FN", "TP", "Frag")
PHASES = ("validation", "confirmation", "exploratory", "diagnostic")
PALETTE = {"ink": "#172b43", "teal": "#138575", "blue": "#547ca8", "orange": "#bf773d",
           "red": "#b26868", "muted": "#788597", "grid": "#e3e9ef", "soft": "#f5f8fa"}
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.labelcolor": PALETTE["ink"],
                     "text.color": PALETTE["ink"], "xtick.color": PALETTE["muted"],
                     "ytick.color": PALETTE["ink"], "savefig.facecolor": "white"})


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def relative(path: Path) -> str:
    path = Path(path).resolve()
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        return path.name


def pretty(value: str) -> str:
    names = {"baseline_v2": "Frozen v2 control", "two_stage": "Two-stage control",
             "every": "Every frame", "fixed2": "Every 2 frames", "fixed4": "Every 4 frames",
             "motion_hybrid_full": "Hybrid motion + recovery",
             "motion_hybrid_no_recovery": "Hybrid without recovery",
             "motion_hybrid_iou": "Hybrid IoU fallback", "motion_full": "Strict motion + recovery",
             "motion_full_gate25": "Strict motion, gate25", "motion_uncertainty": "Uncertainty gate + cost",
             "motion_direction": "Direction + uncertainty gate", "recovery": "Conservative third pass",
             "adaptive50_legacy_all": "Adaptive · legacy cue / all risk",
             "adaptive50_legacy_active": "Adaptive · legacy cue / active risk",
             "adaptive50_robust_all": "Adaptive · robust cue / all risk",
             "adaptive50_robust_active": "Adaptive · robust cue / active risk"}
    return names.get(value, value.replace("_", " "))


def case_key(record):
    return "__".join(Path(record["case"]["path"]).parts[1:])


def source_group(record):
    meta = record.get("metadata", {})
    if meta.get("synthetic") or "synthetic" in record["case"]["path"]:
        return "Synthetic diagnostics"
    if record["case"].get("source") == "oracle" or meta.get("dataset") == "native_bee":
        return "Processed bee / oracle"
    if record["phase"] == "confirmation":
        return "Additional non-insect clips"
    return "Natural insect clips"


def metric(value, digits=2):
    return "—" if value is None else f"{100 * float(value):.{digits}f}"


def number(value, digits=2):
    if value is None:
        return "—"
    return str(int(value)) if isinstance(value, int) else f"{float(value):.{digits}f}"


def delta(value):
    return "—" if value is None else f"{100 * value:+.2f}"


def mean(values):
    values = list(values)
    return sum(values) / len(values) if values else None


def comparable_metrics(metrics):
    # Repeated controls have different output filenames. Keep scores, hashes
    # and evaluator settings; only the destination filename may differ.
    return {key: ({name: value for name, value in item.items() if name != 'pred_path'}
                  if key == 'metadata' and isinstance(item, dict) else item)
            for key, item in metrics.items()}


def safe_record_path(raw_path: str, runs: Path) -> Path:
    path = Path(raw_path)
    if not path.is_absolute():
        path = ROOT / path
    path = path.resolve()
    if not path.is_relative_to(runs.resolve()):
        raise ValueError(f"experiment path leaves this study: {raw_path}")
    return path


def compact_record(record, path):
    summary = record["summary"]
    return {
        "phase": record["phase"], "evidence_label": record.get("evidence_label", record["phase"]),
        "case": record["case"]["path"], "case_key": case_key(record), "sequence": record["metadata"].get("name", Path(record["case"]["path"]).name),
        "source_group": source_group(record), "source": record["case"].get("source", "provided"),
        "profile": record["profile"], "schedule": record["schedule_name"], "output_policy": record["output_policy"],
        "frames": summary["frames"], "detector_calls": summary["detector_calls"], "detector_rate": summary["detector_rate"],
        "metrics": {key: record["metrics"].get(key) for key in SCORES + COUNTS},
        "discovery": {key: record.get("discovery", {}).get(key) for key in
                      ("targets", "missed_targets", "short_targets", "missed_short_targets", "median_discovery_frames", "p95_discovery_frames")},
        "schedule_diagnostics": record.get("schedule_diagnostics", {}), "knobs": record["knobs"],
        "experiment": relative(path), "experiment_sha256": sha(path),
        "executable_sha256": record.get("executable_sha256"), "inputs": record.get("inputs", {}),
        "mot_sha256": record.get("mot_sha256"),
    }


def collect(runs: Path, allow_incomplete=False):
    selection_path, freeze_path = runs / "validation/selection.json", runs / "freeze.json"
    if not selection_path.is_file():
        raise ValueError("validation/selection.json is required; this reporter never performs selection")
    selection = read_json(selection_path)
    frozen = read_json(freeze_path) if freeze_path.is_file() else None
    if not allow_incomplete and (not frozen or not frozen.get("frozen")):
        raise ValueError("final reporting requires freeze.json; use --allow-incomplete only for a labelled preview")
    if frozen:
        if selection["selected_profile"] != frozen["selected_profile"] or sha(selection_path) != frozen["selection_sha256"]:
            raise ValueError("selection differs from the frozen selection artifact")
        protocol = frozen["protocol"]
    else:
        protocol = read_json(ROOT / "configs/refinement_protocol.json")
    warnings, source_files, paths, phase_documents = [], {relative(selection_path): sha(selection_path)}, [], {}
    if frozen:
        source_files[relative(freeze_path)] = sha(freeze_path)
    else:
        warnings.append("INCOMPLETE PREVIEW: validation has not been frozen; no confirmation claim is available.")
    paths.extend(selection.get("experiment_paths", []))
    for phase in PHASES:
        aggregate_path = runs / phase / "aggregate.json"
        if aggregate_path.is_file():
            document = read_json(aggregate_path)
            phase_documents[phase] = document
            if frozen and document.get("protocol_sha256") != frozen["protocol_sha256"]:
                raise ValueError(f"{phase} aggregate uses a different protocol from the freeze")
            source_files[relative(aggregate_path)] = sha(aggregate_path)
            paths.extend(document.get("experiments", []))
        elif phase != "validation":
            warnings.append(f"{phase.capitalize()} output is not yet present in this report snapshot.")
    scheduling_path = runs / "scheduling_validation/aggregate.json"
    scheduling_document = read_json(scheduling_path) if scheduling_path.is_file() else None
    if scheduling_document:
        source_files[relative(scheduling_path)] = sha(scheduling_path)
        paths.extend(scheduling_document.get("experiments", []))
    recommendation = (frozen or {}).get("recommended_profile", {}).get("content")
    if recommendation is None and scheduling_document:
        recommendation = scheduling_document.get("selection")
    if frozen and scheduling_document and recommendation != scheduling_document.get("selection"):
        raise ValueError("budget recommendation differs from the frozen recommendation")
    if not allow_incomplete and "confirmation" not in phase_documents:
        raise ValueError("final reporting requires confirmation/aggregate.json")
    records, unique_records = [], {}
    for path in sorted({safe_record_path(raw, runs) for raw in paths}):
        if not path.is_file():
            raise ValueError(f"declared experiment is missing: {relative(path)}")
        record = read_json(path)
        if record["phase"] not in PHASES or record["output_policy"] not in {"active", "all"}:
            raise ValueError("unrecognised evidence phase/output policy")
        if record["summary"].get("mode") != "cached_detection_replay":
            raise ValueError("refinement quality tables only accept cached replay; runtime is reported separately")
        if record["summary"].get("output_policy", record["output_policy"]) != record["output_policy"]:
            raise ValueError("record/output summary emission policies disagree")
        if int(record["metadata"]["num_frames"]) != int(record["summary"]["frames"]):
            raise ValueError("record length does not cover the declared sequence")
        for key in SCORES + COUNTS:
            if key not in record["metrics"] or not math.isfinite(float(record["metrics"][key])):
                raise ValueError(f"metric missing or nonfinite: {key}")
        if frozen:
            executable = protocol["profiles"][record["profile"]]["executable"]
            if record["executable_sha256"] != frozen["executable_hashes"][executable]:
                raise ValueError(f"experiment executable differs from freeze: {relative(path)}")
        record["_path"] = path
        record["_collection"] = path.relative_to(runs).parts[0]
        source_files[relative(path)] = sha(path)
        identity = (record["phase"], case_key(record), record["profile"], record["schedule_name"], record["output_policy"])
        if identity in unique_records:
            previous = unique_records[identity]
            if (any(previous.get(key) != record.get(key) for key in ("inputs", "knobs", "mot_sha256", "executable_sha256"))
                    or comparable_metrics(previous['metrics']) != comparable_metrics(record['metrics'])):
                raise ValueError("duplicate semantic comparison has conflicting inputs, settings or results")
            # The budget-selection runner deliberately repeats some every/fixed2
            # controls. Identical repeats are provenance, not extra clip evidence.
            if previous["_collection"] != "validation" and record["_collection"] == "validation":
                unique_records[identity] = record
        else:
            unique_records[identity] = record
    records = list(unique_records.values())
    selected = frozen["selected_profile"] if frozen else selection["selected_profile"]
    confirmation = [record for record in records if record["phase"] == "confirmation" and
                    record["output_policy"] == "active" and record["schedule_name"] == "every"]
    required_cases = {"__".join(Path(case["path"]).parts[1:]) for case in protocol.get("confirmation_cases", [])}
    if confirmation or not allow_incomplete:
        for profile in {"baseline_v2", selected}:
            found = {case_key(record) for record in confirmation if record["profile"] == profile}
            if found != required_cases:
                raise ValueError(f"confirmation is incomplete for profile {profile}")
    return selection, frozen, protocol, records, warnings, source_files, recommendation


def validation_rows(selection):
    accepted = set(selection.get("raw_guard_accepted_profiles", []))
    rows = []
    for profile, outcome in selection["outcomes"].items():
        row = {"profile": profile, "selected": profile == selection["selected_profile"], "raw_guard_pass": profile in accepted,
               **{f"balanced_{key}": outcome["source_balanced"][key] for key in SCORES}}
        for group, group_scores in outcome["groups"].items():
            row.update({f"{group}_{key}": value for key, value in group_scores.items()})
        rows.append(row)
    return rows


def confirmation_rows(records, selected):
    active = [r for r in records if r["phase"] == "confirmation" and r["output_policy"] == "active" and r["schedule_name"] == "every"]
    lookup = {(case_key(r), r["profile"]): r for r in active}
    rows = []
    for case in sorted({case_key(r) for r in active}):
        baseline, treatment = lookup.get((case, "baseline_v2")), lookup.get((case, selected))
        if not baseline or not treatment:
            continue
        if baseline["inputs"] != treatment["inputs"]:
            raise ValueError("confirmation baseline and selected profile use different inputs")
        row = {"case": case, "sequence": treatment["metadata"].get("name", case), "frames": treatment["summary"]["frames"],
               "baseline_profile": "baseline_v2", "selected_profile": selected,
               "baseline_calls": baseline["summary"]["detector_calls"], "selected_calls": treatment["summary"]["detector_calls"]}
        for key in SCORES + COUNTS:
            row[f"baseline_{key}"] = baseline["metrics"][key]
            row[f"selected_{key}"] = treatment["metrics"][key]
            row[f"delta_{key}"] = treatment["metrics"][key] - baseline["metrics"][key]
        for key in ("targets", "missed_targets", "short_targets", "missed_short_targets"):
            row[f"baseline_{key}"] = baseline["discovery"].get(key, 0)
            row[f"selected_{key}"] = treatment["discovery"].get(key, 0)
        rows.append(row)
    return rows


def budget_confirmation_rows(records, profile, schedule):
    """Separate association-at-fixed2 from full budget-policy comparisons."""
    active = [r for r in records if r['phase'] == 'confirmation' and r['output_policy'] == 'active']
    cases = sorted({case_key(r) for r in active})
    rows = []
    for case in cases:
        treatments = [('baseline_fixed2', 'baseline_v2', 'fixed2'),
                      ('budget_fixed2', profile, 'fixed2'), ('budget_adaptive', profile, schedule)]
        found = []
        for label, name, cadence in treatments:
            candidates = [r for r in active if case_key(r) == case and r['profile'] == name and r['schedule_name'] == cadence]
            if len(candidates) != 1:
                raise ValueError(f'Incomplete budget confirmation: {case} {name}/{cadence}')
            found.append((label, candidates[0]))
        if any(r['inputs'] != found[0][1]['inputs'] for _, r in found):
            raise ValueError('Budget confirmation changed source inputs')
        row = {'case': case, 'sequence': found[0][1]['metadata']['name'], 'frames': found[0][1]['summary']['frames'],
               'budget_profile': profile, 'budget_schedule': schedule,
               'scope': 'fixed2 association ablation and full adaptive policy; adaptive versus fixed2 is not an equal-call comparison'}
        for label, record in found:
            row[label + '_calls'] = record['summary']['detector_calls']
            for key in SCORES + COUNTS:
                row[label + '_' + key] = record['metrics'][key]
        rows.append(row)
    return rows


def matched_rows(records, selected):
    active = [r for r in records if r["output_policy"] == "active" and r["profile"] == selected]
    lookup = {(r["phase"], case_key(r), r["schedule_name"]): r for r in active}
    rows = []
    for treatment in active:
        if treatment["knobs"].get("schedule") != "adaptive":
            continue
        control = lookup.get((treatment["phase"], case_key(treatment), "matched__" + treatment["schedule_name"]))
        if not control:
            raise ValueError("adaptive treatment has no exact-count matched control")
        excluded = {"schedule", "uniform_calls"}
        a = {key: value for key, value in treatment["knobs"].items() if key not in excluded}
        b = {key: value for key, value in control["knobs"].items() if key not in excluded}
        if treatment["inputs"] != control["inputs"] or a != b or treatment["summary"]["detector_calls"] != control["summary"]["detector_calls"]:
            raise ValueError("matched comparison changed input, association/output knobs or realized detector calls")
        row = {"phase": treatment["phase"], "case": case_key(treatment), "sequence": treatment["metadata"]["name"],
               "source_group": source_group(treatment), "profile": selected, "schedule": treatment["schedule_name"],
               "output_policy": "active", "scene_method": treatment["knobs"].get("scene_method", "legacy"),
               "risk_scope": treatment["knobs"].get("risk_scope", "all"),
               "frames": treatment["summary"]["frames"], "calls": treatment["summary"]["detector_calls"],
               "detector_rate": treatment["summary"]["detector_rate"]}
        for key in SCORES + COUNTS:
            row[f"adaptive_{key}"] = treatment["metrics"][key]; row[f"uniform_{key}"] = control["metrics"][key]
            row[f"delta_{key}"] = treatment["metrics"][key] - control["metrics"][key]
        for key in ("targets", "missed_targets", "short_targets", "missed_short_targets"):
            row[f"adaptive_{key}"] = treatment["discovery"].get(key, 0); row[f"uniform_{key}"] = control["discovery"].get(key, 0)
        rows.append(row)
    return rows


def novelty_rows(records, selected):
    groups = defaultdict(dict)
    for record in records:
        knobs = record["knobs"]
        if (record["output_policy"] != "active" or record["profile"] != selected or knobs.get("schedule") != "adaptive"
                or knobs.get("risk_scope", "all") != "active"):
            continue
        fingerprint = json.dumps({key: value for key, value in knobs.items() if key != "scene_method"}, sort_keys=True)
        groups[(record["phase"], case_key(record), fingerprint)][knobs.get("scene_method", "legacy")] = record
    rows = []
    for (_, case, _), pair in sorted(groups.items()):
        if not {"legacy", "robust"}.issubset(pair):
            continue
        legacy, robust = pair["legacy"], pair["robust"]
        if legacy["inputs"] != robust["inputs"]:
            raise ValueError("legacy and robust scene cues use different source inputs")
        row = {"phase": robust["phase"], "case": case, "sequence": robust["metadata"]["name"],
               "source_group": source_group(robust), "profile": selected, "risk_scope": "active",
               "frames": robust["summary"]["frames"],
               "legacy_trigger_fraction": legacy["schedule_diagnostics"]["novelty_threshold_exceeded_fraction"],
               "robust_trigger_fraction": robust["schedule_diagnostics"]["novelty_threshold_exceeded_fraction"],
               "legacy_call_rate": legacy["summary"]["detector_rate"], "robust_call_rate": robust["summary"]["detector_rate"],
               "legacy_HOTA": legacy["metrics"]["HOTA"], "robust_HOTA": robust["metrics"]["HOTA"],
               "delta_HOTA": robust["metrics"]["HOTA"] - legacy["metrics"]["HOTA"],
               "legacy_IDF1": legacy["metrics"]["IDF1"], "robust_IDF1": robust["metrics"]["IDF1"]}
        rows.append(row)
    return rows


def write_csv(path, rows):
    if not rows:
        path.write_text("status\nno_records_available\n", encoding="utf-8")
        return
    columns = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, columns); writer.writeheader(); writer.writerows(rows)


def save_figure(figure, path):
    figure.savefig(path, dpi=170, bbox_inches="tight")
    plt.close(figure)


def plot_validation(rows, selection, path):
    figure, axes = plt.subplots(1, 2, figsize=(13.6, max(4.3, len(rows)*.48)), gridspec_kw={"width_ratios": [1.6, 1]})
    y = np.arange(len(rows)); names = [pretty(row["profile"]) for row in rows]
    colours = [PALETTE["teal"] if row["selected"] else PALETTE["blue"] if row["raw_guard_pass"] else PALETTE["red"] for row in rows]
    values = [100*row["balanced_HOTA"] for row in rows]
    axes[0].barh(y, values, color=colours, height=.64)
    axes[0].set_yticks(y, names); axes[0].invert_yaxis()
    axes[0].set_xlabel("Source-balanced HOTA (0–100)"); axes[0].set_xlim(0, min(100, max(values)*1.18))
    axes[0].grid(axis="x", color=PALETTE["grid"], zorder=0); axes[0].set_axisbelow(True)
    for index, value in enumerate(values):
        axes[0].text(value+.35, index, f"{value:.2f}" + ("  selected" if rows[index]["selected"] else ""), va="center", fontsize=9)
    for offset, group, colour, label in ((-.15, "synthetic", PALETTE["blue"], "Synthetic mean"),
                                          (.15, "raw_insect", PALETTE["orange"], "Natural insect validation")):
        axes[1].scatter([100*row.get(f"{group}_HOTA", np.nan) for row in rows], y+offset, color=colour, s=44, label=label)
    axes[1].set_yticks(y, ["guard pass" if row["raw_guard_pass"] else "guard rejected" for row in rows])
    axes[1].invert_yaxis(); axes[1].set_xlim(0, 100); axes[1].set_xlabel("Per-source HOTA (0–100)")
    axes[1].grid(axis="x", color=PALETTE["grid"]); axes[1].legend(loc="lower right", fontsize=8)
    figure.suptitle("Validation selection keeps all declared candidate outcomes", x=.02, ha="left", fontsize=15, weight="bold")
    figure.text(.02, .015, "Active output only. Red candidates failed the raw-insect guard; thresholds and weights are from the recorded selection protocol.", fontsize=9, color=PALETTE["muted"])
    figure.tight_layout(rect=(0, .045, 1, .94)); save_figure(figure, path)


def plot_confirmation(rows, selected, path):
    if not rows:
        return
    figure, axes = plt.subplots(1, 2, figsize=(13.0, 5.1))
    y = np.arange(len(rows)); labels = [row["sequence"] for row in rows]
    for axis, name in zip(axes, ("HOTA", "IDF1")):
        axis.bar(y-.18, [100*row[f"baseline_{name}"] for row in rows], .34, color=PALETTE["blue"], label="Frozen v2")
        axis.bar(y+.18, [100*row[f"selected_{name}"] for row in rows], .34, color=PALETTE["teal"], label=pretty(selected))
        axis.set_xticks(y, labels, rotation=35, ha='right'); axis.set_ylim(0, 100); axis.set_ylabel(f"{name} (0–100)")
        axis.grid(axis="y", color=PALETTE["grid"]); axis.set_axisbelow(True); axis.legend(fontsize=8)
        for index, row in enumerate(rows):
            top = 100*max(row[f"baseline_{name}"], row[f"selected_{name}"])
            axis.text(index, min(97, top+3), f"Δ {delta(row[f'delta_{name}'])}", ha="center", fontsize=9)
    figure.suptitle("Additional confirmation: per-clip results, without retuning", x=.02, ha="left", fontsize=15, weight="bold")
    figure.text(.02, .01, "Additional non-insect categories broaden coverage; one clip per category is not full-benchmark or session-independent generalisation.", fontsize=9, color=PALETTE["muted"])
    figure.tight_layout(rect=(0, .045, 1, .93)); save_figure(figure, path)


def curve_groups(records, selected):
    groups = defaultdict(list)
    for record in records:
        if record["output_policy"] == "active" and record["profile"] == selected and record["phase"] in {"exploratory", "confirmation"}:
            groups[(record["phase"], source_group(record))].append(record)
    output = []
    for (phase, group_name), group in sorted(groups.items()):
        cohort = {case_key(record) for record in group}
        by_schedule = defaultdict(list)
        for record in group:
            by_schedule[record["schedule_name"]].append(record)
        points = []
        for schedule, rows in sorted(by_schedule.items()):
            if {case_key(record) for record in rows} != cohort:
                continue  # Never average methods over different case cohorts.
            frames = sum(row["summary"]["frames"] for row in rows)
            calls = sum(row["summary"]["detector_calls"] for row in rows)
            points.append({"phase": phase, "source_group": group_name, "profile": selected, "schedule": schedule,
                           "sequences": len(rows), "frames": frames, "calls": calls, "detector_rate": calls/frames,
                           "HOTA": mean(row["metrics"]["HOTA"] for row in rows), "IDF1": mean(row["metrics"]["IDF1"] for row in rows)})
        if points:
            output.append((phase, group_name, points))
    return output


def plot_scheduling(groups, path):
    if not groups:
        return
    figure, axes = plt.subplots(1, len(groups), figsize=(max(7, 5.2*len(groups)), 5), squeeze=False)
    schedules = sorted({point["schedule"] for _, _, points in groups for point in points})
    cmap = plt.get_cmap("tab10"); colours = {schedule: cmap(index % 10) for index, schedule in enumerate(schedules)}
    for axis, (phase, group_name, points) in zip(axes[0], groups):
        for point in points:
            schedule = point["schedule"]; matched = schedule.startswith("matched__")
            axis.scatter(100*point["detector_rate"], 100*point["HOTA"], s=76,
                         marker="o" if matched else "D" if "adaptive" in schedule else "s",
                         facecolors="none" if matched else colours[schedule], edgecolors=colours[schedule],
                         linewidths=1.4, label=schedule)
        axis.set_title(f"{group_name}\n{phase}", fontsize=11)
        axis.set_xlabel("Actual detector calls / observations (%)"); axis.set_ylabel("Mean HOTA (0–100)")
        axis.set_xlim(-3, 104); axis.set_ylim(0, 100); axis.grid(color=PALETTE["grid"]); axis.set_axisbelow(True)
    handles, labels = {}, {}
    for axis in axes[0]:
        for handle, label in zip(*axis.get_legend_handles_labels()):
            handles[label] = handle; labels[label] = label.replace("matched__", "matched uniform: ").replace("adaptive50_", "adaptive: ").replace("_", " ")
    figure.legend([handles[key] for key in schedules if key in handles], [labels[key] for key in schedules if key in labels],
                  loc="lower center", ncol=3, fontsize=8, frameon=False)
    figure.suptitle("Quality / detector-call trade-off with association and output rules held fixed", x=.02, ha="left", fontsize=14, weight="bold")
    figure.text(.02, .19, "Hollow points are retrospective exact-count uniform controls. Cached replay excludes neural detector work; these are not live speed measurements.", fontsize=8.5, color=PALETTE["muted"])
    figure.tight_layout(rect=(0, .24, 1, .93)); save_figure(figure, path)


def plot_novelty(rows, path):
    if not rows:
        return
    figure, axis = plt.subplots(figsize=(11.8, max(4.2, len(rows)*.43)))
    y = np.arange(len(rows))
    axis.barh(y-.16, [100*row["legacy_trigger_fraction"] for row in rows], .30, color=PALETTE["orange"], label="Legacy pixel difference")
    axis.barh(y+.16, [100*row["robust_trigger_fraction"] for row in rows], .30, color=PALETTE["teal"], label="Causal robust cue")
    axis.set_yticks(y, [row["sequence"] + (" · additional" if row["phase"] == "confirmation" else " · reused") for row in rows])
    axis.invert_yaxis(); axis.set_xlim(0, 103); axis.set_xlabel("Observations exceeding the declared novelty threshold (%)")
    axis.grid(axis="x", color=PALETTE["grid"]); axis.set_axisbelow(True); axis.legend(loc="lower right", fontsize=9)
    figure.suptitle("Scene-cue activation, with active-only risk scope held fixed", x=.02, ha="left", fontsize=14, weight="bold")
    figure.text(.02, .015, "A lower trigger fraction is a diagnostic, not a tracking-quality gain. Track identities, misses and exact-call controls are reported separately.", fontsize=9, color=PALETTE["muted"])
    figure.tight_layout(rect=(0, .045, 1, .94)); save_figure(figure, path)


def html_table(headers, rows, classes="", attributes=None):
    head = "".join(f"<th>{html.escape(str(value))}</th>" for value in headers)
    body = []
    for index, row in enumerate(rows):
        attrs = ""
        if attributes:
            attrs = " " + " ".join(f'data-{key}="{html.escape(str(value), quote=True)}"' for key, value in attributes[index].items())
        body.append(f"<tr{attrs}>" + "".join(f"<td>{html.escape(str(value))}</td>" for value in row) + "</tr>")
    return f'<div class="table-scroll"><table class="{classes}"><thead><tr>{head}</tr></thead><tbody>{"".join(body)}</tbody></table></div>'


def markdown_table(headers, rows):
    def cell(value):
        return str(value).replace("|", "\\|").replace("\n", " ")
    return "\n".join(["| " + " | ".join(map(cell, headers)) + " |", "| " + " | ".join("---" for _ in headers) + " |"] +
                     ["| " + " | ".join(map(cell, row)) + " |" for row in rows])


def image_embed(path: Path, caption: str):
    if not path.is_file():
        return '<div class="empty">This evidence phase is not present in the current snapshot.</div>'
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f'<figure><img src="data:image/png;base64,{encoded}" alt="{html.escape(caption, quote=True)}"><figcaption>{html.escape(caption)}</figcaption></figure>'


def flatten_numeric(value, prefix=""):
    output = []
    if isinstance(value, dict):
        for key, child in value.items():
            if any(token in key.lower() for token in ("environment", "command", "path", "directory", "user", "hostname", "token", "secret")):
                continue
            output.extend(flatten_numeric(child, f"{prefix}.{key}" if prefix else key))
    elif isinstance(value, list):
        # Keep repeat-level measurements available without dumping commands or hosts.
        for index, child in enumerate(value[:20]):
            output.extend(flatten_numeric(child, f"{prefix}[{index}]"))
    elif isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value):
        output.append({"measurement": prefix, "value": value})
    return output


def optional_evidence(runs):
    references, sources = [], {}
    directory = runs / "reference_confirmation"
    if directory.is_dir():
        for metrics_path in sorted(directory.glob("**/metrics.json")):
            scores = read_json(metrics_path)
            if isinstance(scores.get("metrics"), dict):
                scores = scores["metrics"]
            row = {"sequence": metrics_path.parent.parent.name, "algorithm": metrics_path.parent.name}
            for key in SCORES + COUNTS:
                if key in scores and isinstance(scores[key], (int, float)) and math.isfinite(scores[key]):
                    row[key] = scores[key]
            if "HOTA" in row:
                row["metrics_sha256"] = sha(metrics_path); references.append(row)
                sources[relative(metrics_path)] = row["metrics_sha256"]
        for manifest in sorted(directory.glob("**/*manifest*.json")):
            sources[relative(manifest)] = sha(manifest)
    runtime_path = runs / "runtime/aggregate.json"
    runtime = read_json(runtime_path) if runtime_path.is_file() else None
    if runtime_path.is_file():
        sources[relative(runtime_path)] = sha(runtime_path)
    return references, runtime, sources


def public_experiment_rows(records):
    rows = []
    for record in records:
        rows.append({"phase": record["phase"], "source_group": record["source_group"], "sequence": record["sequence"],
                     "profile": record["profile"], "schedule": record["schedule"], "output_policy": record["output_policy"],
                     "frames": record["frames"], "calls": record["detector_calls"], "detector_rate": record["detector_rate"],
                     **record["metrics"], **record["discovery"], "experiment": record["experiment"],
                     "experiment_sha256": record["experiment_sha256"]})
    return rows


CSS = """
:root{--ink:#172b43;--muted:#617387;--teal:#138575;--line:#dfe7ee;--bg:#f3f6f9}*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.6 system-ui,-apple-system,Segoe UI,sans-serif}a{color:#126e8b;text-underline-offset:3px}header{background:#11263c;color:#fff;padding:58px max(24px,calc((100vw - 1180px)/2)) 48px}.eyebrow{text-transform:uppercase;letter-spacing:.17em;font-size:11px;color:#a9c9d6;font-weight:700}h1{font-size:clamp(34px,5vw,62px);line-height:1.05;letter-spacing:-.045em;margin:17px 0 20px;max-width:850px}header p{max-width:840px;color:#d2e0e9;font-size:17px}.badge{display:inline-block;padding:4px 11px;border-radius:30px;font-size:11px;letter-spacing:.07em;font-weight:750;background:#d9f3ea;color:#096b59}.badge.preview{background:#ffe5c1;color:#875005}main{max-width:1230px;margin:0 auto;padding:26px 24px 70px}.cards{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:14px;margin:-3px 0 26px}.card,section{background:white;border:1px solid var(--line);border-radius:13px}.card{padding:19px}.card span{display:block;color:var(--muted);font-size:12px}.card strong{display:block;font-size:23px;line-height:1.25;margin:8px 0;overflow-wrap:anywhere}.card small{color:var(--muted)}section{padding:27px 30px;margin:20px 0}h2{font-size:23px;letter-spacing:-.025em;margin:0 0 12px}h3{font-size:17px;margin:24px 0 10px}p{margin:10px 0 16px}.lead{font-size:18px}.note{padding:15px 18px;border-left:4px solid var(--teal);background:#edf7f4;border-radius:5px}.warning{border-left-color:#c2893f;background:#fff7e8}.scope{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:14px}.scope article{padding:17px;background:#f5f8fa;border-radius:8px}.scope strong{display:block;margin-bottom:6px}.muted,figcaption{color:var(--muted);font-size:13px}figure{margin:24px 0 16px}figure img{display:block;width:100%;height:auto;border:1px solid #edf0f4;border-radius:8px}figcaption{margin:9px 0}.table-scroll{overflow:auto;border:1px solid var(--line);border-radius:9px;margin:15px 0 20px}table{border-collapse:collapse;min-width:100%;font-size:12px;font-variant-numeric:tabular-nums}th,td{padding:11px 12px;white-space:nowrap;text-align:right;border-bottom:1px solid #e8edf2}th{background:#f4f7fa;color:#53677d;font-size:11px;font-weight:700;position:sticky;top:0}td:first-child,th:first-child{text-align:left}tbody tr:last-child td{border-bottom:0}tbody tr:hover{background:#f6fbfa}code{font-size:12px;background:#eef3f6;padding:2px 5px;border-radius:4px;overflow-wrap:anywhere}.empty{padding:20px;background:#f6f8fa;border:1px dashed #c9d4df;border-radius:7px;color:var(--muted)}.downloads{display:flex;flex-wrap:wrap;gap:10px;margin:16px 0}.downloads a{padding:7px 11px;border:1px solid #ccdce6;border-radius:6px;background:#f8fbfc;text-decoration:none;font-size:12px}.filters{display:flex;gap:10px;flex-wrap:wrap;margin:18px 0}.filters input,.filters select{font:inherit;font-size:13px;border:1px solid #ccd7e1;border-radius:6px;padding:9px 11px;background:white}.filters input{flex:1;min-width:220px}.video-grid{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:14px}.video-grid video{width:100%;background:#0f1e2a;border-radius:7px}.video-grid article{padding:14px;background:#f4f7fa;border-radius:9px}.video-grid h3{margin:0 0 10px;font-size:15px}details{margin-top:18px}summary{cursor:pointer;font-weight:650}footer{font-size:12px;color:var(--muted);padding:18px 3px}ul{padding-left:22px}li{margin:6px 0}@media(max-width:850px){.cards{grid-template-columns:repeat(2,minmax(0,1fr))}.scope,.video-grid{grid-template-columns:1fr}section{padding:23px 19px}}@media(max-width:520px){header{padding:38px 22px}.cards{grid-template-columns:1fr 1fr;gap:9px}.card{padding:13px}.card strong{font-size:18px}main{padding:20px 12px}.card small{font-size:10px}h2{font-size:21px}}@media print{header{background:white;color:var(--ink);padding:20px}header p{color:var(--muted)}main{max-width:none;padding:0}.filters,.video-grid{display:none}section,.card{break-inside:avoid;box-shadow:none}.table-scroll{overflow:visible}body{background:white}}
"""


def render_report(output, evidence, protocol, warnings):
    selected, validation = evidence["selected_profile"], evidence["validation"]
    budget_profile = evidence["budget_profile"]
    budget_validation = evidence["budget_validation"]
    budget_confirmation = evidence['budget_confirmation']
    budget_confirm_headers = ['Clip', 'Baseline fixed2 HOTA', 'Motion fixed2 HOTA', 'Budget adaptive HOTA', 'Calls: baseline / fixed2 / adaptive']
    budget_confirm_cells = [[row['sequence'], metric(row['baseline_fixed2_HOTA']), metric(row['budget_fixed2_HOTA']),
                            metric(row['budget_adaptive_HOTA']), f"{row['baseline_fixed2_calls']} / {row['budget_fixed2_calls']} / {row['budget_adaptive_calls']}"]
                           for row in budget_confirmation]
    if budget_confirmation:
        budget_confirm_cells.append(['Clip mean / total calls', metric(mean(r['baseline_fixed2_HOTA'] for r in budget_confirmation)),
                                     metric(mean(r['budget_fixed2_HOTA'] for r in budget_confirmation)), metric(mean(r['budget_adaptive_HOTA'] for r in budget_confirmation)),
                                     ' / '.join(str(sum(r[key + '_calls'] for r in budget_confirmation)) for key in ('baseline_fixed2', 'budget_fixed2', 'budget_adaptive'))])
    budget_cells = [[pretty(row["profile"]), metric(row["balanced_HOTA"]), metric(row["balanced_IDF1"]),
                     metric(row.get("synthetic_HOTA")), metric(row.get("raw_insect_HOTA")),
                     "pass" if row["raw_guard_pass"] else "rejected", "yes" if row["selected"] else ""] for row in budget_validation]
    confirmation, matching, novelty = evidence["confirmation"], evidence["matched_scheduling"], evidence["novelty"]
    complete = evidence["frozen"] and bool(confirmation)
    validation_selected = next(row for row in validation if row["profile"] == selected)
    frames = sum(row["frames"] for row in confirmation)
    change = mean(row["delta_HOTA"] for row in confirmation)
    if confirmation:
        outcome = (f"The selected profile matched the frozen baseline's mean HOTA across {len(confirmation)} additional clips."
                   if abs(change) < 1e-10 else
                   f"The selected profile changed mean HOTA by {delta(change)} points across {len(confirmation)} additional clips.")
    else:
        outcome = "Additional confirmation results are not yet available. This is an incomplete preview."
    headline = f"Validation selected {pretty(selected)}. " + outcome
    if budget_confirmation:
        base = mean(r['baseline_fixed2_HOTA'] for r in budget_confirmation)
        motion = mean(r['budget_fixed2_HOTA'] for r in budget_confirmation)
        better = sum(r['budget_fixed2_HOTA'] > r['baseline_fixed2_HOTA'] for r in budget_confirmation)
        worse = sum(r['budget_fixed2_HOTA'] < r['baseline_fixed2_HOTA'] for r in budget_confirmation)
        calls = sum(r['budget_fixed2_calls'] for r in budget_confirmation)
        headline = (f"At every-second-frame detection, hybrid motion changed mean HOTA from {metric(base)} to {metric(motion)} "
                    f"({delta(motion-base)} points) across {len(budget_confirmation)} additional clips with identical {calls} detector calls. "
                    f"{better} clips improved and {worse} regressed. Every-frame quality retained {pretty(selected)}.")
    headers = ["Candidate", "Balanced HOTA", "Balanced IDF1", "Synthetic HOTA", "Insect HOTA", "Raw guard", "Selected"]
    val_cells = [[pretty(row["profile"]), metric(row["balanced_HOTA"]), metric(row["balanced_IDF1"]),
                  metric(row.get("synthetic_HOTA")), metric(row.get("raw_insect_HOTA")),
                  "pass" if row["raw_guard_pass"] else "rejected", "yes" if row["selected"] else ""] for row in validation]
    confirm_headers = ["Clip", "Frames", "v2 HOTA", "Selected HOTA", "Δ HOTA", "v2 IDF1", "Selected IDF1", "Δ IDF1", "v2 / new IDSW", "v2 / new FP", "v2 / new FN"]
    confirm_cells = [[row["sequence"], row["frames"], metric(row["baseline_HOTA"]), metric(row["selected_HOTA"]), delta(row["delta_HOTA"]),
                      metric(row["baseline_IDF1"]), metric(row["selected_IDF1"]), delta(row["delta_IDF1"]),
                      f"{row['baseline_IDSW']} / {row['selected_IDSW']}", f"{row['baseline_FP']} / {row['selected_FP']}",
                      f"{row['baseline_FN']} / {row['selected_FN']}"] for row in confirmation]
    if confirmation:
        confirm_cells.append(["Unweighted mean / summed counts", frames,
                              metric(mean(row["baseline_HOTA"] for row in confirmation)), metric(mean(row["selected_HOTA"] for row in confirmation)), delta(change),
                              metric(mean(row["baseline_IDF1"] for row in confirmation)), metric(mean(row["selected_IDF1"] for row in confirmation)),
                              delta(mean(row["delta_IDF1"] for row in confirmation)),
                              f"{sum(row['baseline_IDSW'] for row in confirmation)} / {sum(row['selected_IDSW'] for row in confirmation)}",
                              f"{sum(row['baseline_FP'] for row in confirmation)} / {sum(row['selected_FP'] for row in confirmation)}",
                              f"{sum(row['baseline_FN'] for row in confirmation)} / {sum(row['selected_FN'] for row in confirmation)}"])
    match_headers = ["Scope", "Clip", "Adaptive schedule", "Calls", "Adaptive HOTA", "Uniform HOTA", "Δ HOTA", "Δ IDF1", "Adaptive / uniform missed targets"]
    match_cells = [[row["phase"], row["sequence"], row["schedule"], f"{row['calls']}/{row['frames']}", metric(row["adaptive_HOTA"]),
                    metric(row["uniform_HOTA"]), delta(row["delta_HOTA"]), delta(row["delta_IDF1"]),
                    f"{row['adaptive_missed_targets']} / {row['uniform_missed_targets']}"] for row in matching]
    novelty_headers = ["Scope", "Clip", "Legacy trigger %", "Robust trigger %", "Legacy calls %", "Robust calls %", "Δ HOTA", "Legacy / robust IDF1"]
    novelty_cells = [[row["phase"], row["sequence"], metric(row["legacy_trigger_fraction"]), metric(row["robust_trigger_fraction"]),
                      metric(row["legacy_call_rate"]), metric(row["robust_call_rate"]), delta(row["delta_HOTA"]),
                      f"{metric(row['legacy_IDF1'])} / {metric(row['robust_IDF1'])}"] for row in novelty]
    diagnostics = [row for row in evidence["experiments"] if row["phase"] == "diagnostic"]
    diagnostic_headers = ["Processed / oracle clip", "Profile", "Schedule", "Output", "HOTA", "IDF1", "IDSW", "FP", "FN"]
    diagnostic_cells = [[row["sequence"], row["profile"], row["schedule"], row["output_policy"], metric(row["metrics"]["HOTA"]),
                         metric(row["metrics"]["IDF1"]), row["metrics"]["IDSW"], row["metrics"]["FP"], row["metrics"]["FN"]] for row in diagnostics]
    record_headers = ["Phase", "Source", "Clip", "Profile", "Schedule", "Calls %", "HOTA", "IDF1", "AssA", "IDSW", "FP", "FN", "Missed targets"]
    record_cells, record_attrs = [], []
    for row in evidence["experiments"]:
        record_cells.append([row["phase"], row["source_group"], row["sequence"], row["profile"], row["schedule"], metric(row["detector_rate"]),
                             metric(row["metrics"]["HOTA"]), metric(row["metrics"]["IDF1"]), metric(row["metrics"]["AssA"]),
                             row["metrics"]["IDSW"], row["metrics"]["FP"], row["metrics"]["FN"], row["discovery"].get("missed_targets", "—")])
        record_attrs.append({"phase": row["phase"], "source": row["source_group"]})
    refs = evidence["reference_confirmation"]
    ref_cells = [[row["sequence"], row["algorithm"], metric(row.get("HOTA")), metric(row.get("IDF1")),
                  number(row.get("IDSW"), 0), number(row.get("FP"), 0), number(row.get("FN"), 0)] for row in refs]
    runtime_cells = [[pretty(name) if name == 'baseline_v2' else ('Quality · every frame' if name == 'quality' else 'Budget · selected adaptive'),
                      data['runs'], number(data['wall_fps_mean']), number(data['wall_fps_stdev']),
                      f"{number(data['wall_fps_min'])}–{number(data['wall_fps_max'])}", number(data['detector_calls_mean'], 0)]
                     for name, data in evidence.get('runtime_profiles', {}).items()]
    runtime_headers = ['CPU profile', 'Runs', 'Mean wall FPS', 'SD', 'FPS range', 'Detector calls /240']
    warnings_html = "".join(f'<p class="note warning">{html.escape(warning)}</p>' for warning in warnings)
    videos = []
    video_specs = [("demo_insect.mp4", "Insect comparison", "Previously inspected insect footage: exploratory evidence."),
                   ("demo_background.mp4", "Background-motion diagnostic", "Rendered synthetic stimulus; nuisance response and failures remain visible.")]
    video_specs.extend((path.name, path.stem.removeprefix('demo_') + ' · additional confirmation',
                        'Baseline versus hybrid motion, both detect every2frames. All frames retained; improvements and regressions included.')
                       for path in sorted(output.glob('demo_*.mp4')) if path.name not in {'demo_insect.mp4', 'demo_background.mp4'})
    for name, title, scope in video_specs:
        if (output / name).is_file():
            content = f'<video controls preload="metadata" src="{name}"></video><p><a href="{name}">Open video</a>'
            attribution = Path(name).with_suffix(".attribution.json").name
            if (output / attribution).is_file():
                content += f' · <a href="{attribution}">Attribution / hashes</a>'
            content += "</p>"
        else:
            content = '<div class="empty">Video not generated in this snapshot.</div>'
        videos.append(f'<article><h3>{title}</h3>{content}<p class="muted">{scope}</p></article>')
    phase_options = "".join(f'<option value="{phase}">{phase.capitalize()}</option>' for phase in PHASES)
    source_options = "".join(f'<option value="{html.escape(source, quote=True)}">{html.escape(source)}</option>' for source in sorted({row["source_group"] for row in evidence["experiments"]}))
    selected_description = protocol["profiles"][selected].get("description", "")
    regression = evidence["baseline_regression_checks"]
    identical = sum(bool(row.get("mot_bytes_identical")) for row in regression)
    regression_text = (f"The current two-stage control produced byte-identical MOT outputs to the frozen v2 executable on {identical}/{len(regression)} recorded validation cases."
                       if regression else "No frozen-v2 byte-regression checks are present in this snapshot.")
    document = f'''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Kinetic-Tracker · Refinement evidence</title><style>{CSS}</style></head><body>
<header><div class="eyebrow">Kinetic-Tracker · Experimental refinement</div><h1>Kinetic-Tracker<br>Motion &amp; Compute Study</h1><p>A native C++ tracking study with validation-based selection, frozen comparison inputs, and additional confirmation clips. Every score below is derived from saved artifacts.</p><span class="badge {'preview' if not complete else ''}">{'FROZEN CONFIRMATION' if complete else 'INCOMPLETE PREVIEW'}</span></header>
<main><div class="cards"><article class="card"><span>Quality association · every frame</span><strong>{html.escape(pretty(selected))}</strong><small>Balanced validation HOTA {metric(validation_selected['balanced_HOTA'])}</small></article><article class="card"><span>Budget association · validated separately</span><strong>{html.escape(pretty(budget_profile))}</strong><small>Schedule: {html.escape(pretty(evidence.get('budget_schedule') or 'not selected in this snapshot'))}</small></article><article class="card"><span>Additional confirmation</span><strong>{len(confirmation)} clips · {frames} frames</strong><small>Nine additional non-insect categories</small></article><article class="card"><span>Exact-count schedule pairs</span><strong>{len(matching)}</strong><small>Inputs and association/output knobs checked</small></article></div>
{warnings_html}
<section><h2>The recorded outcome</h2><p class="lead">{html.escape(headline)}</p><p>{html.escape(selected_description)}</p><p class="note">There was no new detector training. Quality experiments replay fixed author-provided detections, simulated detections, or explicitly labelled oracle boxes. Detector-call reduction does not itself establish a live speedup.</p><p class="muted">{html.escape(regression_text)}</p></section>
<section><h2>Read the evidence by scope</h2><div class="scope"><article><strong>Validation / selection</strong>Six synthetic scenarios and a natural insect validation clip. Source weights and the raw-insect guard are declared in the protocol. Every failed candidate remains visible.</article><article><strong>Reused exploratory evidence</strong>Earlier insect and synthetic test clips were already inspected. Their refinement results are exploratory and are not relabelled as untouched test evidence.</article><article><strong>Additional confirmation</strong>Nine additional non-insect category clips were declared before scoring and evaluated after validation selection was frozen. One clip per category cannot establish recording-independent generalisation or full-benchmark performance.</article></div><p class="muted">Primary tables use <code>output_policy=active</code>: confirmed, non-lost estimates within the output horizon, including deliberate detector skips. Lost-output diagnostics remain separate.</p></section>
<section><h2>Validation chose two operating profiles</h2><p>The quality profile is selected with detection on every frame. {html.escape(evidence['selection_rule'])}. The permitted raw-insect HOTA minimum was {metric(evidence['raw_hota_minimum'])}. Every failed candidate remains visible.</p>{image_embed(output/'validation.png', 'Every declared quality-validation candidate, including strict-gating and hybrid failures.')}{html_table(headers,val_cells)}<h3>Budget association: a separate fixed-2 validation comparison</h3><p>The budget profile is {html.escape(pretty(budget_profile))}. Association is selected again under the declared every-second-frame schedule using the same source weights and raw-insect non-regression guard. Schedule selection then compares the declared budget options on validation only; neither choice uses additional confirmation scores.</p>{image_embed(output/'budget_validation.png', 'Association candidates under fixed-2 validation; this table is not mixed with every-frame quality selection.')}{html_table(headers,budget_cells) if budget_cells else '<div class="empty">Budget association selection is not present in this snapshot.</div>'}<p class="muted">Scores are unweighted sequence means inside each source group, followed by the declared source weights. They are not official pooled benchmark scores. Parameter calibration and method selection are not neural-network training.</p></section>
<section><h2>Additional category coverage</h2>{image_embed(output / "coverage.png", "Nine predeclared categories. GMOT-40 / Bai et al., CVPR2021 / CC BY-NC-SA4.0. Original frames; author detector caches, no new training.")}</section>
<section><h2>Additional confirmation results</h2><p>{html.escape(outcome)} Both methods use the same detection stream, frame map, active output policy and detector-every-frame schedule.</p>{image_embed(output/'confirmation.png', 'Per-clip frozen-baseline versus selected-profile HOTA and IDF1; all additional clips retained.')}{html_table(confirm_headers,confirm_cells) if confirm_cells else '<div class="empty">Confirmation has not been scored in this snapshot.</div>'}<p class="muted">Δ values are selected minus frozen baseline in score points. Mean scores are unweighted across clips; error counts are summed. These categories provide additional evidence, not insect-specific transfer proof.</p></section>
<section><h2>Budget profile on additional confirmation</h2>{html_table(budget_confirm_headers,budget_confirm_cells)}<p class="muted">Fixed2 holds detector availability fixed to isolate association. Adaptive versus fixed2 compares complete policies and can use different call counts; use the exact-count table below for a matched budget. No confirmation scores changed the recommendation.</p></section>
<section><h2>Scheduling: quality at measured detector-call rates</h2><p>Each panel keeps the validation-selected budget association ({html.escape(pretty(budget_profile))}) and active output rules fixed within the same complete clip cohort. Hollow points are content-independent uniform schedules given the adaptive method's final per-clip call count; this is a retrospective comparator.</p>{image_embed(output/'scheduling.png','Budget association is held fixed; different source groups and evidence phases are not pooled into one claimed gain.')}<details><summary>Every exact-count adaptive / uniform comparison</summary>{html_table(match_headers,match_cells)}</details><p class="muted">Missed-target counts measure availability of a selected detector box at score ≥0.08 and IoU ≥0.5, matched one-to-one. They are separate from correct track initialization and identity continuity. First annotated visibility is not verified physical birth; real-source discovery delay is reported in frames.</p></section>
<section><h2>Scene cues: activation is a diagnostic</h2><p>The robust cue uses prior-only recurring-motion histories, brightness correction and guarded small-translation alignment. Here legacy and robust cues share the same active-only risk scope and association settings.</p>{image_embed(output/'novelty.png','Fewer novelty triggers can reduce nuisance requests, but do not prove better tracking or fewer missed arrivals.')}{html_table(novelty_headers,novelty_cells) if novelty_cells else '<div class="empty">Paired scene-cue outputs are not yet present.</div>'}<p class="note warning">A new low-contrast object inside a repeatedly moving region can be suppressed. Nonuniform lighting, rotations, larger camera movement and adverse texture remain failure cases. The robust cue is a training-free heuristic, not a foreground-recognition guarantee.</p></section>
<section><h2>Reference implementations and runtime</h2><h3>Reference methods on additional confirmation clips</h3>{html_table(['Clip','Reference','HOTA','IDF1','IDSW','FP','FN'],ref_cells) if ref_cells else '<div class="empty">Optional reference-confirmation outputs are not present in this snapshot.</div>'}<p class="muted">Reference trackers retain their documented native lifecycle/emission rules. They are labelled separately from the controlled same-output-policy ablations. No tracker-only advertised FPS is compared against a detector-inclusive pipeline.</p><h3>Separately measured runtime</h3><p>{html.escape(evidence.get("runtime_scope", ""))}</p>{html_table(runtime_headers,runtime_cells) if runtime_cells else '<div class="empty">No separate runtime aggregate is present. Cached replay is not a live inference benchmark.</div>'}<p class="muted">Runtime values, when present, come only from the separately recorded runtime artifact. Source capture FPS may be nominal, and playback FPS is not processing performance.</p></section>
<section><h2>Processed / oracle diagnostics remain separate</h2><p>Monash bee JPEGs contain ROI masks and embedded counters. Their oracle boxes test association only. These results cannot establish natural-scene discovery or the detection accuracy of a newly trained model.</p><details><summary>Active-output processed/oracle diagnostic table</summary>{html_table(diagnostic_headers,diagnostic_cells) if diagnostic_cells else '<div class="empty">No processed/oracle diagnostics in this snapshot.</div>'}</details></section>
<section><h2>Comparison footage</h2><div class="video-grid">{''.join(videos)}</div><p class="muted">Videos retain failure frames and distinguish observations, predictions and tentative states. GMOT-derived imagery carries CC BY-NC-SA 4.0 attribution; code licensing does not relicense dataset footage.</p></section>
<section><h2>Inspect the underlying records</h2><p>All figures and primary tables are reproducible from the JSON and CSV exports. No network access is needed to read this page; plot pixels are embedded.</p><div class="downloads">{''.join(f'<a href="{name}">{label}</a>' for name,label in [('REPORT.md','Markdown report'),('evidence.json','Compact evidence JSON'),('validation.csv','Validation CSV'),('confirmation.csv','Confirmation CSV'),('matched_scheduling.csv','Matched scheduling CSV'),('novelty.csv','Scene-cue CSV'),('experiments.csv','All active-output records'),('diagnostic_all_output.csv','Separate all-output diagnostics'),('source_manifest.json','Source hashes')])}</div><details><summary>Filter all active-output experiment rows</summary><div class="filters"><input id="search" placeholder="Search clip, profile or schedule" aria-label="Search experiment rows"><select id="phase"><option value="">All evidence phases</option>{phase_options}</select><select id="source"><option value="">All source groups</option>{source_options}</select></div><p id="visible-count" class="muted"></p>{html_table(record_headers,record_cells,'filterable',record_attrs)}</details></section>
<footer>HOTA / IDF1 / AssA / DetA are displayed on a 0–100 scale. Negative MOTA is retained. Deterministic clip scores are not independent-frame statistical samples. See docs/SYSTEM_CARD.md for intended use and remaining limitations.</footer></main>
<script>const q=document.getElementById('search'),p=document.getElementById('phase'),s=document.getElementById('source');function filterRows(){{let n=0;const rows=document.querySelectorAll('.filterable tbody tr');for(const r of rows){{const ok=(!p.value||r.dataset.phase===p.value)&&(!s.value||r.dataset.source===s.value)&&r.textContent.toLowerCase().includes(q.value.toLowerCase());r.hidden=!ok;if(ok)n++}}document.getElementById('visible-count').textContent=n+' of '+rows.length+' active-output experiment rows shown.'}}for(const e of [q,p,s])e.addEventListener('input',filterRows);filterRows();</script></body></html>'''
    (output / "index.html").write_text(document, encoding="utf-8")
    sections = ["# Kinetic-Tracker: refinement evidence", "", "**" + ("Frozen confirmation report" if complete else "INCOMPLETE PREVIEW") + "**", "", headline, "",
                "No new detector training. Quality results replay fixed detections; cached replay is not live detector throughput.", "",
                "## Evidence scope", "", "Validation selection uses the declared source-balanced criterion. Earlier insect/synthetic tests are reused exploratory evidence. Nine additional non-insect category clips were declared before scoring and evaluated after freeze; one clip per category does not establish full-benchmark or recording-independent generalisation.", "",
                "## Validation candidates", "", evidence["selection_rule"], "", markdown_table(headers,val_cells), "", regression_text, "",
                "![Validation candidates](validation.png)", "", "## Budget association selection under fixed-2", "",
                f"Budget profile: {budget_profile}; schedule: {evidence.get('budget_schedule')}. This separate validation comparison uses fixed-2 detector availability and is not mixed with every-frame quality selection.", "",
                markdown_table(headers,budget_cells) if budget_cells else "Budget selection not yet present.", "",
                "## Additional confirmation", "", outcome, "",
                markdown_table(confirm_headers,confirm_cells) if confirm_cells else "Not present in this snapshot.", "",
                "Scores use active output, identical cache/timestamps and every-frame detection. Mean scores are unweighted; counts are summed. Δ is selected minus frozen baseline.", "",
                "## Budget profile on additional confirmation", "", markdown_table(budget_confirm_headers,budget_confirm_cells) if budget_confirm_cells else "Not present.", "", "Fixed2 isolates association at fixed detector availability. Adaptive versus fixed2 is a complete-policy comparison with potentially different call counts; exact-count controls follow.", "",
                "## Exact-count scheduling comparisons", "", "Every row uses the same per-sequence detector-call count, source inputs, association and output knobs. Uniform matching is retrospective and content-independent.", "",
                markdown_table(match_headers,match_cells) if match_cells else "Not present in this snapshot.", "",
                "## Scene-cue diagnostics", "", markdown_table(novelty_headers,novelty_cells) if novelty_cells else "Not present in this snapshot.", "",
                "A lower trigger rate is not a tracking gain. Recurring motion can hide weak new objects; nonuniform lighting and unmodelled camera movement remain limitations.", "",
                "## Reference methods", "", markdown_table(["Clip","Reference","HOTA","IDF1","IDSW","FP","FN"],ref_cells) if ref_cells else "Optional reference outputs not present.", "",
                "References keep their native emission/lifecycle rules; they are separate from the same-output-policy controlled ablations.", "",
                "## Runtime", "", evidence.get("runtime_scope", ""), "", markdown_table(runtime_headers,runtime_cells) if runtime_cells else "No separate runtime artifact present; no live speed claim.", "",
                "## Diagnostic-only data", "", "Monash source frames are processed with ROI masks and counters. Oracle boxes isolate association and do not establish detector or natural new-arrival performance.", "",
                "## Reproduction and limitations", "", "All primary results use output_policy=active; all-output diagnostics are exported separately. Real capture timing is unverified/nominal where stated; discovery is primary in frames. First annotated visibility is not physical birth. No novelty, publication, ranking or universal performance claim is made.", "",
                "See [compact evidence](evidence.json), [source hashes](source_manifest.json), [all active records](experiments.csv), and [system card](../../docs/SYSTEM_CARD.md)."]
    if warnings:
        sections.extend(["", "## Snapshot warnings", ""] + ["- " + warning for warning in warnings])
    (output / "REPORT.md").write_text("\n".join(sections)+"\n", encoding="utf-8")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, default=ROOT / "runs/refinement")
    parser.add_argument("--output", type=Path, default=ROOT / "results/refinement")
    parser.add_argument("--allow-incomplete", action="store_true", help="Label the page as a preview until freeze/confirmation exists")
    args = parser.parse_args(argv)
    try:
        runs, output = args.runs.resolve(), args.output.resolve()
        if output.is_relative_to(runs) or runs.is_relative_to(output):
            raise ValueError("report output must be separate from experiment inputs")
        selection, frozen, protocol, records, warnings, source_files, recommendation = collect(runs, args.allow_incomplete)
        selected = frozen["selected_profile"] if frozen else selection["selected_profile"]
        budget_profile = recommendation.get("budget_profile", selected) if recommendation else selected
        budget_selection = recommendation.get("budget_association_validation") if recommendation else None
        active = [record for record in records if record["output_policy"] == "active"]
        val = validation_rows(selection); confirmation = confirmation_rows(active, selected)
        budget_confirmation = budget_confirmation_rows(active, budget_profile, recommendation['budget_schedule']) if confirmation and recommendation else []
        matching = matched_rows(active, budget_profile); novelty = novelty_rows(active, budget_profile)
        groups = curve_groups(active, budget_profile)
        references, runtime, optional_sources = optional_evidence(runs); source_files.update(optional_sources)
        compact = [compact_record(record, record["_path"]) for record in active]
        all_output = [compact_record(record, record["_path"]) for record in records if record["output_policy"] == "all"]
        source_files.update({row["experiment"]: row["experiment_sha256"] for row in compact + all_output})
        evidence = {"schema": "kinetic-refinement-report-v1", "frozen": bool(frozen and frozen.get("frozen")),
                    "selected_profile": selected, "budget_profile": budget_profile,
                    "budget_schedule": recommendation.get("budget_schedule") if recommendation else None,
                    "budget_validation": validation_rows(budget_selection) if budget_selection else [],
                    "schedule_validation": recommendation.get("schedule_validation", {}) if recommendation else {},
                    "selection_rule": selection["selection"]["metric"],
                    "raw_hota_minimum": selection.get("raw_hota_minimum"), "validation": val,
                    "baseline_regression_checks": selection.get("baseline_v2_regression_checks", []),
                    "confirmation": confirmation, "budget_confirmation": budget_confirmation, "matched_scheduling": matching, "novelty": novelty,
                    "scheduling_curve": [point for _, _, points in groups for point in points],
                    "reference_confirmation": references, "runtime_measurements": flatten_numeric(runtime) if runtime else [],
                    "runtime_profiles": runtime.get('profiles', {}) if runtime else {},
                    "runtime_scope": runtime.get('scope', '') if runtime else '',
                    "experiments": compact, "separate_all_output_record_count": len(all_output),
                    "scope": protocol["scope"], "warnings": warnings,
                    "metric_scale": "Raw exported scores are normalized; displayed HOTA/IDF1/AssA/DetA/MOTA are multiplied by 100.",
                    "aggregation": "Unweighted clip means within displayed cohorts; counts summed; detector rate frame-weighted. Not official pooled scores."}
        output.mkdir(parents=True, exist_ok=True)
        plot_validation(val, selection, output / "validation.png")
        if budget_selection:
            plot_validation(evidence["budget_validation"], budget_selection, output / "budget_validation.png")
        plot_confirmation(confirmation, selected, output / "confirmation.png")
        plot_scheduling(groups, output / "scheduling.png")
        plot_novelty(novelty, output / "novelty.png")
        write_csv(output / "validation.csv", val); write_csv(output / "confirmation.csv", confirmation)
        write_csv(output / "budget_validation.csv", evidence["budget_validation"])
        write_csv(output / 'budget_confirmation.csv', budget_confirmation)
        write_csv(output / "matched_scheduling.csv", matching); write_csv(output / "novelty.csv", novelty)
        write_csv(output / "scheduling_curve.csv", evidence["scheduling_curve"])
        write_csv(output / "experiments.csv", public_experiment_rows(compact))
        write_csv(output / "diagnostic_all_output.csv", public_experiment_rows(all_output))
        write_csv(output / "reference_confirmation.csv", references)
        write_csv(output / "runtime.csv", evidence["runtime_measurements"])
        (output / "evidence.json").write_text(json.dumps(evidence, indent=2, allow_nan=False)+"\n", encoding="utf-8")
        manifest = {"reporter_sha256": sha(Path(__file__)), "sources": source_files,
                    "protocol_sha256": frozen["protocol_sha256"] if frozen else selection.get("protocol_sha256"),
                    "note": "Relative source paths and content hashes only; no command lines, environment variables or credentials are exported."}
        (output / "source_manifest.json").write_text(json.dumps(manifest, indent=2, allow_nan=False)+"\n", encoding="utf-8")
        render_report(output, evidence, protocol, warnings)
        print(json.dumps({"report": relative(output / "index.html"), "frozen": evidence["frozen"],
                          "selected_profile": selected, "active_records": len(active), "confirmation_clips": len(confirmation),
                          "confirmation_frames": sum(row["frames"] for row in confirmation), "matched_pairs": len(matching)}, indent=2))
    except (OSError, ValueError, KeyError, RuntimeError) as error:
        parser.exit(2, f"Refinement report error: {error}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
