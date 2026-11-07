#!/usr/bin/env python3
"""Render an honest side-by-side cached-tracking replay without detector execution.

Every source frame is retained. Observed and predicted boxes have different styles;
tentative tracks are identified explicitly. Videos are presentation artifacts, not
measurements of live inference throughput.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import csv

import cv2
import numpy as np


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_bool(value: str) -> bool:
    if value.lower() in {"1", "true"}:
        return True
    if value.lower() in {"0", "false"}:
        return False
    raise ValueError(f"invalid Boolean field: {value!r}")


@dataclass(frozen=True)
class Track:
    identity: int
    x: float
    y: float
    width: float
    height: float
    score: float
    state: str
    observed: bool


@dataclass
class Run:
    directory: Path
    label: str
    frames: dict
    tracks: dict
    summary: dict
    input_hashes: dict


def load_run(directory: Path, label: str, total_frames: int) -> Run:
    directory = directory.resolve()
    paths = {name: directory / name for name in ("summary.json", "frames.csv", "telemetry.csv")}
    if not all(path.is_file() for path in paths.values()):
        raise ValueError(f"run needs summary.json, frames.csv and telemetry.csv: {directory}")
    summary = json.loads(paths["summary.json"].read_text(encoding="utf-8"))
    if summary.get("mode") != "cached_detection_replay":
        raise ValueError("this visualizer labels cached detector replay; both input summaries must have mode=cached_detection_replay")
    if int(summary.get("frames", -1)) != total_frames:
        raise ValueError("run length differs from sequence metadata")
    frame_rows = {}
    with paths["frames.csv"].open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        if not {"frame", "timestamp", "detector_ran"}.issubset(reader.fieldnames or []):
            raise ValueError("frame log lacks frame,timestamp,detector_ran")
        for row in reader:
            frame = int(row["frame"]); timestamp = float(row["timestamp"])
            if frame in frame_rows or frame < 1 or frame > total_frames or not math.isfinite(timestamp):
                raise ValueError("frame log has duplicate, out-of-range or nonfinite entries")
            frame_rows[frame] = {"timestamp": timestamp, "detector_ran": parse_bool(row["detector_ran"])}
    if set(frame_rows) != set(range(1, total_frames+1)):
        raise ValueError("frame log must contain every source observation exactly once")
    observed_calls = sum(row["detector_ran"] for row in frame_rows.values())
    if int(summary.get("detector_calls", -1)) != observed_calls:
        raise ValueError("summary detector-call count disagrees with frame log")
    tracks = defaultdict(list)
    seen = set()
    with paths["telemetry.csv"].open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        required = {"frame", "id", "x", "y", "w", "h", "score", "state", "observed"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError("telemetry lacks required track fields")
        for row in reader:
            frame, identity = int(row["frame"]), int(row["id"])
            x, y, width, height, score = (float(row[key]) for key in ("x", "y", "w", "h", "score"))
            state, observed = row["state"], parse_bool(row["observed"])
            if frame not in frame_rows or identity < 0 or (frame, identity) in seen:
                raise ValueError("telemetry contains invalid frames/IDs or duplicate IDs within a frame")
            if not all(math.isfinite(v) for v in (x, y, width, height, score)) or width <= 0 or height <= 0 or not 0 <= score <= 1:
                raise ValueError("telemetry has invalid box geometry or confidence")
            if state not in {"tentative", "confirmed", "lost"}:
                raise ValueError(f"unrecognised track state: {state}")
            if observed and not frame_rows[frame]["detector_ran"]:
                raise ValueError("an observed box is recorded on a skipped detector frame")
            seen.add((frame, identity)); tracks[frame].append(Track(identity, x, y, width, height, score, state, observed))
    return Run(directory, label, frame_rows, tracks, summary, {name: sha256_file(path) for name, path in paths.items()})


def load_sequence(directory: Path) -> tuple[dict, list[Path], dict]:
    directory = directory.resolve()
    metadata_path = directory / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    count, width, height = (int(metadata[key]) for key in ("num_frames", "width", "height"))
    if min(count, width, height) <= 0:
        raise ValueError("invalid sequence dimensions/length")
    images_dir = (directory / metadata.get("images", "img1")).resolve()
    if not images_dir.is_relative_to(directory):
        raise ValueError("sequence image path escapes its directory")
    images = sorted(path for path in images_dir.iterdir() if path.suffix.lower() in {".jpg", ".jpeg", ".png"})
    if len(images) != count or any(path.stem != f"{index:06d}" for index, path in enumerate(images, 1)):
        raise ValueError("sequence images must match metadata and be named 000001..N without gaps")
    timestamps = {}
    times_path = (directory / metadata.get("timestamp_file", "frames.csv")).resolve()
    if not times_path.is_relative_to(directory):
        raise ValueError("sequence timestamp path escapes its directory")
    with times_path.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            frame, timestamp = int(row["frame"]), float(row["timestamp"])
            if frame in timestamps or not math.isfinite(timestamp):
                raise ValueError("invalid sequence timestamp map")
            timestamps[frame] = timestamp
    if set(timestamps) != set(range(1, count+1)):
        raise ValueError("timestamp map does not cover every source image")
    if any(timestamps[n] <= timestamps[n-1] for n in range(2, count+1)):
        raise ValueError("sequence timestamps must increase strictly")
    return metadata, images, timestamps


def id_colour(identity: int) -> tuple[int, int, int]:
    # Stable colours within each method; ID numbers are not correspondences between methods.
    hue = (identity * 47 + 19) % 180
    return tuple(int(x) for x in cv2.cvtColor(np.uint8([[[hue, 185, 235]]]), cv2.COLOR_HSV2BGR)[0, 0])


def text(image, value: str, origin, scale=.55, colour=(230, 235, 241), thickness=1, max_width=None):
    if max_width:
        while scale > .25 and cv2.getTextSize(value, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)[0][0] > max_width:
            scale -= .025
    cv2.putText(image, value, origin, cv2.FONT_HERSHEY_SIMPLEX, scale, colour, thickness, cv2.LINE_AA)


def dashed_rectangle(image, start, end, colour, dash=8):
    x0, y0 = start; x1, y1 = end
    for a in range(x0, x1+1, dash*2):
        cv2.line(image, (a, y0), (min(a+dash, x1), y0), colour, 2)
        cv2.line(image, (a, y1), (min(a+dash, x1), y1), colour, 2)
    for a in range(y0, y1+1, dash*2):
        cv2.line(image, (x0, a), (x0, min(a+dash, y1)), colour, 2)
        cv2.line(image, (x1, a), (x1, min(a+dash, y1)), colour, 2)


def render_panel(source, run: Run, frame: int, calls: int, total: int, panel_width: int, panel_height: int):
    image = cv2.resize(source, (panel_width, panel_height), interpolation=cv2.INTER_AREA)
    sx, sy = panel_width/source.shape[1], panel_height/source.shape[0]
    for track in run.tracks.get(frame, []):
        x0, y0 = int(round(track.x*sx)), int(round(track.y*sy))
        x1, y1 = int(round((track.x+track.width)*sx)), int(round((track.y+track.height)*sy))
        tentative = track.state == "tentative"
        colour = (155, 158, 164) if tentative else (70, 220, 100) if track.observed else (0, 190, 255)
        # Clamp line endpoints only for rendering; telemetry geometry is not altered.
        a = (max(-10, min(panel_width+10, x0)), max(-10, min(panel_height+10, y0)))
        b = (max(-10, min(panel_width+10, x1)), max(-10, min(panel_height+10, y1)))
        if tentative or not track.observed:
            dashed_rectangle(image, a, b, colour)
        else:
            cv2.rectangle(image, a, b, colour, 2)
        tag = ("TENTATIVE " if tentative else "") + f"#{track.identity} " + ("OBS" if track.observed else "PRED")
        if track.state == "lost":
            tag += " LOST"
        origin = (max(2, min(panel_width-15, x0)), max(14, min(panel_height-4, y0-4)))
        (tw, th), baseline = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, .38, 1)
        cv2.rectangle(image, (origin[0]-1, origin[1]-th-3), (min(panel_width-1, origin[0]+tw+4), origin[1]+baseline+2), (24, 28, 34), -1)
        cv2.line(image, (origin[0], origin[1]-th-2), (min(panel_width-1, origin[0]+tw), origin[1]-th-2), id_colour(track.identity), 3)
        text(image, tag, origin, .38, colour)
    panel = np.full((panel_height+110, panel_width, 3), (25, 29, 36), dtype=np.uint8)
    panel[110:] = image
    text(panel, run.label, (14, 29), .65, thickness=2, max_width=panel_width-28)
    detector = run.frames[frame]["detector_ran"]
    text(panel, ("DETECT" if detector else "PREDICT") + f"  |  calls {calls}/{frame} ({100*calls/frame:.1f}%)",
         (14, 58), .53, (70, 220, 100) if detector else (0, 190, 255), max_width=panel_width-28)
    tentative_count = sum(track.state == "tentative" for track in run.tracks.get(frame, []))
    confirmed_count = len(run.tracks.get(frame, []))-tentative_count
    text(panel, f"Frame {frame}/{total}  |  active {confirmed_count}  tentative {tentative_count}", (14, 86), .45, max_width=panel_width-28)
    return panel


def dataset_notice(metadata: dict) -> tuple[str, str, dict]:
    dataset = metadata.get("dataset", "unknown")
    if metadata.get("synthetic"):
        return "SYNTHETIC DIAGNOSTIC / SIMULATED DETECTIONS", "Generated Kinetic-Tracker diagnostic imagery; no external dataset footage", {
            "kind": "synthetic", "source": "tools/generate_synthetic.py", "license": "Repository source and generated synthetic assets: MIT"}
    if dataset == "gmot40":
        return "REAL FULL-SCENE IMAGES / FIXED DETECTION REPLAY", "GMOT-40 / Bai et al., CVPR 2021 / CC BY-NC-SA 4.0", {
            "kind": "real_author_dataset", "source": "https://spritea.github.io/GMOT40/download.html",
            "author_repository": "https://github.com/Spritea/GMOT40", "license": "CC BY-NC-SA 4.0",
            "license_url": "https://creativecommons.org/licenses/by-nc-sa/4.0/",
            "attribution": "Hexin Bai, Wensheng Cheng, Peng Chu, Kai Zhang and Haibin Ling. GMOT-40: A Benchmark for Generic Multiple Object Tracking. CVPR 2021.",
            "derivative_video_license": "CC BY-NC-SA 4.0", "original_watermarks": "Preserved in source frames."}
    if dataset == "native_bee" or metadata.get("processed_image_warning"):
        return "PROCESSED REAL IMAGES / ROI MASKS AND COUNTERS", "Monash Native Bee Dataset / Asaduzzaman et al., 2026 / CC BY 4.0", {
            "kind": "processed_real_dataset", "source": "https://doi.org/10.26180/32315067.v1", "license": "CC BY 4.0",
            "license_url": "https://creativecommons.org/licenses/by/4.0/",
            "attribution": "Asad Asaduzzaman, Malika Nisal Ratnayake, Alan Dorin and Vanessa Kellermann. Australian Native Bee Detection and Tracking Dataset. Monash University, 2026.",
            "limitation": "Source ROI boundaries and changing counters confound natural new-arrival discovery claims."}
    return "REAL IMAGES / FIXED DETECTION REPLAY / SOURCE TERMS UNVERIFIED", "Consult the source dataset licence before sharing this video", {
        "kind": "unknown_real_dataset", "license": "Unverified; consult source dataset."}


class VideoSink:
    def __init__(self, path: Path, width: int, height: int, fps: float, encoder="auto"):
        self.process = None; self.writer = None
        ffmpeg = shutil.which("ffmpeg") if encoder == "auto" else None
        if ffmpeg:
            command = [ffmpeg, "-hide_banner", "-loglevel", "error", "-nostdin", "-n", "-f", "rawvideo",
                       "-pix_fmt", "bgr24", "-s:v", f"{width}x{height}", "-r", f"{fps:.8g}", "-i", "pipe:0",
                       "-an", "-c:v", "libx264", "-preset", "fast", "-crf", "20", "-pix_fmt", "yuv420p",
                       "-movflags", "+faststart", str(path)]
            self.process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
                                            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
            self.encoder = "FFmpeg libx264 / yuv420p / CRF20 / faststart"
        else:
            self.writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
            if not self.writer.isOpened():
                raise RuntimeError("cannot create MP4 video; install FFmpeg from its official source or use a working OpenCV video backend")
            self.encoder = "OpenCV mp4v"

    def write(self, frame):
        if self.process:
            try:
                self.process.stdin.write(frame.tobytes())
            except BrokenPipeError as error:
                raise RuntimeError("FFmpeg stopped while encoding video") from error
        else:
            self.writer.write(frame)

    def close(self):
        if self.process:
            if not self.process.stdin.closed:
                self.process.stdin.close()
            errors = self.process.stderr.read().decode("utf-8", "replace")
            result = self.process.wait(timeout=60)
            self.process.stderr.close()
            if result:
                raise RuntimeError(f"FFmpeg video encoding failed: {errors.strip()}")
        elif self.writer:
            self.writer.release()

    def abort(self):
        if self.process:
            if self.process.poll() is None:
                self.process.kill()
            if not self.process.stdin.closed:
                try:
                    self.process.stdin.close()
                except (BrokenPipeError, OSError):
                    pass
            if not self.process.stderr.closed:
                self.process.stderr.close()
            self.process.wait(timeout=10)
        elif self.writer:
            self.writer.release()


def make_demo(sequence: Path, left_dir: Path, right_dir: Path, output: Path, left_label="Reference",
              right_label="Adaptive", panel_width=800, playback_fps=None, encoder="auto") -> dict:
    sequence, left_dir, right_dir, output = (Path(path).resolve() for path in (sequence, left_dir, right_dir, output))
    sidecar = output.with_suffix(".attribution.json")
    partial = output.with_name(output.stem+".partial.mp4")
    if output.suffix.lower() != ".mp4" or panel_width < 320 or panel_width > 1920 or panel_width % 2:
        raise ValueError("output must end in .mp4 and panel width must be even, between 320 and 1920")
    for destination in (output, sidecar, partial):
        if destination.exists():
            raise ValueError(f"refusing to overwrite existing output: {destination}")
        if any(destination.is_relative_to(source) for source in (sequence, left_dir, right_dir)):
            raise ValueError("demo outputs must be outside source sequence and experiment directories")
    metadata, image_paths, timestamps = load_sequence(sequence)
    count, width, height = (int(metadata[key]) for key in ("num_frames", "width", "height"))
    left = load_run(left_dir, left_label, count); right = load_run(right_dir, right_label, count)
    source_image_dir = (sequence / metadata.get("images", "img1")).resolve()
    for run in (left, right):
        if run.summary.get("input_images") and Path(run.summary["input_images"]).resolve() != source_image_dir:
            raise ValueError("run source image directory differs from the requested sequence")
    for frame in range(1, count+1):
        if any(abs(run.frames[frame]["timestamp"]-timestamps[frame]) > 1e-6 for run in (left, right)):
            raise ValueError("run timestamp map differs from the source sequence")
    fps = float(playback_fps if playback_fps is not None else metadata.get("fps", 25))
    if not math.isfinite(fps) or fps <= 0 or fps > 120:
        raise ValueError("playback FPS must be finite and between 0 and 120")
    panel_height = int(round(panel_width*height/width)); panel_height += panel_height % 2
    frame_width, frame_height = panel_width*2+12, panel_height+110+112
    notice, credit, attribution = dataset_notice(metadata)
    metadata_hash = sha256_file(sequence / "metadata.json")
    source_times_path = (sequence / metadata.get("timestamp_file", "frames.csv")).resolve()
    source_times_hash = sha256_file(source_times_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    source_hash = hashlib.sha256(); sink = None
    try:
        sink = VideoSink(partial, frame_width, frame_height, fps, encoder)
        encoder_name = sink.encoder
        calls = [0, 0]
        for frame, image_path in enumerate(image_paths, 1):
            image_bytes = image_path.read_bytes(); source_hash.update(image_bytes)
            source = cv2.imdecode(np.frombuffer(image_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
            if source is None or source.shape[:2] != (height, width):
                raise ValueError(f"source frame dimensions/decoding differ: {image_path}")
            canvas = np.full((frame_height, frame_width, 3), (17, 20, 25), dtype=np.uint8)
            for index, run in enumerate((left, right)):
                calls[index] += int(run.frames[frame]["detector_ran"])
                panel = render_panel(source, run, frame, calls[index], count, panel_width, panel_height)
                x = index*(panel_width+12); canvas[:panel.shape[0], x:x+panel_width] = panel
            y = panel_height+110
            text(canvas, f"{metadata.get('dataset', 'dataset')} / {metadata.get('name', sequence.name)} / {notice}", (14, y+24), .48, max_width=frame_width-28)
            text(canvas, "Green: observed | Amber dashed: predicted | Gray: tentative | ID numbers are independent per method", (14, y+49), .46, max_width=frame_width-28)
            text(canvas, f"CACHED DETECTOR REPLAY - NOT LIVE LATENCY | Playback {fps:g} FPS; source timing may be nominal", (14, y+73), .47, (110, 205, 250), max_width=frame_width-28)
            text(canvas, credit, (14, y+98), .43, max_width=frame_width-28)
            sink.write(canvas)
        sink.close(); sink = None
        if metadata.get("image_content_sha256") and source_hash.hexdigest() != metadata["image_content_sha256"]:
            raise ValueError("source image bytes differ from sequence provenance")
        for run in (left, right):
            for name, expected_hash in run.input_hashes.items():
                if sha256_file(run.directory / name) != expected_hash:
                    raise ValueError("experiment input changed while rendering; freeze runs before producing a demo")
        if sha256_file(sequence / "metadata.json") != metadata_hash:
            raise ValueError("sequence metadata changed while rendering")
        if sha256_file(source_times_path) != source_times_hash:
            raise ValueError("source timestamp map changed while rendering")
        capture = cv2.VideoCapture(str(partial)); decoded = 0
        while True:
            valid, rendered = capture.read()
            if not valid:
                break
            if rendered.shape[:2] != (frame_height, frame_width):
                capture.release(); raise ValueError("encoded demo dimensions differ")
            decoded += 1
        capture.release()
        if decoded != count:
            raise ValueError(f"encoded video lost frames: expected {count}, decoded {decoded}")
        # A hard link creates the final name without replacing a concurrently created
        # output. The fallback also uses exclusive creation rather than overwriting.
        try:
            os.link(partial, output)
        except FileExistsError:
            raise ValueError("output appeared during rendering; refusing to overwrite it")
        except OSError:
            with output.open("xb") as target, partial.open("rb") as source_stream:
                shutil.copyfileobj(source_stream, target)
        partial.unlink()
        result = {
            "kind": "cached_tracking_comparison_video", "sequence": str(sequence), "sequence_name": metadata.get("name", sequence.name),
            "source_metadata_sha256": metadata_hash, "source_image_content_sha256": source_hash.hexdigest(),
            "source_frame_times_sha256": source_times_hash,
            "output": str(output), "output_sha256": sha256_file(output), "frames": count,
            "width": frame_width, "height": frame_height, "playback_fps": fps,
            "encoder": encoder_name,
            "all_source_frames_retained": True, "decoded_output_frames_verified": decoded,
            "detector_execution": "none; this tool only renders saved tracking outputs",
            "timing_claim": "Playback rate and cached replay do not measure live inference latency or source capture FPS.",
            "attribution": attribution,
            "methods": [{"label": run.label, "directory": str(run.directory), "input_hashes": run.input_hashes,
                         "detector_calls": calls[index], "total_frames": count,
                         "schedule": run.summary.get("schedule"), "association": run.summary.get("association")} for index, run in enumerate((left, right))],
        }
        with sidecar.open("x", encoding="utf-8") as stream:
            stream.write(json.dumps(result, indent=2, allow_nan=False)+"\n")
        return result
    except BaseException:
        if sink is not None:
            sink.abort()
        if partial.exists() and partial.parent == output.parent:
            partial.unlink()
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sequence", required=True, type=Path)
    parser.add_argument("--left", required=True, type=Path)
    parser.add_argument("--right", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--left-label", default="Reference")
    parser.add_argument("--right-label", default="Adaptive")
    parser.add_argument("--panel-width", type=int, default=800)
    parser.add_argument("--playback-fps", type=float)
    parser.add_argument("--encoder", choices=["auto", "opencv"], default="auto")
    args = parser.parse_args()
    try:
        result = make_demo(args.sequence, args.left, args.right, args.output, args.left_label,
                           args.right_label, args.panel_width, args.playback_fps, args.encoder)
    except (OSError, ValueError, RuntimeError, KeyError) as error:
        parser.exit(1, f"Demo error: {error}\n")
    print(json.dumps({"output": result["output"], "frames": result["frames"], "sha256": result["output_sha256"]}, indent=2))


if __name__ == "__main__":
    main()
