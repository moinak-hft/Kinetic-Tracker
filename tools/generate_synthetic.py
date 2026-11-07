#!/usr/bin/env python3
"""Deterministic synthetic MOT stress tests with rendered frames and noisy detections.

These data diagnose tracking/scheduling behaviour. The supplied detection stream is
generated from a simulator, not a learned detector, and is never real-world evidence.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import csv
import hashlib
import json
import math
from pathlib import Path
import random
from typing import Callable

from PIL import Image, ImageDraw

SCENARIOS = ("easy", "crossings", "abrupt_turns", "occlusion", "births", "background_motion")
GENERATOR_VERSION = "kinetic-synthetic-v1"


@dataclass
class ObjectPath:
    identity: int
    born: int
    died: int
    width: float
    height: float
    position: Callable[[int], tuple[float, float]]


def overlap(a, b):
    x, y = max(a[0], b[0]), max(a[1], b[1])
    right, bottom = min(a[0]+a[2], b[0]+b[2]), min(a[1]+a[3], b[1]+b[3])
    return max(0, right-x) * max(0, bottom-y)


def make_paths(scenario: str, seed: int, frames: int, width: int, height: int) -> list[ObjectPath]:
    rng = random.Random(seed)
    paths = []
    sx, sy = width / 640, height / 360

    def linear(x, y, vx, vy, phase=0.0):
        return lambda frame: ((x+vx*(frame-1))*sx, (y+vy*(frame-1)+2*math.sin((frame-1)/19+phase))*sy)

    if scenario == "crossings":
        speed = 460 / max(1, frames-1)
        for identity, (x, y, vx) in enumerate(((90, 165, speed), (550, 169, -speed), (90, 225, speed), (550, 221, -speed)), 1):
            paths.append(ObjectPath(identity, 1, frames, 18*sx, 13*sy, linear(x, y+rng.uniform(-2, 2), vx, 0, identity)))
    elif scenario == "abrupt_turns":
        for identity in range(1, 5):
            phase = rng.uniform(0, 2*math.pi)
            x0, y0 = 110+identity*85, 65+identity*48
            def turn(frame, x0=x0, y0=y0, phase=phase, identity=identity):
                t = (frame-1)/max(1, frames-1)
                # Continuous positions with abrupt velocity reversals.
                triangle = 1-abs(((t*5+phase/6) % 2)-1)
                return ((x0+95*(triangle-.5))*sx, (y0+28*math.sin(t*10+phase)+14*(t>.55))*sy)
            paths.append(ObjectPath(identity, 1, frames, 17*sx, 12*sy, turn))
    elif scenario == "occlusion":
        speed = 470/max(1, frames-1)
        for identity in range(1, 4):
            paths.append(ObjectPath(identity, 1, frames, 19*sx, 13*sy, linear(70, 70+identity*66, speed, 0, identity)))
    else:
        count = 2 if scenario == "births" else 4
        for identity in range(1, count+1):
            x0, y0 = 70+identity*95, 40+identity*52
            speed = rng.uniform(.55, 1.1) * 180/max(1, frames)
            paths.append(ObjectPath(identity, 1, frames, rng.uniform(15, 21)*sx, rng.uniform(10, 15)*sy,
                                    linear(x0, y0, speed, rng.uniform(-.10, .10), identity)))
        if scenario == "births":
            for index in range(10):
                born = max(2, min(frames-5, int(frames*(.10+.076*index))+rng.randint(-2, 2)))
                life = rng.randint(6, 13)
                x0, y0 = rng.uniform(45, 585), rng.uniform(45, 315)
                vx, vy = rng.choice((-1, 1))*rng.uniform(1, 2.2), rng.uniform(-.8, .8)
                def short(frame, born=born, x0=x0, y0=y0, vx=vx, vy=vy):
                    return ((x0+vx*(frame-born))*sx, (y0+vy*(frame-born))*sy)
                paths.append(ObjectPath(index+3, born, min(frames, born+life-1), 13*sx, 10*sy, short))
    return paths


def make_background(width: int, height: int, seed: int) -> Image.Image:
    rng = random.Random(seed+7919)
    image = Image.new("RGB", (width, height), (103, 122, 94))
    draw = ImageDraw.Draw(image)
    for _ in range(160):
        x, y = rng.randrange(width), rng.randrange(height)
        radius = rng.randrange(2, 13)
        colour = (rng.randrange(82, 120), rng.randrange(105, 142), rng.randrange(72, 108))
        draw.ellipse((x-radius*2, y-radius, x+radius*2, y+radius), fill=colour)
    for i in range(8):
        x = int(width*(i+.5)/8)
        draw.line((x, height, x+int(width*.07), 0), fill=(91, 111, 78), width=max(1, width//320))
    return image


def generate_sequence(output: Path, scenario: str, seed: int, frames: int = 180,
                      width: int = 640, height: int = 360, fps: float = 25) -> dict:
    if scenario not in SCENARIOS or frames < 12 or width < 64 or height < 64 or fps <= 0:
        raise ValueError("invalid synthetic sequence configuration")
    rng = random.Random(seed+15485863)
    paths = make_paths(scenario, seed, frames, width, height)
    background = make_background(width, height, seed)
    image_dir = output / "img1"; image_dir.mkdir(parents=True, exist_ok=True)
    (output / "gt").mkdir(parents=True, exist_ok=True)
    gt_rows, detection_rows, frame_rows = [], [], []
    born_log = {obj.identity: obj.born for obj in paths}
    image_hash = hashlib.sha256()
    for frame in range(1, frames+1):
        timestamp = (frame-1)/fps
        image = background.copy()
        if scenario == "background_motion":
            shadow = Image.new("RGBA", (width, height))
            sd = ImageDraw.Draw(shadow)
            shadow_x = int((frame*3.5) % (width+width//3))-width//3
            sd.rectangle((shadow_x, 0, shadow_x+width//3, height), fill=(22, 30, 18, 85))
            image = Image.alpha_composite(image.convert("RGBA"), shadow).convert("RGB")
        draw = ImageDraw.Draw(image)
        boxes = []
        for obj in paths:
            if not obj.born <= frame <= obj.died:
                continue
            cx, cy = obj.position(frame)
            box = (cx-obj.width/2, cy-obj.height/2, obj.width, obj.height)
            if box[0]+box[2] <= 0 or box[1]+box[3] <= 0 or box[0] >= width or box[1] >= height:
                continue
            boxes.append((obj, box))
            inset = min(2.0, box[3]*.45)
            draw.ellipse((box[0]-2, box[1]+inset, box[0]+box[2]+2, box[1]+box[3]-inset), fill=(174, 175, 153))
            draw.ellipse((box[0], box[1], box[0]+box[2], box[1]+box[3]), fill=(43, 39, 30), outline=(25, 26, 20))
            draw.line((cx, box[1]+1, cx, box[1]+box[3]-1), fill=(91, 73, 35), width=max(1, width//320))
        occluder = None
        if scenario == "occlusion":
            occluder = (width*.46, 0, width*.075, height)
            draw.rectangle((occluder[0], 0, occluder[0]+occluder[2], height), fill=(74, 88, 56))
        for obj, box in boxes:
            visible = 1.0 if occluder is None else max(0.0, 1-overlap(box, occluder)/(box[2]*box[3]))
            # Fully hidden instances are not observable ground-truth objects in this
            # diagnostic; their identity resumes unchanged when visible again.
            if visible > .05:
                gt_rows.append((frame, obj.identity, *[f"{x:.6f}" for x in box], 1, 1, f"{visible:.6f}"))
            crowded = any(other.identity != obj.identity and overlap(box, other_box) > box[2]*box[3]*.08
                          for other, other_box in boxes)
            miss_probability = .02 if scenario == "easy" else .055
            if scenario == "abrupt_turns" and (frame+obj.identity*5) % 47 in (0, 1, 2):
                miss_probability = .75
            if visible < .75:
                miss_probability = max(miss_probability, 1-visible)
            if visible <= .05 or rng.random() < miss_probability:
                continue
            low_confidence = crowded or visible < .80 or (frame+obj.identity*7) % 29 in (0, 1)
            score = rng.uniform(.13, .31) if low_confidence else rng.uniform(.72, .97)
            noise = .3 if scenario == "easy" else .7
            noisy = (box[0]+rng.gauss(0, noise), box[1]+rng.gauss(0, noise),
                     max(3, box[2]+rng.gauss(0, noise*.4)), max(3, box[3]+rng.gauss(0, noise*.4)))
            detection_rows.append((frame, f"{timestamp:.8f}", *[f"{x:.6f}" for x in noisy], f"{score:.6f}", 0))
        if scenario != "easy" and rng.random() < .06:
            detection_rows.append((frame, f"{timestamp:.8f}", f"{rng.uniform(0,width-20):.6f}",
                                   f"{rng.uniform(0,height-15):.6f}", "16", "11", f"{rng.uniform(.10,.48):.6f}", 0))
        image_file = image_dir / f"{frame:06d}.jpg"
        image.save(image_file, quality=88, subsampling=0)
        image_hash.update(image_file.read_bytes())
        frame_rows.append((frame, f"{timestamp:.8f}", frame))
    with (output / "gt" / "gt.txt").open("w", newline="", encoding="utf-8") as stream:
        csv.writer(stream).writerows(gt_rows)
    with (output / "detections.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream); writer.writerow(["frame", "timestamp", "x", "y", "w", "h", "score", "class_id"])
        writer.writerows(detection_rows)
    with (output / "frames.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream); writer.writerow(["frame", "timestamp", "source_frame"]); writer.writerows(frame_rows)
    metadata = {
        "name": output.name, "dataset": "synthetic", "synthetic": True, "scenario": scenario,
        "generator": GENERATOR_VERSION, "seed": seed, "num_frames": frames, "width": width, "height": height, "fps": fps,
        "timestamp_file": "frames.csv", "ground_truth": "gt/gt.txt", "images": "img1", "detections": "detections.csv",
        "gt_rows": len(gt_rows), "detections_rows": len(detection_rows), "objects": len(paths),
        "object_birth_frames": born_log, "image_content_sha256": image_hash.hexdigest(),
        "detection_sha256": hashlib.sha256((output / "detections.csv").read_bytes()).hexdigest(),
        "gt_sha256": hashlib.sha256((output / "gt" / "gt.txt").read_bytes()).hexdigest(),
        "limitations": ["Rendered artificial insects and vegetation, not captured footage.",
                        "Cached detections are independently perturbed simulator boxes, not neural inference.",
                        "Validation/test seeds differ within the same scenario families; this is not unseen-family generalisation.",
                        "No future or current skipped-frame detections may be used as scheduler inputs."]}
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    (output / "seqinfo.ini").write_text(f"[Sequence]\nname={output.name}\nimDir=img1\nframeRate={fps:g}\nseqLength={frames}\nimWidth={width}\nimHeight={height}\nimExt=.jpg\n", encoding="utf-8")
    return metadata


def generate_dataset(output: Path, frames: int = 180, width: int = 640, height: int = 360,
                     fps: float = 25, splits_path: Path | None = None) -> dict:
    manifest = {"generator": GENERATOR_VERSION, "synthetic": True, "sequences": [],
                "validation_seed_base": 17011, "test_seed_base": 29009}
    for split, seed_base in (("validation", 17011), ("test", 29009)):
        for index, scenario in enumerate(SCENARIOS):
            sequence_dir = output / split / scenario
            meta = generate_sequence(sequence_dir, scenario, seed_base+index*101, frames, width, height, fps)
            meta["split"] = split
            (sequence_dir / "metadata.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
            manifest["sequences"].append({"path": sequence_dir.as_posix(), "split": split,
                                           **{key: meta[key] for key in ("scenario", "seed", "num_frames", "gt_sha256", "detection_sha256", "image_content_sha256")}})
            print(f"Generated {sequence_dir}: {frames} frames, {meta['objects']} objects", flush=True)
    output.mkdir(parents=True, exist_ok=True)
    (output / "provenance.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    if splits_path is not None:
        splits_path.parent.mkdir(parents=True, exist_ok=True)
        previous = json.loads(splits_path.read_text(encoding="utf-8")) if splits_path.exists() else {}
        previous["synthetic"] = {
            "policy": "Independent deterministic validation/test seeds within the same six scenario families. No synthetic training is performed.",
            **{split: [item["path"] for item in manifest["sequences"] if item["split"] == split] for split in ("validation", "test")}}
        splits_path.write_text(json.dumps(previous, indent=2) + "\n", encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("data/synthetic"))
    parser.add_argument("--frames", type=int, default=180)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=360)
    parser.add_argument("--fps", type=float, default=25)
    parser.add_argument("--splits", type=Path, default=Path("configs/splits.json"))
    args = parser.parse_args()
    if args.frames < 12 or args.width < 64 or args.height < 64 or args.fps <= 0:
        parser.error("frames>=12, dimensions>=64 and fps>0 are required")
    generate_dataset(args.output, args.frames, args.width, args.height, args.fps, args.splits)


if __name__ == "__main__":
    main()
