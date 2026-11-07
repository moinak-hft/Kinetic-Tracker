#!/usr/bin/env python3
"""Download the Monash bee data and convert identity labels to MOT format.

Downloads are data only. Archive paths, checksums, file types and sizes are checked;
no downloaded script, installer or model is executed by this tool.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
from pathlib import Path, PurePosixPath
import re
import shutil
import stat
import struct
import urllib.parse
import urllib.request
import zipfile

ARTICLE = "https://api.figshare.com/v2/articles/32315067"
SOURCE = "https://bridges.monash.edu/articles/dataset/Australian_Native_Bee_Detection_and_Tracking_Dataset/32315067"
ARCHIVE_URL = "https://ndownloader.figshare.com/files/64894986"
ARCHIVE_NAME = "native-bee-tracking-dataset.zip"
ARCHIVE_BYTES = 107325128
ARCHIVE_MD5 = "9c87d57716c879c2b569196ba4197021"
ARCHIVE_SHA256 = "fb79d8bb77cdedd394713d56396755314fa6446ecc2795dbce62a9da3109cd8b"
PREFIX = "data/tracking/clean_tracking/"
USER_AGENT = "KineticTracker-reproducible-data/1.0"
GMOT_IMAGE_URL = "https://github.com/Spritea/GMOT40/releases/download/v0.1/GenericMOT_JPEG_Sequence.zip"
GMOT_TOTAL_BYTES = 1498450771
GMOT_DIRECTORY_OFFSET = 1497536451
GMOT_DIRECTORY_BYTES = 914298
GMOT_IMAGE_START, GMOT_IMAGE_END = 876055443, 990099285
GMOT_FILES = {
    "gmot40_central.bin": (GMOT_IMAGE_URL, GMOT_DIRECTORY_BYTES, "fb253779df8abc366aabe3134b847544603f8c681f3a6025975ad512681741d3", (GMOT_DIRECTORY_OFFSET, GMOT_DIRECTORY_OFFSET+GMOT_DIRECTORY_BYTES-1)),
    "gmot40_insect_images.range": (GMOT_IMAGE_URL, GMOT_IMAGE_END-GMOT_IMAGE_START+1, "be046b10e7d1e8bbc01066624f44731ced26f1c72a86a1fa45fbb9f4c3d2c513", (GMOT_IMAGE_START, GMOT_IMAGE_END)),
    "gmot40_track_label.zip": ("https://github.com/Spritea/GMOT40/releases/download/v0.1/track_label.zip", 1978567, "aecbca612d696ecd6a536aa381ac45de2d6a57430520fd8418ec15376c0e6515", None),
    "gmot40_detection_label.zip": ("https://github.com/Spritea/GMOT40/releases/download/v0.1/detecion_label.zip", 1227411, "344be91c6112f13ee9fa50d43f47a60c4b7a02f9acd4951d5b976ff1dcefaa31", None),
}
SPLIT_DATES = {
    "train": ["20240411", "20240412", "20240423"],
    "validation": ["20240502", "20240531"],
    "test": ["20240420", "20240505"],
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def safe_destination(root: Path, member: str) -> Path:
    """Resolve a portable ZIP member without traversal, drive or UNC escapes."""
    portable = member.replace("\\", "/")
    parts = PurePosixPath(portable).parts
    if (not parts or portable.startswith("/") or any(p in ("..", ".") or ":" in p for p in parts)):
        raise ValueError(f"unsafe archive path: {member!r}")
    destination = root.joinpath(*parts).resolve()
    if not destination.is_relative_to(root.resolve()):
        raise ValueError(f"archive path leaves destination: {member!r}")
    return destination


def _trusted_download_url(url: str) -> bool:
    parsed = urllib.parse.urlparse(url)
    return parsed.scheme == "https" and parsed.hostname in {
        "ndownloader.figshare.com", "api.figshare.com", "au-east.erc.monash.edu.au"
    }


class TrustedRedirects(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        if not _trusted_download_url(newurl):
            raise ValueError(f"download redirected outside trusted Figshare/Monash hosts: {newurl}")
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def verify_archive(path: Path) -> dict:
    if path.stat().st_size != ARCHIVE_BYTES:
        raise ValueError(f"archive size differs: expected {ARCHIVE_BYTES}, got {path.stat().st_size}")
    md5 = hashlib.md5()
    sha = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            md5.update(chunk)
            sha.update(chunk)
    if md5.hexdigest() != ARCHIVE_MD5 or sha.hexdigest() != ARCHIVE_SHA256:
        raise ValueError("archive checksum differs from the pinned Monash/Figshare release")
    return {"bytes": path.stat().st_size, "md5": md5.hexdigest(), "sha256": sha.hexdigest()}


def download_archive(destination: Path) -> dict:
    destination.parent.mkdir(parents=True, exist_ok=True)
    opener = urllib.request.build_opener(TrustedRedirects())
    if not destination.exists():
        partial = destination.with_suffix(".part")
        request = urllib.request.Request(ARCHIVE_URL, headers={"User-Agent": USER_AGENT})
        with opener.open(request, timeout=90) as response, partial.open("wb") as output:
            if not _trusted_download_url(response.url):
                raise ValueError("untrusted final download host")
            total = 0
            for chunk in iter(lambda: response.read(1024 * 1024), b""):
                total += len(chunk)
                if total > ARCHIVE_BYTES:
                    raise ValueError("download exceeds pinned archive size")
                output.write(chunk)
        verify_archive(partial)
        partial.replace(destination)
    verified = verify_archive(destination)
    request = urllib.request.Request(ARTICLE, headers={"User-Agent": USER_AGENT})
    try:
        with opener.open(request, timeout=30) as response:
            metadata = json.load(response)
        destination.with_name("native_bee_figshare.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    except (OSError, ValueError) as error:
        # Checksums pin the data even if the descriptive metadata service is offline.
        print(f"Metadata refresh unavailable: {error}")
    return verified


def parse_yolo_identity(line: str, width: int, height: int) -> tuple[str, float, float, float, float]:
    parts = line.split()
    if len(parts) != 5 or not re.fullmatch(r"bee_\d+", parts[0]):
        raise ValueError(f"invalid identity annotation: {line!r}")
    cx, cy, bw, bh = map(float, parts[1:])
    if not all(math.isfinite(x) for x in (cx, cy, bw, bh)):
        raise ValueError("nonfinite annotation")
    if not (0 <= cx <= 1 and 0 <= cy <= 1 and 0 < bw <= 1 and 0 < bh <= 1):
        raise ValueError(f"normalised annotation outside valid range: {line!r}")
    # Preserve the source box, including any extent just outside an image boundary.
    return parts[0], (cx - bw / 2) * width, (cy - bh / 2) * height, bw * width, bh * height


def date_split(sequence: str) -> str:
    match = re.fullmatch(r"Camera_\d+_(\d{8})_\d{6}", sequence)
    if not match:
        raise ValueError(f"unrecognised source sequence name: {sequence}")
    for split, dates in SPLIT_DATES.items():
        if match.group(1) in dates:
            return split
    return "excluded"


def write_seqinfo(directory: Path, sequence: str, frames: int, width: int, height: int, fps: float) -> None:
    (directory / "seqinfo.ini").write_text(
        f"[Sequence]\nname={sequence}\nimDir=img1\nframeRate={fps:g}\nseqLength={frames}\n"
        f"imWidth={width}\nimHeight={height}\nimExt=.jpg\n", encoding="utf-8")


def convert_archive(archive: Path, output: Path, splits_path: Path) -> dict:
    from PIL import Image

    checksums = verify_archive(archive)
    output.mkdir(parents=True, exist_ok=True)
    sequences = {}
    excluded = []
    source_member_hashes = {}
    with zipfile.ZipFile(archive) as package:
        members = {}
        if sum(item.file_size for item in package.infolist()) > 512 * 1024 * 1024:
            raise ValueError("archive uncompressed size exceeds expected bound")
        for item in package.infolist():
            safe_destination(output, item.filename)
            if stat.S_ISLNK(item.external_attr >> 16):
                raise ValueError("archive contains a symbolic link")
            if item.is_dir():
                continue
            if item.filename in members:
                raise ValueError("duplicate ZIP entry")
            if Path(item.filename).suffix.lower() not in {".jpg", ".txt"}:
                raise ValueError(f"unexpected dataset file type: {item.filename}")
            members[item.filename] = item
        grouped = {}
        for member in members:
            if not member.startswith(PREFIX) or "/images/" not in member or not member.endswith(".jpg"):
                continue
            rest = member[len(PREFIX):].split("/")
            if len(rest) != 3:
                raise ValueError("unexpected dataset directory shape")
            sequence = rest[0]
            frame_number = int(re.search(r"_frame_(\d+)\.jpg$", rest[2]).group(1))
            grouped.setdefault(sequence, []).append((frame_number, member))
        for sequence, frames in sorted(grouped.items()):
            frames.sort()
            if len(frames) < 20:
                excluded.append({"sequence": sequence, "available_frames": len(frames), "reason": "fewer than 20 available frames"})
                continue
            split = date_split(sequence)
            if split == "excluded":
                raise ValueError(f"sequence has no predeclared date split: {sequence}")
            destination = safe_destination(output, sequence)
            (destination / "img1").mkdir(parents=True, exist_ok=True)
            (destination / "gt").mkdir(parents=True, exist_ok=True)
            label_records = []
            valid_frames = []
            quarantined = []
            source_ids = set()
            width, height = 1280, 720
            for source_frame, member in frames:
                label_member = member.replace("/images/", "/labels/").removesuffix(".jpg") + ".txt"
                if label_member not in members:
                    raise ValueError(f"image missing identity labels: {member}")
                labels = package.read(label_member).decode("utf-8-sig")
                parsed = [parse_yolo_identity(line, width, height) for line in labels.splitlines() if line.strip()]
                identities = [p[0] for p in parsed]
                source_member_hashes[label_member] = hashlib.sha256(labels.encode()).hexdigest()
                if len(identities) != len(set(identities)):
                    quarantined.append({"source_frame": source_frame, "source_member": label_member,
                                        "reason": "Two distinct annotations use the same identity in this frame.",
                                        "duplicate_ids": sorted({identity for identity in identities if identities.count(identity) > 1})})
                    continue
                source_ids.update(identities)
                label_records.append(parsed)
                valid_frames.append((source_frame, member))
            frames = valid_frames
            id_map = {identity: index + 1 for index, identity in enumerate(sorted(source_ids, key=lambda x: int(x.split("_")[-1])))}
            gt_lines = []
            oracle_rows = []
            map_rows = []
            image_digest = hashlib.sha256()
            for index, ((source_frame, member), labels) in enumerate(zip(frames, label_records), 1):
                image_bytes = package.read(member)
                with Image.open(io.BytesIO(image_bytes)) as image:
                    if image.size != (width, height):
                        raise ValueError(f"unexpected source image size: {member}: {image.size}")
                    image.verify()
                (destination / "img1" / f"{index:06d}.jpg").write_bytes(image_bytes)
                image_digest.update(image_bytes)
                timestamp = (source_frame - frames[0][0]) / 25.0
                map_rows.append((index, f"{timestamp:.8f}", source_frame, member))
                for identity, x, y, w, h in labels:
                    gt_lines.append(f"{index},{id_map[identity]},{x:.6f},{y:.6f},{w:.6f},{h:.6f},1,1,1")
                    oracle_rows.append((index, f"{timestamp:.8f}", f"{x:.6f}", f"{y:.6f}", f"{w:.6f}", f"{h:.6f}", "1.0", 0))
            (destination / "gt" / "gt.txt").write_text("\n".join(gt_lines) + "\n", encoding="utf-8")
            with (destination / "frames.csv").open("w", newline="", encoding="utf-8") as stream:
                writer = csv.writer(stream)
                writer.writerow(["frame", "timestamp", "source_frame", "source_member"])
                writer.writerows(map_rows)
            with (destination / "oracle_detections.csv").open("w", newline="", encoding="utf-8") as stream:
                writer = csv.writer(stream)
                writer.writerow(["frame", "timestamp", "x", "y", "w", "h", "score", "class_id"])
                writer.writerows(oracle_rows)
            gaps = sum(b[0] - a[0] - 1 for a, b in zip(frames, frames[1:]))
            metadata = {
                "name": sequence, "dataset": "native_bee", "split": split, "synthetic": False,
                "num_frames": len(frames), "width": width, "height": height, "fps": 25,
                "timestamp_file": "frames.csv", "source_first_frame": frames[0][0], "source_last_frame": frames[-1][0],
                "timestamp_basis": "Nominal published 25 FPS and filename indices; sensor timing is not independently verified.",
                "processed_image_warning": "Source images contain black masked backgrounds, insect ROI patches and embedded counters. ROI boundaries confound natural new-arrival scheduling; use as processed association/probe data only.",
                "missing_source_frames": gaps, "gt_rows": len(gt_lines), "identity_map": id_map,
                "quarantined_source_frames": quarantined,
                "ground_truth": "gt/gt.txt", "images": "img1", "source_doi": "10.26180/32315067.v1",
                "oracle_detections": "oracle_detections.csv", "oracle_note": "Perfect per-available-frame GT boxes, score 1, no identity in detector input. Association diagnostic only; never an end-to-end detector result.",
                "image_content_sha256": image_digest.hexdigest(), "gt_sha256": sha256_file(destination / "gt" / "gt.txt"),
                "annotation_note": "Only supplied annotated images are scored. Local indices are contiguous; source timestamp gaps are preserved in frames.csv.",
            }
            (destination / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
            write_seqinfo(destination, sequence, len(frames), width, height, 25)
            sequences[sequence] = metadata
            print(f"Converted {sequence}: {len(frames)} available images, {len(id_map)} IDs, split={split}", flush=True)
    provenance = {
        "dataset": "Australian Native Bee Detection and Tracking Dataset", "source": SOURCE, "doi": "10.26180/32315067.v1",
        "download_url": ARCHIVE_URL, "archive": checksums, "license": "CC BY 4.0",
        "license_url": "https://creativecommons.org/licenses/by/4.0/",
        "attribution": "Asaduzzaman, Asad; Ratnayake, Malika Nisal; Dorin, Alan; Kellermann, Vanessa (2026), Monash University.",
        "source_author_code": "https://github.com/asadiceiu/native-bee-detection-tracking/tree/9589ef2c88d0b8e85112550519185a02a21df9c3",
        "available_sequences": len(sequences), "available_frames": sum(m["num_frames"] for m in sequences.values()),
        "excluded_sequences": excluded, "sequences": {name: {k: m[k] for k in ("split", "num_frames", "gt_rows", "gt_sha256", "image_content_sha256", "missing_source_frames")} for name, m in sequences.items()},
        "quarantined_frames": [item for meta in sequences.values() for item in meta["quarantined_source_frames"]],
        "label_member_hashes": source_member_hashes,
        "detector_training_provenance": "The repository's original best.onnx training data are unknown; no detector-generalisation claim is made by this conversion.",
        "processed_image_warning": "Visual audit: black masked backgrounds, insect ROI patches and embedded green counters. This is not raw full-scene footage; moving masks/counters confound scene-change/new-arrival claims.",
        "timing_warning": "Filename gaps are preserved using nominal provider 25 FPS. Embedded counters indicate additional upstream processing; no independently verified sensor timestamps are supplied.",
    }
    (output / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
    splits_path.parent.mkdir(parents=True, exist_ok=True)
    previous = json.loads(splits_path.read_text(encoding="utf-8")) if splits_path.exists() else {}
    previous["native_bee"] = {
        "policy": "Entire recording days are disjoint across splits; same camera and hive, so no cross-camera generalisation claim.",
        "date_groups": SPLIT_DATES,
        **{split: [f"data/native_bee/{name}" for name, metadata in sequences.items() if metadata["split"] == split] for split in SPLIT_DATES},
        "excluded": excluded,
    }
    splits_path.write_text(json.dumps(previous, indent=2) + "\n", encoding="utf-8")
    return provenance


def export_yolo(output: Path) -> Path:
    """Create explicit one-class detector splits from the identity-labelled videos."""
    lists = {split: [] for split in SPLIT_DATES}
    for sequence in sorted(output.iterdir()):
        metadata_path = sequence / "metadata.json"
        if not metadata_path.exists():
            continue
        meta = json.loads(metadata_path.read_text(encoding="utf-8"))
        labels_by_frame = {}
        for line in (sequence / "gt" / "gt.txt").read_text().splitlines():
            values = line.split(",")
            frame = int(values[0]); x, y, w, h = map(float, values[2:6])
            labels_by_frame.setdefault(frame, []).append(f"0 {(x+w/2)/meta['width']:.8f} {(y+h/2)/meta['height']:.8f} {w/meta['width']:.8f} {h/meta['height']:.8f}")
        # Ultralytics finds labels by replacing '/images/' with '/labels/'. Preserve
        # this conventional layout via hard links/copies; no source images change.
        images_dir = output.parent / "native_bee_detector" / "images" / sequence.name
        labels_dir = output.parent / "native_bee_detector" / "labels" / sequence.name
        images_dir.mkdir(parents=True, exist_ok=True); labels_dir.mkdir(parents=True, exist_ok=True)
        for image in sorted((sequence / "img1").glob("*.jpg")):
            target = images_dir / image.name
            if not target.exists():
                try:
                    target.hardlink_to(image)
                except OSError:
                    shutil.copyfile(image, target)
            (labels_dir / f"{image.stem}.txt").write_text("\n".join(labels_by_frame.get(int(image.stem), [])) + "\n", encoding="utf-8")
            lists[meta["split"]].append(target.resolve().as_posix())
    detector_root = output.parent / "native_bee_detector"
    for split, paths in lists.items():
        (detector_root / f"{split}.txt").write_text("\n".join(paths) + "\n", encoding="utf-8")
    config = detector_root / "dataset.yaml"
    config.write_text(f"path: {detector_root.resolve().as_posix()}\ntrain: train.txt\nval: validation.txt\ntest: test.txt\nnames:\n  0: insect\n", encoding="utf-8")
    return config


class SegmentedFile(io.RawIOBase):
    """Read an archive at original offsets from verified, disjoint local ranges."""
    def __init__(self, length: int, segments: list[tuple[int, bytes | Path]]):
        super().__init__()
        self.length = length
        self.position = 0
        self.segments = []
        for start, data in segments:
            if isinstance(data, Path):
                self.segments.append((start, data.stat().st_size, data.open("rb")))
            else:
                self.segments.append((start, len(data), io.BytesIO(data)))

    def seekable(self):
        return True

    def readable(self):
        return True

    def tell(self):
        return self.position

    def seek(self, offset, whence=io.SEEK_SET):
        target = offset if whence == io.SEEK_SET else self.position+offset if whence == io.SEEK_CUR else self.length+offset
        if target < 0 or target > self.length:
            raise ValueError("archive seek outside expected size")
        self.position = target
        return target

    def read(self, size=-1):
        remaining = self.length-self.position if size is None or size < 0 else min(size, self.length-self.position)
        result = []
        while remaining:
            segment = next((part for part in self.segments if part[0] <= self.position < part[0]+part[1]), None)
            if segment is None:
                raise ValueError(f"archive requested an undownloaded range at {self.position}")
            start, length, stream = segment
            stream.seek(self.position-start)
            chunk = stream.read(min(remaining, start+length-self.position))
            if not chunk:
                raise ValueError("truncated archive range")
            self.position += len(chunk); remaining -= len(chunk); result.append(chunk)
        return b"".join(result)

    def close(self):
        if not self.closed:
            for _, _, stream in self.segments:
                stream.close()
        super().close()


class GithubDataRedirects(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        parsed = urllib.parse.urlparse(newurl)
        if parsed.scheme != "https" or parsed.hostname not in {"github.com", "release-assets.githubusercontent.com", "objects.githubusercontent.com"}:
            raise ValueError("GitHub dataset download redirected outside expected official hosts")
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def prepare_gmot_downloads(downloads: Path, download: bool) -> dict:
    downloads.mkdir(parents=True, exist_ok=True)
    opener = urllib.request.build_opener(GithubDataRedirects())
    recorded = {}
    for name, (url, expected_size, expected_hash, byte_range) in GMOT_FILES.items():
        destination = downloads / name
        if not destination.exists():
            if not download:
                raise ValueError(f"Missing {destination}; pass --download for selective official GitHub data download")
            headers = {"User-Agent": USER_AGENT}
            if byte_range is not None:
                headers["Range"] = f"bytes={byte_range[0]}-{byte_range[1]}"
            request = urllib.request.Request(url, headers=headers)
            partial = destination.with_suffix(".part")
            with opener.open(request, timeout=90) as response, partial.open("wb") as output:
                if byte_range is not None:
                    content_range = f"bytes {byte_range[0]}-{byte_range[1]}/{GMOT_TOTAL_BYTES}"
                    if response.status != 206 or response.headers.get("Content-Range") != content_range:
                        raise ValueError("Server did not honor the exact byte range; refusing full image archive download")
                total = 0
                for chunk in iter(lambda: response.read(1024*1024), b""):
                    total += len(chunk)
                    if total > expected_size:
                        raise ValueError("GitHub data download exceeded pinned size")
                    output.write(chunk)
            if partial.stat().st_size != expected_size or sha256_file(partial) != expected_hash:
                raise ValueError(f"GitHub release checksum mismatch: {name}")
            partial.replace(destination)
        if destination.stat().st_size != expected_size or sha256_file(destination) != expected_hash:
            raise ValueError(f"Stored GitHub release checksum mismatch: {destination}")
        recorded[name] = {"url": url, "bytes": expected_size, "sha256": expected_hash, "range": byte_range}
    return recorded


def prepare_gmot(output: Path, downloads: Path, splits_path: Path, download=False, fps=25.0) -> dict:
    """Extract only four insect sequences from the author's 1.5 GB image archive."""
    from PIL import Image
    sources = prepare_gmot_downloads(downloads, download)
    # The pinned central directory ends with an ordinary non-ZIP64 EOCD. Only the
    # four contiguous insect ranges are needed to read their original ZIP entries.
    directory = (downloads / "gmot40_central.bin").read_bytes() + struct.pack(
        "<4s4H2LH", b"PK\x05\x06", 0, 0, 9723, 9723, GMOT_DIRECTORY_BYTES, GMOT_DIRECTORY_OFFSET, 0)
    output.mkdir(parents=True, exist_ok=True)
    split_by_sequence = {"insect-0": "train", "insect-1": "validation", "insect-2": "test", "insect-3": "test"}
    summaries = {}
    with SegmentedFile(GMOT_TOTAL_BYTES, [(GMOT_IMAGE_START, downloads / "gmot40_insect_images.range"), (GMOT_DIRECTORY_OFFSET, directory)]) as segmented, \
            zipfile.ZipFile(segmented) as images, zipfile.ZipFile(downloads / "gmot40_track_label.zip") as gt_archive, \
            zipfile.ZipFile(downloads / "gmot40_detection_label.zip") as det_archive:
        for sequence, split in split_by_sequence.items():
            prefix = f"GenericMOT_JPEG_Sequence/{sequence}/img1/"
            members = sorted((item for item in images.infolist() if item.filename.startswith(prefix) and item.filename.endswith(".jpg")), key=lambda item: item.filename)
            if not members:
                raise ValueError(f"No image entries for {sequence}")
            destination = safe_destination(output, sequence)
            (destination / "img1").mkdir(parents=True, exist_ok=True); (destination / "gt").mkdir(parents=True, exist_ok=True)
            source_to_local = {}; frame_rows = []; image_hash = hashlib.sha256()
            for index, item in enumerate(members, 1):
                safe_destination(output, item.filename)
                source_frame = int(Path(item.filename).stem)
                image_bytes = images.read(item)  # zipfile checks CRC after decompression.
                with Image.open(io.BytesIO(image_bytes)) as image:
                    if index == 1:
                        width, height = image.size
                    if image.size != (width, height):
                        raise ValueError("GMOT image dimensions changed within a sequence")
                    image.verify()
                (destination / "img1" / f"{index:06d}.jpg").write_bytes(image_bytes)
                image_hash.update(image_bytes); source_to_local[source_frame] = index
                frame_rows.append((index, f"{(index-1)/fps:.8f}", source_frame, item.filename))
            gt_rows, oracle_rows = [], []
            gt_text = gt_archive.read(f"track_label/{sequence}.txt").decode("utf-8-sig")
            identities = sorted({int(row[1]) for row in csv.reader(gt_text.splitlines()) if row})
            identity_map = {identity: index+1 for index, identity in enumerate(identities)}
            seen = set()
            for row in csv.reader(gt_text.splitlines()):
                if not row:
                    continue
                source_frame, identity = int(row[0]), int(row[1]); x, y, w, h = map(float, row[2:6])
                frame = source_to_local[source_frame]
                if (frame, identity) in seen or not all(math.isfinite(v) for v in (x, y, w, h)) or w <= 0 or h <= 0:
                    raise ValueError("Invalid/duplicate GMOT identity box")
                seen.add((frame, identity)); gt_rows.append((frame, identity_map[identity], x, y, w, h, 1, 1, 1))
                oracle_rows.append((frame, f"{(frame-1)/fps:.8f}", x, y, w, h, 1, 0))
            gt_rows.sort(key=lambda row: (row[0], row[1])); oracle_rows.sort(key=lambda row: row[0])
            detection_rows = []
            det_text = det_archive.read(f"detecion_label/{sequence}.txt").decode("utf-8-sig")
            for row in csv.reader(det_text.splitlines()):
                if not row:
                    continue
                source_frame = int(row[0]); x, y, w, h, score = map(float, row[2:7])
                if source_frame not in source_to_local or not all(math.isfinite(v) for v in (x, y, w, h, score)) or w <= 0 or h <= 0 or not 0 <= score <= 1:
                    raise ValueError("Invalid GMOT author detection")
                frame = source_to_local[source_frame]
                detection_rows.append((frame, f"{(frame-1)/fps:.8f}", x, y, w, h, score, 0))
            detection_rows.sort(key=lambda row: row[0])
            for name, header, rows in (("gt/gt.txt", None, gt_rows),
                                       ("frames.csv", ["frame", "timestamp", "source_frame", "source_member"], frame_rows),
                                       ("detections.csv", ["frame", "timestamp", "x", "y", "w", "h", "score", "class_id"], detection_rows),
                                       ("oracle_detections.csv", ["frame", "timestamp", "x", "y", "w", "h", "score", "class_id"], oracle_rows)):
                with (destination / name).open("w", newline="", encoding="utf-8") as stream:
                    writer = csv.writer(stream)
                    if header is not None:
                        writer.writerow(header)
                    writer.writerows(rows)
            metadata = {
                "name": sequence, "dataset": "gmot40", "synthetic": False, "split": split, "num_frames": len(members),
                "width": width, "height": height, "fps": fps, "fps_verified": False,
                "timestamp_file": "frames.csv", "timestamp_basis": "Observation indices at an assumed nominal 25 FPS; capture timestamps/FPS are not supplied. Report discovery delays in frames.",
                "images": "img1", "ground_truth": "gt/gt.txt", "detections": "detections.csv", "oracle_detections": "oracle_detections.csv",
                "detection_source": "Author-released GMOT40 baseline one-shot detections, not the project's best.onnx. Test-sequence first-frame target exemplar belongs to the author's one-shot protocol.",
                "oracle_note": "Perfect per-frame GT boxes without identity, association diagnostic only.",
                "gt_rows": len(gt_rows), "detections_rows": len(detection_rows), "identity_map": identity_map,
                "image_content_sha256": image_hash.hexdigest(), "gt_sha256": sha256_file(destination / "gt" / "gt.txt"),
                "detection_sha256": sha256_file(destination / "detections.csv"),
                "evaluation_scope": "Insect-only subset with a custom sequence split, not the complete official GMOT40 one-shot benchmark or a new-detector end-to-end result.",
            }
            (destination / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
            write_seqinfo(destination, sequence, len(members), width, height, fps)
            summaries[sequence] = metadata
            print(f"Converted GMOT40 {sequence}: {len(members)} images, {len(identities)} IDs, {len(detection_rows)} author detections, split={split}", flush=True)
    provenance = {
        "dataset": "GMOT40 insect subset", "source": "https://spritea.github.io/GMOT40/download.html",
        "author_repository": "https://github.com/Spritea/GMOT40", "release": "v0.1", "download_files": sources,
        "license": "CC BY-NC-SA 4.0", "license_url": "https://creativecommons.org/licenses/by-nc-sa/4.0/",
        "attribution": "Hexin Bai, Wensheng Cheng, Peng Chu, Kai Zhang and Haibin Ling. GMOT-40: A Benchmark for Generic Multiple Object Tracking, CVPR 2021.",
        "full_image_archive_downloaded": False, "selective_image_range_bytes": GMOT_IMAGE_END-GMOT_IMAGE_START+1,
        "sequences": {name: {key: data[key] for key in ("split", "num_frames", "gt_rows", "detections_rows", "image_content_sha256", "gt_sha256", "detection_sha256")} for name, data in summaries.items()},
        "timing_warning": "Capture timing is not provided; timestamps are nominal observation indices. Frame-based algorithm comparisons do not establish physical real-time motion accuracy.",
        "protocol_warning": "Own insect-only split and cached author detector outputs; not an official benchmark submission or an end-to-end deployment speed measurement.",
    }
    (output / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
    previous = json.loads(splits_path.read_text(encoding="utf-8")) if splits_path.exists() else {}
    previous["gmot40"] = {
        "policy": "Whole insect sequences: insect-0 train, insect-1 validation, insect-2/3 held-out test. Custom subset protocol; source recording session independence beyond sequence IDs is not verified.",
        **{split: [f"data/gmot40/{sequence}" for sequence, role in split_by_sequence.items() if role == split] for split in ("train", "validation", "test")}}
    splits_path.parent.mkdir(parents=True, exist_ok=True)
    splits_path.write_text(json.dumps(previous, indent=2) + "\n", encoding="utf-8")
    return provenance


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--archive", type=Path, default=Path("data/_downloads") / ARCHIVE_NAME)
    parser.add_argument("--download", action="store_true", help="Download the pinned 107 MB tracking archive from Figshare/Monash")
    parser.add_argument("--splits", type=Path, default=Path("configs/splits.json"))
    parser.add_argument("--export-yolo", action="store_true", help="Also prepare day-disjoint one-class detector training lists")
    parser.add_argument("--gmot", action="store_true", help="Prepare the four GMOT40 insect clips through selective verified GitHub ZIP ranges")
    args = parser.parse_args()
    if args.gmot:
        result = prepare_gmot(args.output or Path("data/gmot40"), args.archive.parent, args.splits, args.download)
        print(json.dumps({"gmot40_sequences": len(result["sequences"]), "images": sum(x["num_frames"] for x in result["sequences"].values())}, indent=2))
        return
    args.output = args.output or Path("data/native_bee")
    if args.download:
        download_archive(args.archive)
    if not args.archive.is_file():
        parser.error("archive is missing; pass --download or --archive PATH")
    provenance = convert_archive(args.archive, args.output, args.splits)
    if args.export_yolo:
        print(f"Detector dataset: {export_yolo(args.output)}")
    print(json.dumps({"sequences": provenance["available_sequences"], "frames": provenance["available_frames"], "archive_sha256": provenance["archive"]["sha256"]}, indent=2))


if __name__ == "__main__":
    main()
