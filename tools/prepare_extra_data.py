#!/usr/bin/env python3
"""Prepare nine predeclared, previously unused GMOT-40 confirmation clips.

Downloads data only from the author's GitHub release. Never scores the clips,
loads model weights or executes downloaded code. JPEG bytes are preserved.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import io
import json
import math
from pathlib import Path
import stat
import struct
import time
import urllib.parse
import urllib.request
import zipfile

from prepare_data import (GMOT_FILES, GMOT_IMAGE_URL, GMOT_TOTAL_BYTES,
                          GMOT_DIRECTORY_OFFSET, GMOT_DIRECTORY_BYTES,
                          GithubDataRedirects, SegmentedFile, safe_destination,
                          sha256_file, write_seqinfo, USER_AGENT)

ROOT = Path(__file__).resolve().parents[1]
SELECTION = {
    "airplane-2": {"start": 31066754, "end": 44322749, "frames": 251, "sha256": "96527741825ccb99331c30fab93d4590922c2655c3735e52bdb10e4453eec27a"},
    "ball-3": {"start": 203726785, "end": 227781569, "frames": 150, "sha256": "2445aea394246e6146aab00529e4e2a5aae1777796e72d152e0660576a52407c"},
    "balloon-2": {"start": 262549896, "end": 266412072, "frames": 151, "sha256": "a225404a644d32f638a6d3e778f28848d74dad1b794c66982dd20a903cbf3ec3"},
    "bird-2": {"start": 339746043, "end": 398549114, "frames": 357, "sha256": "4b49ed3f16a9dfe476f0434af9969a205229010b4aac69cd56f2f5bf40ddd53a"},
    "boat-2": {"start": 627357528, "end": 631566909, "frames": 198, "sha256": "456fb82a08ad238382e71625b4b2883f15170fa9f26e74210b0ae5a1966abcdd"},
    "car-2": {"start": 750326056, "end": 786926545, "frames": 199, "sha256": "9733143c7b802125901246d98d5cc9a73eacb7d55a771a961c708edae229c1a6"},
    "fish-2": {"start": 852148670, "end": 866920048, "frames": 151, "sha256": "f698370beaee3707f65dd0a7cfccf93aa4d19719f115856697154b1c5c7c699c"},
    "person-0": {"start": 990099419, "end": 1008342323, "frames": 184, "sha256": "3dd344910167ababb407561520cf2e0de4693c527caefc43c83b54d5d82066b0"},
    "stock-1": {"start": 1274143145, "end": 1305239699, "frames": 140, "sha256": "bcf82eba0223a8fe91d3f5e838a43ffe2472ce9fd37fd7ed26f7e36a2c1d6e76"},
}
MAX_DOWNLOAD_BYTES = 300_000_000
# Official release asset API avoids intermittent 504s at the browser download URL.
GMOT_ASSET_API = "https://api.github.com/repos/Spritea/GMOT40/releases/assets/242989653"


def validate_response(status, headers, start, end):
    expected = f"bytes {start}-{end}/{GMOT_TOTAL_BYTES}"
    if status != 206 or headers.get("Content-Range") != expected:
        raise ValueError("Server must honor the exact HTTP 206 byte range; refusing full archive download")
    length = headers.get("Content-Length")
    if length is not None and int(length) != end - start + 1:
        raise ValueError("Range Content-Length differs from the selected bytes")


def verify_source_archives(downloads):
    sources = {}
    for name in ("gmot40_central.bin", "gmot40_track_label.zip", "gmot40_detection_label.zip"):
        url, size, digest, region = GMOT_FILES[name]
        path = downloads / name
        if not path.is_file() or path.stat().st_size != size or sha256_file(path) != digest:
            raise ValueError(f"Missing or altered pinned source {path}; prepare the original GMOT metadata/label archives first")
        sources[name] = {"url": url, "bytes": size, "sha256": digest, "range": region}
    return sources


def download_ranges(downloads, download):
    total = sum(item["end"] - item["start"] + 1 for item in SELECTION.values())
    if total > MAX_DOWNLOAD_BYTES:
        raise ValueError("Selected raw clip download exceeds the declared 300 MB ceiling")
    opener = urllib.request.build_opener(GithubDataRedirects())
    records = {}
    for sequence, item in SELECTION.items():
        path = downloads / f"gmot40_{sequence}_images.range"
        size = item["end"] - item["start"] + 1
        if not path.exists():
            if not download:
                raise ValueError(f"Missing {path}; pass --download for the declared author-release ranges")
            partial = path.with_suffix(".part")
            with partial.open("wb") as output:
                # Small exact ranges avoid a proxy buffering the complete 59 MB
                # clip. Retries are bounded; three complete transfers are <300 MB.
                for first in range(item["start"], item["end"] + 1, 8 * 1024 * 1024):
                    last = min(first + 8 * 1024 * 1024 - 1, item["end"])
                    for attempt in range(3):
                        try:
                            url = GMOT_ASSET_API + f"?kinetic_range={first}-{last}"
                            request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT, "Accept": "application/octet-stream",
                                                           "Range": f"bytes={first}-{last}"})
                            with opener.open(request, timeout=60) as response:
                                validate_response(response.status, response.headers, first, last)
                                host = urllib.parse.urlparse(response.url).hostname
                                if host not in {"github.com", "release-assets.githubusercontent.com", "objects.githubusercontent.com"}:
                                    raise ValueError("Untrusted final data download host")
                                block = response.read(last - first + 2)
                                if len(block) != last - first + 1:
                                    raise ValueError("Downloaded range chunk is truncated or oversized")
                            output.write(block)
                            print(f"Downloaded {sequence} range {first}-{last}", flush=True)
                            break
                        except OSError:
                            if attempt == 2:
                                raise
                            time.sleep(attempt + 1)
            if partial.stat().st_size != size:
                raise ValueError("Downloaded range is truncated")
            if item["sha256"] and sha256_file(partial) != item["sha256"]:
                raise ValueError("Downloaded range checksum differs from its pinned release bytes")
            partial.replace(path)
        if path.stat().st_size != size or (item["sha256"] and sha256_file(path) != item["sha256"]):
            raise ValueError(f"Stored range failed checksum/size: {path}")
        records[sequence] = {"path": str(path), "url": GMOT_IMAGE_URL, "download_api": GMOT_ASSET_API,
                             "range": [item["start"], item["end"]], "bytes": size,
                             "sha256": sha256_file(path), "expected_frames": item["frames"]}
    return records


def parse_labels(text, source_to_local, fps, ground_truth):
    rows = []
    seen = set()
    identities = sorted({int(row[1]) for row in csv.reader(text.splitlines()) if row}) if ground_truth else []
    identity_map = {identity: index + 1 for index, identity in enumerate(identities)}
    for row in csv.reader(text.splitlines()):
        if not row:
            continue
        if len(row) < (6 if ground_truth else 7):
            raise ValueError("Malformed GMOT annotation row")
        source_frame = int(row[0])
        if source_frame not in source_to_local:
            raise ValueError("Annotation references an unavailable source frame")
        x, y, width, height = map(float, row[2:6])
        if not all(math.isfinite(value) for value in (x, y, width, height)) or width <= 0 or height <= 0:
            raise ValueError("Invalid annotation geometry")
        frame = source_to_local[source_frame]
        if ground_truth:
            identity = int(row[1])
            if (frame, identity) in seen:
                raise ValueError("Duplicate ground-truth frame/identity")
            seen.add((frame, identity))
            rows.append((frame, identity_map[identity], x, y, width, height, 1, 1, 1))
        else:
            score = float(row[6])
            if not math.isfinite(score) or not 0 <= score <= 1:
                raise ValueError("Invalid author detector confidence")
            rows.append((frame, f"{(frame-1)/fps:.8f}", x, y, width, height, score, 0))
    rows.sort(key=lambda row: (row[0], row[1]) if ground_truth else (row[0],))
    return rows, identity_map


def prepare(output: Path, downloads: Path, download=False, fps=25.0):
    from PIL import Image
    if not math.isfinite(fps) or fps <= 0:
        raise ValueError("Nominal fps must be finite and positive")
    sources = verify_source_archives(downloads)
    regions = download_ranges(downloads, download)
    central = (downloads / "gmot40_central.bin").read_bytes() + struct.pack(
        "<4s4H2LH", b"PK\x05\x06", 0, 0, 9723, 9723, GMOT_DIRECTORY_BYTES, GMOT_DIRECTORY_OFFSET, 0)
    segments = [(item["range"][0], Path(item["path"])) for item in regions.values()]
    segments.append((GMOT_DIRECTORY_OFFSET, central))
    sequences = {}
    output.mkdir(parents=True, exist_ok=True)
    with SegmentedFile(GMOT_TOTAL_BYTES, segments) as stream, zipfile.ZipFile(stream) as images, \
            zipfile.ZipFile(downloads / "gmot40_track_label.zip") as gt_zip, \
            zipfile.ZipFile(downloads / "gmot40_detection_label.zip") as det_zip:
        for sequence, item in SELECTION.items():
            prefix = f"GenericMOT_JPEG_Sequence/{sequence}/img1/"
            members = sorted((entry for entry in images.infolist() if entry.filename.startswith(prefix)
                              and entry.filename.endswith(".jpg")), key=lambda entry: entry.filename)
            if len(members) != item["frames"] or sum(entry.file_size for entry in members) > 200_000_000:
                raise ValueError("Selected central-directory membership/size differs from declaration")
            destination = safe_destination(output, sequence)
            for subdirectory in ("img1", "gt", "source_annotations"):
                (destination / subdirectory).mkdir(parents=True, exist_ok=True)
            mapping, frame_rows, per_image = {}, [], {}
            image_hash = hashlib.sha256()
            for index, entry in enumerate(members, 1):
                safe_destination(output, entry.filename)
                if stat.S_ISLNK(entry.external_attr >> 16) or entry.file_size > 10_000_000:
                    raise ValueError("Invalid image ZIP member")
                if entry.header_offset < item["start"] or entry.header_offset >= item["end"]:
                    raise ValueError("Selected entry header lies outside its downloaded range")
                source_frame = int(Path(entry.filename).stem)
                if source_frame in mapping:
                    raise ValueError("Duplicate source image index")
                raw = images.read(entry)  # Checks each original JPEG's ZIP CRC.
                with Image.open(io.BytesIO(raw)) as image:
                    if index == 1:
                        width, height = image.size
                    if image.size != (width, height) or image.format != "JPEG":
                        raise ValueError("Unexpected image format/dimension change")
                    image.verify()
                (destination / "img1" / f"{index:06d}.jpg").write_bytes(raw)
                mapping[source_frame] = index
                image_hash.update(raw)
                per_image[entry.filename] = {"sha256": hashlib.sha256(raw).hexdigest(), "crc32": f"{entry.CRC:08x}"}
                frame_rows.append((index, f"{(index-1)/fps:.8f}", source_frame, entry.filename))
            gt_raw = gt_zip.read(f"track_label/{sequence}.txt")
            det_raw = det_zip.read(f"detecion_label/{sequence}.txt")
            (destination / "source_annotations/track_label.txt").write_bytes(gt_raw)
            (destination / "source_annotations/detecion_label.txt").write_bytes(det_raw)
            ground, identities = parse_labels(gt_raw.decode("utf-8-sig"), mapping, fps, True)
            detections, _ = parse_labels(det_raw.decode("utf-8-sig"), mapping, fps, False)
            for name, header, rows in (("gt/gt.txt", None, ground),
                                      ("frames.csv", ["frame", "timestamp", "source_frame", "source_member"], frame_rows),
                                      ("detections.csv", ["frame", "timestamp", "x", "y", "w", "h", "score", "class_id"], detections)):
                with (destination / name).open("w", newline="", encoding="utf-8") as output_stream:
                    writer = csv.writer(output_stream)
                    if header:
                        writer.writerow(header)
                    writer.writerows(rows)
            metadata = {"name": sequence, "dataset": "gmot40_confirmation", "split": "confirmation",
                        "synthetic": False, "num_frames": len(members), "width": width, "height": height,
                        "fps": fps, "fps_verified": False, "timestamp_file": "frames.csv", "images": "img1",
                        "ground_truth": "gt/gt.txt", "detections": "detections.csv", "identity_map": identities,
                        "gt_rows": len(ground), "detections_rows": len(detections),
                        "timestamp_basis": "Nominal assumed FPS; physical capture timing not supplied. Use frame delays.",
                        "detection_source": "Unmodified author-released one-shot detector labels; not this project's ONNX or retrained detector.",
                        "image_content_sha256": image_hash.hexdigest(), "gt_sha256": sha256_file(destination / "gt/gt.txt"),
                        "detection_sha256": sha256_file(destination / "detections.csv"),
                        "source_gt_sha256": hashlib.sha256(gt_raw).hexdigest(), "source_detections_sha256": hashlib.sha256(det_raw).hexdigest(),
                        "confirmation_scope": "Nine predeclared previously unused non-insect category clips. No scoring before refinement freeze. Not full GMOT40 or verified recording-independent generalization."}
            (destination / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
            (destination / "source_images.json").write_text(json.dumps(per_image, indent=2) + "\n", encoding="utf-8")
            write_seqinfo(destination, sequence, len(members), width, height, fps)
            sequences[sequence] = metadata
            print(f"Prepared (NOT SCORED) {sequence}: {len(members)} original JPEGs, {len(identities)} identities", flush=True)
    provenance = {"dataset": "GMOT40 additional confirmation clips", "author_repository": "https://github.com/Spritea/GMOT40",
                  "source": "https://spritea.github.io/GMOT40/download.html", "release": "v0.1",
                  "license": "CC BY-NC-SA 4.0", "license_url": "https://creativecommons.org/licenses/by-nc-sa/4.0/",
                  "attribution": "Hexin Bai, Wensheng Cheng, Peng Chu, Kai Zhang and Haibin Ling. GMOT-40, CVPR 2021.",
                  "selection": "Nine non-insect categories declared before scoring, expanded at the user's request. Index2 except ball3/person0/stock1 selected to keep total download below300MB; not selected by performance.",
                  "sources": sources, "image_ranges": regions, "selective_download_bytes": sum(record["bytes"] for record in regions.values()),
                  "integrity": "Pinned central/label archive SHA-256, exact HTTP206 ranges, original member CRC32 and recorded JPEG/range SHA-256; no image re-encoding.",
                  "preparation_performs_scoring": False, "sequences": sequences,
                  "limitations": ["One additional clip per non-insect category; no full benchmark or insect-specific improvement claim.",
                                  "Recording/session independence beyond sequence labels is unverified.",
                                  "Author one-shot detections use their declared first-frame exemplar protocol.",
                                  "Capture timing unknown; nominal FPS cannot verify physical real-time accuracy."]}
    (output / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
    return provenance


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--downloads", type=Path, default=ROOT / "data/_downloads")
    parser.add_argument("--output", type=Path, default=ROOT / "data/gmot40_confirmation")
    parser.add_argument("--download", action="store_true")
    args = parser.parse_args(argv)
    try:
        prepare(args.output, args.downloads, args.download)
    except (OSError, ValueError, zipfile.BadZipFile) as error:
        parser.exit(2, f"error: {error}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
