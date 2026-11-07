"""Input-safety, split-isolation and synthetic-data semantic checks."""
from __future__ import annotations

import csv
import json
from pathlib import Path
import tempfile
import unittest
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from prepare_data import SPLIT_DATES, SegmentedFile, date_split, parse_yolo_identity, safe_destination
from generate_synthetic import generate_sequence, make_paths


def read_csv(path: Path, dictionary=False):
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream) if dictionary else csv.reader(stream))


class DataTests(unittest.TestCase):
    def test_archive_paths_cannot_escape(self):
        root = ROOT / "data" / "_test_safety"
        for member in ("../outside.jpg", "/outside.jpg", "C:/outside.jpg", "C:\\outside.jpg", "x/../../bad.jpg", "\\\\server\\share\\bad.jpg"):
            with self.subTest(member=member), self.assertRaises(ValueError):
                safe_destination(root, member)
        self.assertEqual(safe_destination(root, "seq/img1/000001.jpg"), root.resolve() / "seq" / "img1" / "000001.jpg")

    def test_identity_labels_preserve_geometry(self):
        identity, x, y, w, h = parse_yolo_identity("bee_17 0.5 0.25 0.1 0.2", 1280, 720)
        self.assertEqual(identity, "bee_17")
        self.assertAlmostEqual(x, 576)
        self.assertAlmostEqual(y, 108)
        self.assertAlmostEqual(w, 128)
        self.assertAlmostEqual(h, 144)

    def test_invalid_annotations_rejected(self):
        for line in ("person 0.5 0.5 .1 .1", "bee_1 nan .5 .1 .1", "bee_1 .5 .5 -1 .1", "bee_1 .5 .5 .1", "bee_1 1.1 .5 .1 .1"):
            with self.subTest(line=line), self.assertRaises(ValueError):
                parse_yolo_identity(line, 640, 360)

    def test_recording_days_never_cross_splits(self):
        all_days = [day for days in SPLIT_DATES.values() for day in days]
        self.assertEqual(len(all_days), len(set(all_days)))
        self.assertEqual(date_split("Camera_43_20240505_130943"), "test")
        self.assertEqual(date_split("Camera_43_20240531_143838"), "validation")

    def test_selective_archive_reader_refuses_missing_bytes(self):
        with SegmentedFile(12, [(0, b"abcd"), (8, b"ijkl")]) as stream:
            self.assertEqual(stream.read(4), b"abcd")
            stream.seek(-4, 2)
            self.assertEqual(stream.read(), b"ijkl")
            stream.seek(4)
            with self.assertRaises(ValueError):
                stream.read(1)

    def test_short_tracks_and_occlusion_exist(self):
        paths = make_paths("births", 71, 180, 640, 360)
        short_paths = [p for p in paths if p.born > 1]
        self.assertEqual(len(short_paths), 10)
        self.assertTrue(all(6 <= p.died-p.born+1 <= 13 for p in short_paths))

    def test_synthetic_is_deterministic_and_self_described(self):
        scratch = ROOT / "data" / "_test_runs"
        scratch.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="data-check-", dir=scratch) as temp:
            temp_path = Path(temp).resolve()
            self.assertTrue(temp_path.is_relative_to(scratch.resolve()))
            first = generate_sequence(temp_path / "a", "births", 7, 24, 160, 96, 25)
            second = generate_sequence(temp_path / "b", "births", 7, 24, 160, 96, 25)
            third = generate_sequence(temp_path / "c", "births", 9, 24, 160, 96, 25)
            for key in ("image_content_sha256", "gt_sha256", "detection_sha256"):
                self.assertEqual(first[key], second[key])
            self.assertNotEqual(first["image_content_sha256"], third["image_content_sha256"])
            self.assertTrue(first["synthetic"])
            rows = read_csv(temp_path / "a" / "frames.csv", dictionary=True)
            self.assertEqual(len(rows), 24)
            self.assertEqual(float(rows[0]["timestamp"]), 0)
            self.assertAlmostEqual(float(rows[-1]["timestamp"]), 23/25)
            gt = read_csv(temp_path / "a" / "gt" / "gt.txt")
            self.assertTrue(all(len(row) == 9 and float(row[4]) > 0 and float(row[5]) > 0 for row in gt))

    def test_prepared_native_bee_integrity(self):
        manifest = ROOT / "data" / "native_bee" / "provenance.json"
        if not manifest.exists():
            self.skipTest("real dataset not downloaded")
        data = json.loads(manifest.read_text())
        self.assertEqual(data["available_frames"], 3510)
        self.assertEqual(data["available_sequences"], 13)
        self.assertEqual(len(data["quarantined_frames"]), 1)
        for sequence in data["sequences"]:
            directory = manifest.parent / sequence
            meta = json.loads((directory / "metadata.json").read_text())
            frame_rows = read_csv(directory / "frames.csv", dictionary=True)
            self.assertEqual(len(frame_rows), meta["num_frames"])
            self.assertEqual(len(list((directory / "img1").glob("*.jpg"))), meta["num_frames"])
            self.assertTrue(all(float(b["timestamp"]) > float(a["timestamp"]) for a, b in zip(frame_rows, frame_rows[1:])))
            seen = set()
            for row in read_csv(directory / "gt" / "gt.txt"):
                pair = (int(row[0]), int(row[1]))
                self.assertNotIn(pair, seen)
                seen.add(pair)

    def test_prepared_gmot_subset_integrity(self):
        manifest = ROOT / "data" / "gmot40" / "provenance.json"
        if not manifest.exists():
            self.skipTest("GMOT40 insect subset not downloaded")
        provenance = json.loads(manifest.read_text())
        self.assertFalse(provenance["full_image_archive_downloaded"])
        self.assertEqual(sum(sequence["num_frames"] for sequence in provenance["sequences"].values()), 659)
        self.assertEqual(sum(sequence["num_frames"] for sequence in provenance["sequences"].values() if sequence["split"] == "test"), 348)
        for sequence, entry in provenance["sequences"].items():
            directory = manifest.parent / sequence
            times = read_csv(directory / "frames.csv", dictionary=True)
            self.assertEqual(len(times), entry["num_frames"])
            self.assertEqual(int(times[0]["source_frame"]), 0)
            self.assertEqual(len(list((directory / "img1").glob("*.jpg"))), entry["num_frames"])
            seen = set()
            for row in read_csv(directory / "gt" / "gt.txt"):
                key = (int(row[0]), int(row[1]))
                self.assertNotIn(key, seen)
                seen.add(key)
                self.assertGreater(float(row[4]), 0)
                self.assertGreater(float(row[5]), 0)
            for row in read_csv(directory / "detections.csv", dictionary=True):
                self.assertTrue(1 <= int(row["frame"]) <= entry["num_frames"])
                self.assertTrue(0 <= float(row["score"]) <= 1)


if __name__ == "__main__":
    unittest.main()
