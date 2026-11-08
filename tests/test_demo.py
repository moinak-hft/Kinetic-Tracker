"""Comparison rendering must preserve every frame and label predictions honestly."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import sys
import shutil
import tempfile
import unittest

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from make_demo import Run, Track, dataset_notice, id_colour, load_run, make_demo, render_panel


class ConfirmationAttributionTests(unittest.TestCase):
    def test_confirmation_subset_keeps_original_gmot_licence(self):
        notice, credit, attribution = dataset_notice({'dataset': 'gmot40_confirmation'})
        self.assertIn('GMOT-40', credit)
        self.assertEqual(attribution['derivative_video_license'], 'CC BY-NC-SA 4.0')
        self.assertIn('FIXED DETECTION REPLAY', notice)


def write_csv(path, header, rows):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream); writer.writerow(header); writer.writerows(rows)


class DemoTests(unittest.TestCase):
    def setUp(self):
        scratch = ROOT / "data" / "_test_runs"
        scratch.mkdir(parents=True, exist_ok=True)
        self.temporary = tempfile.TemporaryDirectory(prefix="demo-test-", dir=scratch)
        self.root = Path(self.temporary.name).resolve()
        self.assertTrue(self.root.is_relative_to(scratch.resolve()))
        self.sequence = self.root / "sequence"; (self.sequence / "img1").mkdir(parents=True)
        image_hash = hashlib.sha256()
        for frame in range(1, 4):
            image = np.full((40, 80, 3), (30+frame*20, 80, 110), dtype=np.uint8)
            cv2.circle(image, (20+frame*3, 20), 4, (220, 220, 220), -1)
            path = self.sequence / "img1" / f"{frame:06d}.jpg"
            cv2.imwrite(str(path), image)
            image_hash.update(path.read_bytes())
        self.metadata = {"name": "test-clip", "dataset": "synthetic", "synthetic": True, "num_frames": 3,
                         "width": 80, "height": 40, "fps": 25, "images": "img1", "timestamp_file": "frames.csv",
                         "image_content_sha256": image_hash.hexdigest()}
        (self.sequence / "metadata.json").write_text(json.dumps(self.metadata), encoding="utf-8")
        write_csv(self.sequence / "frames.csv", ["frame", "timestamp"], [(1, 0), (2, .04), (3, .08)])
        self.left, self.right = self.root / "left", self.root / "right"
        for run in (self.left, self.right):
            run.mkdir()
            (run / "summary.json").write_text(json.dumps({"frames": 3, "detector_calls": 2, "mode": "cached_detection_replay", "schedule": "test", "association": "sort"}), encoding="utf-8")
            write_csv(run / "frames.csv", ["frame", "timestamp", "detector_ran"], [(1, 0, 1), (2, .04, 0), (3, .08, 1)])
            write_csv(run / "telemetry.csv", ["frame", "id", "x", "y", "w", "h", "score", "state", "observed"],
                      [(1, 1, 20, 15, 12, 9, .9, "confirmed", 1),
                       (2, 1, 23, 15, 12, 9, .9, "confirmed", 0),
                       (3, 1, 26, 15, 12, 9, .9, "confirmed", 1)])

    def tearDown(self):
        self.temporary.cleanup()

    def test_video_retains_frames_and_writes_provenance(self):
        output = self.root / "presentation" / "comparison.mp4"
        result = make_demo(self.sequence, self.left, self.right, output, "Fixed", "Adaptive", 320, encoder="opencv")
        self.assertEqual(result["frames"], 3)
        self.assertEqual(result["decoded_output_frames_verified"], 3)
        self.assertTrue(result["all_source_frames_retained"])
        self.assertEqual(result["detector_execution"], "none; this tool only renders saved tracking outputs")
        self.assertEqual(result["output_sha256"], hashlib.sha256(output.read_bytes()).hexdigest())
        self.assertEqual([run["detector_calls"] for run in result["methods"]], [2, 2])
        self.assertTrue(output.with_suffix(".attribution.json").exists())
        self.assertEqual(result["attribution"]["kind"], "synthetic")

    def test_existing_output_is_never_overwritten(self):
        output = self.root / "existing.mp4"; output.write_bytes(b"keep-original")
        with self.assertRaisesRegex(ValueError, "overwrite"):
            make_demo(self.sequence, self.left, self.right, output, panel_width=320)
        self.assertEqual(output.read_bytes(), b"keep-original")

    @unittest.skipUnless(shutil.which("ffmpeg"), "FFmpeg not on PATH; OpenCV writer is tested separately")
    def test_ffmpeg_encoder_preserves_all_frames(self):
        output = self.root / "presentation" / "h264.mp4"
        result = make_demo(self.sequence, self.left, self.right, output, panel_width=320)
        self.assertEqual(result["decoded_output_frames_verified"], 3)
        self.assertTrue(result["encoder"].startswith("FFmpeg libx264"))

    def test_input_directory_is_protected(self):
        with self.assertRaisesRegex(ValueError, "outside source"):
            make_demo(self.sequence, self.left, self.right, self.sequence / "bad.mp4", panel_width=320)
        self.assertFalse((self.sequence / "bad.mp4").exists())

    def test_dimension_mismatch_leaves_no_output(self):
        self.metadata["width"] = 82
        (self.sequence / "metadata.json").write_text(json.dumps(self.metadata), encoding="utf-8")
        output = self.root / "presentation" / "bad-shape.mp4"
        with self.assertRaisesRegex(ValueError, "dimensions"):
            make_demo(self.sequence, self.left, self.right, output, panel_width=320, encoder="opencv")
        self.assertFalse(output.exists())
        self.assertFalse(output.with_name("bad-shape.partial.mp4").exists())

    def test_incomplete_runs_are_rejected(self):
        write_csv(self.left / "frames.csv", ["frame", "timestamp", "detector_ran"], [(1, 0, 1), (3, .08, 1)])
        with self.assertRaisesRegex(ValueError, "every source"):
            load_run(self.left, "Incomplete", 3)

    def test_fake_observation_on_skipped_frame_is_rejected(self):
        write_csv(self.left / "telemetry.csv", ["frame", "id", "x", "y", "w", "h", "score", "state", "observed"],
                  [(2, 1, 23, 15, 12, 9, .9, "confirmed", 1)])
        with self.assertRaisesRegex(ValueError, "skipped detector"):
            load_run(self.left, "Invalid", 3)

    def test_observed_predicted_and_tentative_styles_differ(self):
        source = np.zeros((90, 160, 3), dtype=np.uint8)
        colours = []
        for state, observed in (("confirmed", True), ("confirmed", False), ("tentative", True)):
            run = Run(Path("."), "Method", {1: {"detector_ran": observed}},
                      {1: [Track(7, 20, 25, 40, 30, .8, state, observed)]}, {}, {})
            panel = render_panel(source, run, 1, int(observed), 1, 320, 180)
            colours.append(tuple(int(value) for value in panel[160, 43]))
        self.assertEqual(colours, [(70, 220, 100), (0, 190, 255), (155, 158, 164)])
        self.assertEqual(id_colour(7), id_colour(7))
        self.assertNotEqual(id_colour(7), id_colour(8))

    def test_dataset_licences_and_processed_flag_are_explicit(self):
        notice, credit, attribution = dataset_notice({"dataset": "gmot40", "synthetic": False})
        self.assertIn("CC BY-NC-SA", credit)
        self.assertEqual(attribution["derivative_video_license"], "CC BY-NC-SA 4.0")
        notice, _, attribution = dataset_notice({"dataset": "native_bee", "synthetic": False})
        self.assertIn("PROCESSED", notice)
        self.assertIn("ROI", notice)
        self.assertIn("confound", attribution["limitation"])


if __name__ == "__main__":
    unittest.main()
