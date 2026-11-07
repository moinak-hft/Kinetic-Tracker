"""Black-box regressions for the built native executable; no model download needed."""
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
HEADER = ["frame", "timestamp", "x", "y", "w", "h", "score", "class_id"]


class NativeCLITests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        override = os.environ.get("KINETIC_TRACKER_EXE")
        candidates = ([Path(override)] if override else []) + [
            ROOT / "build" / "kinetic_tracker.exe",
            ROOT / "build" / "kinetic_tracker",
            ROOT / "build" / "Release" / "kinetic_tracker.exe",
        ]
        cls.executable = next((path for path in candidates if path.is_file()), None)
        if cls.executable is None:
            raise unittest.SkipTest("Build kinetic_tracker first, or set KINETIC_TRACKER_EXE")
        cls.environment = os.environ.copy()
        msys = Path("D:/Tools/msys64/ucrt64/bin")
        if msys.is_dir():
            cls.environment["PATH"] = str(msys) + os.pathsep + cls.environment.get("PATH", "")

    def setUp(self):
        (ROOT / "work").mkdir(exist_ok=True)
        self.scratch = tempfile.TemporaryDirectory(prefix="cli-test-", dir=ROOT / "work")
        self.addCleanup(self.scratch.cleanup)
        self.directory = Path(self.scratch.name)
        self.run_number = 0

    def cache(self, rows=(), name="input.csv"):
        path = self.directory / name
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream, lineterminator="\n")
            writer.writerow(HEADER)
            writer.writerows(rows)
        return path

    def times(self, values, name="timestamps.csv"):
        path = self.directory / name
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream, lineterminator="\n")
            writer.writerow(["frame", "timestamp"])
            writer.writerows(values)
        return path

    def invoke(self, cache, *arguments, success=True, output=None):
        self.run_number += 1
        output = output or self.directory / f"run-{self.run_number}"
        process = subprocess.run(
            [str(self.executable), "--detections", str(cache), "--output", str(output),
             *map(str, arguments)],
            cwd=ROOT, env=self.environment, text=True, capture_output=True, timeout=20,
        )
        if success:
            self.assertEqual(process.returncode, 0, process.stderr + process.stdout)
        else:
            self.assertNotEqual(process.returncode, 0, process.stdout)
        return output, process

    @staticmethod
    def rows(path):
        with path.open(newline="", encoding="utf-8") as stream:
            return list(csv.DictReader(stream))

    @staticmethod
    def detection(frame, score=.9, x=10, timestamp=None):
        return (frame, (frame - 1) / 25 if timestamp is None else timestamp,
                x, 10, 20, 12, score, 0)

    def test_header_only_cache_with_explicit_frame_count(self):
        output, _ = self.invoke(self.cache(), "--frames", 5)
        summary = json.loads((output / "summary.json").read_text())
        self.assertEqual(summary["frames"], 5)
        self.assertEqual(summary["detector_calls"], 5)
        self.assertEqual(len(self.rows(output / "frames.csv")), 5)
        self.assertEqual(self.rows(output / "telemetry.csv"), [])
        self.assertEqual((output / "mot.txt").read_text(), "")

    def test_empty_cache_without_frame_count_is_rejected(self):
        _, process = self.invoke(self.cache(), success=False)
        self.assertIn("frame count", process.stderr.lower())

    def test_zero_frames_infers_maximum_cached_frame(self):
        cache = self.cache([self.detection(1), self.detection(5)])
        output, _ = self.invoke(cache, "--frames", 0)
        self.assertEqual(json.loads((output / "summary.json").read_text())["frames"], 5)

    def test_frame_times_preserve_frames_without_detections(self):
        cache = self.cache([self.detection(1, timestamp=10),
                            self.detection(3, timestamp=10.35)])
        times = self.times([(1, 10), (2, 10.1), (3, 10.35)])
        output, _ = self.invoke(cache, "--frame-times", times, "--min-hits", 1,
                                "--max-age", 2, "--prediction-age", 2)
        frames = self.rows(output / "frames.csv")
        self.assertEqual([float(row["timestamp"]) for row in frames], [10, 10.1, 10.35])
        telemetry = self.rows(output / "telemetry.csv")
        middle = [row for row in telemetry if row["frame"] == "2"]
        self.assertEqual(len(middle), 1)
        self.assertEqual(middle[0]["observed"], "0")
        self.assertEqual(float(middle[0]["timestamp"]), 10.1)

    def test_frame_times_can_define_an_empty_cache_duration(self):
        times = self.times([(1, 0), (2, .1), (3, .3)])
        output, _ = self.invoke(self.cache(), "--frame-times", times)
        self.assertEqual(len(self.rows(output / "frames.csv")), 3)

    def test_incomplete_explicit_frame_times_are_rejected(self):
        times = self.times([(1, 0), (3, .08)])
        self.invoke(self.cache(), "--frames", 3, "--frame-times", times, success=False)

    def test_nonmonotonic_and_duplicate_timestamps_are_rejected(self):
        for values in [[(1, 0), (2, .2), (3, .1)], [(1, 0), (2, 0)],
                       [(1, 0), (1, .1)]]:
            with self.subTest(values=values):
                times = self.times(values)
                self.invoke(self.cache(), "--frames", 3, "--frame-times", times, success=False)

    def test_invalid_configuration_is_rejected(self):
        cache = self.cache([self.detection(1)])
        cases = [("--schedule", "mystery"), ("--frames", "-1"), ("--fps", "0"),
                 ("--risk-threshold", "nan"), ("--low", ".9", "--high", ".2"),
                 ("--score", "1.2"), ("--iou", "0"), ("--min-hits", "0")]
        for arguments in cases:
            with self.subTest(arguments=arguments):
                self.invoke(cache, *arguments, success=False)

    def test_invalid_detection_values_are_rejected(self):
        for row in [self.detection(0), (1, 0, 10, 10, 0, 12, .9, 0),
                    (1, 0, "nan", 10, 20, 12, .9, 0), self.detection(1, score=1.1)]:
            with self.subTest(row=row):
                self.invoke(self.cache([row]), "--frames", 2, success=False)

    def test_fixed_schedule_emits_predictions_and_correct_summary(self):
        cache = self.cache([self.detection(frame, x=10 + frame) for frame in range(1, 5)])
        output, _ = self.invoke(cache, "--frames", 4, "--schedule", "fixed", "--interval", 4,
                                "--min-hits", 1, "--prediction-age", 2, "--max-age", 2)
        frames = self.rows(output / "frames.csv")
        telemetry = self.rows(output / "telemetry.csv")
        self.assertEqual([row["detector_ran"] for row in frames], ["1", "0", "0", "0"])
        self.assertEqual([row["observed"] for row in telemetry], ["1", "0", "0", "0"])
        self.assertEqual({row["id"] for row in telemetry}, {"1"})
        self.assertTrue(all(row["state"] == "confirmed" for row in telemetry))
        mot = (output / "mot.txt").read_text().splitlines()
        self.assertEqual([int(row.split(",")[0]) for row in mot], [1, 2, 3, 4])
        summary = json.loads((output / "summary.json").read_text())
        self.assertEqual(summary["mode"], "cached_detection_replay")
        self.assertEqual(summary["detector_calls"], 1)
        self.assertEqual(summary["detector_rate"], .25)
        self.assertEqual(summary["median_inference_ms"], 0)
        for field in ["processing_fps", "median_frame_ms", "p95_frame_ms", "wall_seconds"]:
            self.assertTrue(math.isfinite(summary[field]) and summary[field] >= 0, field)

    def test_two_stage_recovers_low_confidence_detection(self):
        cache = self.cache([self.detection(1), self.detection(2, score=.15, x=12)])
        sort, _ = self.invoke(cache, "--frames", 2, "--min-hits", 1, "--association", "sort")
        staged, _ = self.invoke(cache, "--frames", 2, "--min-hits", 1, "--association", "two_stage")
        baseline = self.rows(sort / "telemetry.csv")[-1]
        recovered = self.rows(staged / "telemetry.csv")[-1]
        self.assertEqual((baseline["observed"], baseline["state"]), ("0", "lost"))
        self.assertEqual((recovered["observed"], recovered["state"]), ("1", "confirmed"))
        self.assertEqual(recovered["id"], "1")

    def make_images(self, count=8):
        try:
            import cv2
            import numpy as np
        except ImportError:
            self.skipTest("Image causal tests require numpy and opencv-python")
        images = self.directory / "images"
        images.mkdir(exist_ok=True)
        for frame in range(1, count + 1):
            image = np.zeros((120, 160, 3), dtype=np.uint8)
            cv2.rectangle(image, (20 + frame, 40), (40 + frame, 52), (255, 255, 255), -1)
            if frame >= 6:
                cv2.rectangle(image, (110, 20), (125, 35), (255, 255, 255), -1)
            self.assertTrue(cv2.imwrite(str(images / f"{frame:06d}.png"), image))
        return images

    def test_future_cache_rows_do_not_influence_earlier_adaptive_decisions(self):
        images = self.make_images()
        common = [self.detection(frame, x=20 + frame) for frame in range(1, 5)]
        original = self.cache(common + [self.detection(frame, x=20 + frame) for frame in range(5, 9)], "a.csv")
        altered = self.cache(common + [self.detection(frame, score=.1, x=130) for frame in range(5, 9)], "b.csv")
        args = ["--images", images, "--frames", 8, "--schedule", "adaptive", "--min-hits", 1,
                "--risk-threshold", .5, "--novelty-threshold", .001, "--max-gap", 4]
        first, _ = self.invoke(original, *args)
        second, _ = self.invoke(altered, *args)
        keys = ["frame", "timestamp", "detector_ran", "uncertainty", "ambiguity", "novelty"]
        prefix = lambda output: [[row[key] for key in keys] for row in self.rows(output / "frames.csv")[:4]]
        self.assertEqual(prefix(first), prefix(second), "scheduler consumed future detection information")

    def test_duplicate_image_stems_are_rejected(self):
        images = self.make_images(count=1)
        shutil.copyfile(images / "000001.png", images / "000001.jpg")
        self.invoke(self.cache(), "--images", images, "--frames", 2, success=False)

    def test_detection_input_cannot_be_overwritten_by_outputs(self):
        cache = self.cache([self.detection(1)], "source/detections.csv")
        original = cache.read_bytes()
        self.invoke(cache, "--frames", 1, output=cache.parent, success=False)
        self.assertEqual(cache.read_bytes(), original)

    def test_frame_times_input_cannot_be_overwritten_by_outputs(self):
        cache = self.cache([self.detection(1)])
        times = self.times([(1, 0)], "source/frames.csv")
        original = times.read_bytes()
        self.invoke(cache, "--frames", 1, "--frame-times", times, output=times.parent, success=False)
        self.assertEqual(times.read_bytes(), original)

    def test_annotated_video_cannot_overwrite_source_video(self):
        source = ROOT / "test_video.mp4"
        if not source.is_file():
            self.skipTest("Bundled sample video absent")
        target = self.directory / "source.mp4"
        shutil.copyfile(source, target)
        before = hashlib.sha256(target.read_bytes()).digest()
        cache = self.cache([self.detection(1)])
        self.invoke(cache, "--frames", 1, "--video", target, "--output-video", target, success=False)
        self.assertEqual(hashlib.sha256(target.read_bytes()).digest(), before)


if __name__ == "__main__":
    unittest.main()
