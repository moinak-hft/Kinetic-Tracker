"""Black-box motion/scene integration checks using tiny training-free fixtures."""
import csv
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
HEADER = ["frame", "timestamp", "x", "y", "w", "h", "score", "class_id"]


class RefinementCLITests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        override = os.environ.get("KINETIC_TRACKER_EXE")
        candidates = ([Path(override)] if override else []) + [
            ROOT / "build" / "kinetic_tracker.exe", ROOT / "build" / "kinetic_tracker",
            ROOT / "build" / "Release" / "kinetic_tracker.exe",
        ]
        cls.executable = next((path for path in candidates if path.is_file()), None)
        if cls.executable is None:
            raise unittest.SkipTest("Build kinetic_tracker or set KINETIC_TRACKER_EXE")
        cls.environment = os.environ.copy()
        for value in [os.environ.get("KINETIC_TOOLCHAIN"), "D:/Tools/msys64/ucrt64", "C:/msys64/ucrt64"]:
            if value and (Path(value) / "bin").is_dir():
                cls.environment["PATH"] = str(Path(value) / "bin") + os.pathsep + cls.environment.get("PATH", "")
                break

    def setUp(self):
        (ROOT / "work").mkdir(exist_ok=True)
        self.scratch = tempfile.TemporaryDirectory(prefix="refinement cli-", dir=ROOT / "work")
        self.addCleanup(self.scratch.cleanup)
        self.folder = Path(self.scratch.name)
        self.counter = 0

    def cache(self, rows, filename="input.csv"):
        path = self.folder / filename
        with path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream, lineterminator="\n")
            writer.writerow(HEADER)
            writer.writerows(rows)
        return path

    @staticmethod
    def detection(frame, x=10, timestamp=None, score=.9):
        return (frame, (frame - 1) / 25 if timestamp is None else timestamp,
                x, 20, 20, 12, score, 0)

    def invoke(self, cache, *arguments, success=True):
        self.counter += 1
        output = self.folder / f"output-{self.counter}"
        result = subprocess.run(
            [str(self.executable), "--detections", str(cache), "--output", str(output), *map(str, arguments)],
            cwd=ROOT, env=self.environment, capture_output=True, text=True, timeout=25,
        )
        if success:
            self.assertEqual(result.returncode, 0, result.stderr + result.stdout)
        else:
            self.assertNotEqual(result.returncode, 0, result.stdout)
        return output, result

    @staticmethod
    def rows(path):
        with path.open(newline="", encoding="utf-8") as stream:
            return list(csv.DictReader(stream))

    def test_motion_recovers_disjoint_observation_through_cli(self):
        cache = self.cache([self.detection(1, x=0, timestamp=0), self.detection(2, x=35, timestamp=.25)])
        common = ["--frames", 2, "--fps", 4, "--min-hits", 1, "--max-age", 2, "--prediction-age", 1]
        motion, _ = self.invoke(cache, *common, "--association", "motion")
        baseline, _ = self.invoke(cache, *common, "--association", "two_stage")
        current = [row for row in self.rows(motion / "telemetry.csv") if row["frame"] == "2"]
        reference = [row for row in self.rows(baseline / "telemetry.csv") if row["frame"] == "2"]
        self.assertEqual([(row["id"], row["observed"]) for row in current], [("1", "1")])
        self.assertTrue(any(row["id"] == "2" and row["observed"] == "1" for row in reference))
        self.assertTrue(any(row["id"] == "1" and row["state"] == "lost" for row in reference))

    def test_exposed_configuration_roundtrips_to_summary(self):
        cache = self.cache([self.detection(1)])
        expected = {
            "motion_gate": 11.2, "motion_mahalanobis_weight": .2, "motion_direction_weight": .3,
            "motion_recovery": False, "motion_max_center_distance": 2.5, "motion_cost_limit": .95,
            "motion_ambiguity_margin": .04, "scene_method": "robust", "risk_scope": "active",
            "scene_width": 160, "scene_alpha": .12, "scene_pixel_threshold": 16,
            "scene_min_area": 3, "scene_translation": False,
        }
        arguments = ["--frames", 1, "--association", "motion"]
        for name, value in expected.items():
            arguments.extend(["--" + name.replace("_", "-"), int(value) if isinstance(value, bool) else value])
        output, _ = self.invoke(cache, *arguments)
        summary = json.loads((output / "summary.json").read_text())
        self.assertEqual(summary["association"], "motion")
        for name, value in expected.items():
            with self.subTest(name=name):
                self.assertEqual(summary[name], value)

    def test_invalid_motion_parameters_rejected_with_legacy_association(self):
        cache = self.cache([self.detection(1)])
        invalid = [
            ("--motion-gate", "nan"), ("--motion-gate", "-1"),
            ("--motion-mahalanobis-weight", "-.1"), ("--motion-mahalanobis-weight", "1.1"),
            ("--motion-direction-weight", "nan"), ("--motion-direction-weight", "-.1"),
            ("--motion-max-center-distance", "0"), ("--motion-cost-limit", "0"),
            ("--motion-ambiguity-margin", "-.1"), ("--motion-recovery", "true"),
            ("--scene-translation", "2"), ("--risk-scope", "future"),
        ]
        for arguments in invalid:
            with self.subTest(arguments=arguments):
                self.invoke(cache, "--frames", 1, "--association", "two_stage", *arguments, success=False)

    def test_active_scope_excludes_tentative_tracks_and_is_logged(self):
        cache = self.cache([self.detection(frame) for frame in range(1, 5)])
        common = ["--frames", 4, "--schedule", "covariance", "--risk-threshold", .9,
                  "--min-hits", 2, "--max-gap", 8]
        all_tracks, _ = self.invoke(cache, *common, "--risk-scope", "all")
        active, _ = self.invoke(cache, *common, "--risk-scope", "active")
        every_risk = self.rows(all_tracks / "frames.csv")
        active_risk = self.rows(active / "frames.csv")
        self.assertLess(float(every_risk[1]["uncertainty"]), 1)
        self.assertEqual(float(active_risk[1]["uncertainty"]), 1)
        self.assertEqual(every_risk[1]["detector_ran"], "0")
        self.assertEqual(active_risk[1]["detector_ran"], "1")
        self.assertEqual(json.loads((active / "summary.json").read_text())["risk_scope"], "active")

    def test_active_scope_retains_confirmed_deliberate_skip_predictions(self):
        cache = self.cache([self.detection(frame) for frame in range(1, 5)])
        common = ["--frames", 4, "--schedule", "fixed", "--interval", 4,
                  "--min-hits", 1, "--prediction-age", 1]
        outputs = [self.invoke(cache, *common, "--risk-scope", scope)[0] for scope in ("all", "active")]
        first, second = [self.rows(path / "frames.csv") for path in outputs]
        for frame in range(4):
            self.assertEqual(first[frame]["uncertainty"], second[frame]["uncertainty"])
            self.assertEqual(first[frame]["ambiguity"], second[frame]["ambiguity"])
        predictions = self.rows(outputs[1] / "telemetry.csv")
        self.assertEqual([row["observed"] for row in predictions], ["1", "0", "0", "0"])

    def test_robust_scene_requires_images_only_when_adaptive_is_used(self):
        cache = self.cache([self.detection(1)])
        output, _ = self.invoke(cache, "--frames", 1, "--scene-method", "robust", "--schedule", "every")
        self.assertEqual(json.loads((output / "summary.json").read_text())["frames"], 1)
        _, failure = self.invoke(cache, "--frames", 1, "--scene-method", "robust", "--schedule", "adaptive", success=False)
        self.assertIn("input frames", failure.stderr)

    def make_images(self, name, changed_future=False):
        try:
            import cv2
            import numpy as np
        except ImportError:
            self.skipTest("raw-frame scene test needs numpy and OpenCV Python")
        directory = self.folder / name
        directory.mkdir()
        for frame in range(1, 9):
            image = np.zeros((96, 128, 3), dtype=np.uint8)
            image[::8, :, :] = 30
            image[:, ::8, :] = 30
            cv2.rectangle(image, (15 + frame, 20), (35 + frame, 32), (220, 220, 220), -1)
            if frame >= 5:
                cv2.rectangle(image, (85, 60), (95, 70), (240, 240, 240), -1)
            if changed_future and frame >= 6:
                image[:] = (150, 20, 80)
            self.assertTrue(cv2.imwrite(str(directory / f"{frame:06d}.png"), image))
        return directory

    def test_robust_scene_and_tracker_cues_precede_current_detection_rows(self):
        images = self.make_images("images-a")
        altered_images = self.make_images("images-b", changed_future=True)
        original_rows = [self.detection(frame, x=15 + frame) for frame in range(1, 9)]
        altered_rows = original_rows[:4]
        for frame in range(5, 9):
            altered_rows.extend([self.detection(frame, x=80), self.detection(frame, x=81)])
        original = self.cache(original_rows, "original.csv")
        altered = self.cache(altered_rows, "altered.csv")
        common = ["--frames", 8, "--association", "motion", "--schedule", "adaptive",
                  "--scene-method", "robust", "--scene-width", 128, "--risk-scope", "active",
                  "--min-hits", 1, "--target-rate", 1, "--risk-threshold", 0]
        first, _ = self.invoke(original, *common, "--images", images)
        second, _ = self.invoke(altered, *common, "--images", altered_images)
        left, right = self.rows(first / "frames.csv"), self.rows(second / "frames.csv")
        fields = ["frame", "timestamp", "detector_ran", "uncertainty", "ambiguity", "novelty",
                  "scene_raw_novelty", "scene_global_change", "scene_reset", "scene_warming"]
        # Frame 5 cache rows already differ; its cues must still be based only on
        # earlier track observations. Images after frame 5 also differ, so no
        # future cache/pixel content may alter these prefix decisions.
        self.assertEqual([[row[key] for key in fields] for row in left[:5]],
                         [[row[key] for key in fields] for row in right[:5]])
        self.assertEqual(float(left[0]["uncertainty"]), 1)
        self.assertEqual(float(left[0]["ambiguity"]), 0)
        self.assertEqual(left[0]["scene_warming"], "1")
        self.assertNotEqual(left[4]["detections"], right[4]["detections"],
                            "adversarial cache change must actually reach the detector stage")
        self.assertNotEqual(left[5]["ambiguity"], right[5]["ambiguity"],
                            "subsequent cues should reflect the changed past observations")


if __name__ == "__main__":
    unittest.main()
