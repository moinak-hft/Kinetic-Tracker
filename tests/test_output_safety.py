"""Input preservation and controlled-comparison regressions; no benchmarks run."""
import contextlib
import io
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import benchmark
import evaluate
import run_reference


class OutputSafetyTests(unittest.TestCase):
    def setUp(self):
        (ROOT / "work").mkdir(exist_ok=True)
        self.scratch = tempfile.TemporaryDirectory(prefix="output-safety-", dir=ROOT / "work")
        self.addCleanup(self.scratch.cleanup)
        self.folder = Path(self.scratch.name)
        self.gt = self.folder / "gt.txt"
        self.pred = self.folder / "pred.txt"
        self.cache = self.folder / "detections.csv"
        self.times = self.folder / "frames.csv"
        self.gt.write_text("1,1,10,10,20,12,1,1,1\n", encoding="utf-8")
        self.pred.write_text("1,7,10,10,20,12,.9,-1,-1,-1\n", encoding="utf-8")
        self.cache.write_text("frame,timestamp,x,y,w,h,score,class_id\n1,0,10,10,20,12,.9,0\n", encoding="utf-8")
        self.times.write_text("frame,timestamp\n1,0\n", encoding="utf-8")

    def reference_args(self, output=None, manifest=None):
        return SimpleNamespace(
            algorithm="ocsort", detections=self.cache, frame_times=self.times,
            output=output or self.folder / "new" / "mot.txt", manifest=manifest,
            frames=1, fps=25, track_thresh=.35, min_hits=1, max_age=20,
            delta_t=3, iou_threshold=.1, match_thresh=.8, class_id=None,
            inertia=.2, ocsort_use_byte=False,
        )

    def reject_evaluation(self, destination):
        before_gt, before_pred = self.gt.read_bytes(), self.pred.read_bytes()
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as raised:
            evaluate.main(["--gt", str(self.gt), "--pred", str(self.pred),
                           "--frames", "1", "--output", str(destination)])
        self.assertEqual(raised.exception.code, 2)
        self.assertEqual(self.gt.read_bytes(), before_gt)
        self.assertEqual(self.pred.read_bytes(), before_pred)

    def test_evaluation_preserves_gt_and_prediction_inputs(self):
        self.reject_evaluation(self.gt)
        self.reject_evaluation(self.pred)
        self.reject_evaluation(self.folder / "missing" / ".." / "gt.txt")

    def test_evaluation_rejects_hardlink_alias(self):
        alias = self.folder / "alias.json"
        try:
            os.link(self.gt, alias)
        except OSError as error:
            self.skipTest(f"hard links unavailable: {error}")
        original = alias.read_bytes()
        self.reject_evaluation(alias)
        self.assertEqual(alias.read_bytes(), original)

    def test_evaluation_rejects_symlink_alias(self):
        alias = self.folder / "alias.json"
        try:
            alias.symlink_to(self.pred)
        except OSError as error:
            self.skipTest(f"symbolic links unavailable: {error}")
        self.reject_evaluation(alias)

    def test_distinct_generated_evaluation_output_can_be_replaced(self):
        output = self.folder / "metrics.json"
        output.write_text("old generated result", encoding="utf-8")
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(evaluate.main(["--gt", str(self.gt), "--pred", str(self.pred),
                                            "--frames", "1", "--output", str(output)]), 0)
        result = json.loads(output.read_text())
        self.assertEqual(result["HOTA"], 1)
        self.assertEqual(result["IDF1"], 1)

    def test_reference_rejects_input_as_prediction_or_manifest(self):
        for output, manifest in [(self.cache, None), (self.times, None),
                                 (self.folder / "new" / "mot.txt", self.cache),
                                 (self.folder / "new" / "mot.txt", self.times)]:
            with self.subTest(output=output, manifest=manifest):
                original_cache, original_times = self.cache.read_bytes(), self.times.read_bytes()
                with patch.object(run_reference, "make_tracker") as factory:
                    with self.assertRaisesRegex(ValueError, "overwrite input"):
                        run_reference.run(self.reference_args(output, manifest))
                    factory.assert_not_called()
                self.assertEqual(self.cache.read_bytes(), original_cache)
                self.assertEqual(self.times.read_bytes(), original_times)
                self.assertFalse((self.folder / "new").exists(), "guard must run before mkdir")

    def test_reference_duplicate_outputs_are_rejected_before_mkdir(self):
        output = self.folder / "new" / "mot.txt"
        with self.assertRaisesRegex(ValueError, "same file"):
            run_reference.run(self.reference_args(output, output))
        self.assertFalse(output.parent.exists())

    def test_reference_rejects_hardlinked_manifest_input(self):
        alias = self.folder / "manifest.json"
        try:
            os.link(self.cache, alias)
        except OSError as error:
            self.skipTest(f"hard links unavailable: {error}")
        original = self.cache.read_bytes()
        with self.assertRaisesRegex(ValueError, "overwrite input"):
            run_reference.run(self.reference_args(manifest=alias))
        self.assertEqual(self.cache.read_bytes(), original)

    def test_guard_rejects_existing_hardlinked_output_aliases(self):
        first, second = self.folder / "first.json", self.folder / "second.json"
        first.write_text("existing generated bytes", encoding="utf-8")
        try:
            os.link(first, second)
        except OSError as error:
            self.skipTest(f"hard links unavailable: {error}")
        with self.assertRaisesRegex(ValueError, "same file"):
            evaluate.guard_output_paths([self.cache], [first, second])
        self.assertEqual(first.read_text(), "existing generated bytes")

    def test_benchmark_output_cannot_destroy_sequence_inputs(self):
        metadata = {"name": "fixture", "num_frames": 1, "fps": 25,
                    "timestamp_file": "frames.csv", "detections": "detections.csv",
                    "ground_truth": "gt.txt"}
        (self.folder / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
        before = {path: path.read_bytes() for path in (self.cache, self.times, self.gt)}
        with patch.object(benchmark.subprocess, "run") as process:
            with self.assertRaisesRegex(ValueError, "overwrite input"):
                benchmark.run_one(self.folder, {"name": "x", "schedule": "every"}, self.folder)
            process.assert_not_called()
        for path, contents in before.items():
            self.assertEqual(path.read_bytes(), contents)

    def test_benchmark_aggregate_cannot_destroy_config(self):
        root = self.folder / "project"
        (root / "configs").mkdir(parents=True)
        (root / "configs" / "splits.json").write_text(
            json.dumps({"synthetic": {"validation": []}}), encoding="utf-8")
        output = root / "runs"
        output.mkdir()
        config = output / "aggregate.json"
        config.write_text("[]\n", encoding="utf-8")
        with patch.object(benchmark, "ROOT", root), patch.object(
                sys, "argv", ["benchmark.py", "--configs", str(config), "--output", str(output)]):
            with self.assertRaisesRegex(ValueError, "overwrite input"):
                benchmark.main()
        self.assertEqual(config.read_text(), "[]\n")

    def test_matched_control_preserves_tracker_and_output_configuration(self):
        adaptive = {"name": "custom", "schedule": "adaptive", "association": "two_stage",
                    "output_policy": "active", "high": .6, "low": .2, "iou": .3,
                    "min_hits": 4, "max_age": .9, "prediction_age": .1,
                    "target_rate": .5, "risk_threshold": .28, "max_gap": 8}
        before = dict(adaptive)
        matched = benchmark.matched_uniform_config(adaptive, 7)
        self.assertEqual(adaptive, before, "helper must not mutate the adaptive treatment")
        self.assertEqual(matched["name"], "matched_custom")
        self.assertEqual(matched["schedule"], "uniform")
        self.assertEqual(matched["uniform_calls"], 7)
        for key in adaptive.keys() - {"name", "schedule"}:
            self.assertEqual(matched[key], adaptive[key], f"control changed {key}")


if __name__ == "__main__":
    unittest.main()
