"""Semantic checks for evaluation and pinned detector-free reference integration."""
import importlib.util
import json
import math
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from evaluate import evaluate_sequence, box_iou, read_mot
import run_reference


class EvaluationTests(unittest.TestCase):
    def setUp(self):
        self.scratch = tempfile.TemporaryDirectory()
        self.folder = Path(self.scratch.name)

    def tearDown(self):
        self.scratch.cleanup()

    def score(self, gt, pred, frames):
        ground = self.folder / "gt.txt"
        prediction = self.folder / "pred.txt"
        ground.write_text("\n".join(gt) + "\n", encoding="utf-8")
        prediction.write_text("\n".join(pred) + "\n", encoding="utf-8")
        return evaluate_sequence(ground, prediction, frames)

    def test_perfect_and_arbitrary_identity_renumbering(self):
        result = self.score([f"{f},7,10,20,12,13,1,1,1" for f in range(1, 5)],
                            [f"{f},45,10,20,12,13,0.2,-1,-1,-1" for f in range(1, 5)], 4)
        for name in ("HOTA", "AssA", "DetA", "IDF1", "MOTA"):
            self.assertAlmostEqual(result[name], 1)
        for name in ("IDSW", "Frag", "FP", "FN"):
            self.assertEqual(result[name], 0)
        self.assertEqual(result["TP"], 4)

    def test_identity_swap_penalized_without_detection_errors(self):
        ground, predicted = [], []
        for frame in range(1, 5):
            ground.extend([f"{frame},1,0,0,10,10,1,1,1", f"{frame},2,100,0,10,10,1,1,1"])
            ids = (10, 20) if frame <= 2 else (20, 10)
            predicted.extend([f"{frame},{ids[0]},0,0,10,10,0.9,-1,-1,-1", f"{frame},{ids[1]},100,0,10,10,0.9,-1,-1,-1"])
        result = self.score(ground, predicted, 4)
        self.assertEqual(result["IDSW"], 2)
        self.assertEqual(result["FP"], 0)
        self.assertEqual(result["FN"], 0)
        self.assertAlmostEqual(result["IDF1"], 0.5)
        self.assertAlmostEqual(result["AssA"], 1 / 3)
        self.assertAlmostEqual(result["HOTA"], math.sqrt(1 / 3))

    def test_missing_middle_frame(self):
        result = self.score([f"{f},1,0,0,10,10,1,1,1" for f in (1, 2, 3)],
                            [f"{f},1,0,0,10,10,1,-1,-1,-1" for f in (1, 3)], 3)
        self.assertEqual(result["FN"], 1)
        self.assertEqual(result["IDSW"], 0)
        self.assertAlmostEqual(result["IDF1"], 0.8)
        self.assertAlmostEqual(result["HOTA"], 2 / 3)
        # Preserve and disclose this upstream CLEAR early-continue limitation.
        self.assertEqual(result["Frag"], 0)
        self.assertIn("Frag", result["metadata"]["limitations"][0])

    def test_trailing_empty_gt_frames_still_count_false_positives(self):
        result = self.score([f"{f},1,0,0,10,10,1,1,1" for f in (1, 2)],
                            [f"{f},1,0,0,10,10,1,-1,-1,-1" for f in (1, 2, 3)], 3)
        self.assertEqual(result["FP"], 1)
        self.assertEqual(result["frames"], 3)

    def test_all_missing_and_both_empty_are_finite(self):
        missing = self.score(["1,1,0,0,10,10,1,1,1"], [], 3)
        self.assertEqual(missing["FN"], 1)
        self.assertEqual(missing["IDF1"], 0)
        both = self.score([], [], 3)
        self.assertEqual(both["HOTA"], 0)
        json.dumps(both, allow_nan=False)

    def test_unmarked_gt_removed_without_ignore_region_suppression(self):
        result = self.score(["1,1,0,0,10,10,0,1,1"], ["1,1,0,0,10,10,1,-1,-1,-1"], 1)
        self.assertEqual(result["FP"], 1)
        self.assertEqual(result["metadata"]["unmarked_gt_rows_removed"], 1)

    def test_duplicate_ids_and_bad_geometry_rejected(self):
        with self.assertRaisesRegex(ValueError, "duplicate"):
            self.score(["1,1,0,0,10,10,1,1,1"] * 2, [], 1)
        with self.assertRaisesRegex(ValueError, "positive width"):
            self.score(["1,1,0,0,0,10,1,1,1"], [], 1)
        with self.assertRaisesRegex(ValueError, "outside"):
            self.score(["2,1,0,0,10,10,1,1,1"], [], 1)
        with self.assertRaisesRegex(ValueError, "never silently dropped"):
            self.score(["1,1,0,0,10,10,1,4,1"], [], 1)

    def test_continuous_iou_and_threshold_boundary(self):
        self.assertAlmostEqual(box_iou([[0, 0, 3, 3]], [[1, 0, 3, 3]])[0, 0], 0.5)
        result = self.score(["1,1,0,0,3,3,1,1,1"], ["1,1,1,0,3,3,1,-1,-1,-1"], 1)
        self.assertEqual(result["TP"], 1)
        self.assertAlmostEqual(result["IDF1"], 1)


class ReferenceTests(unittest.TestCase):
    def test_inclusive_overlap_adapter_matches_original_loop(self):
        spec = importlib.util.spec_from_file_location("test_bbox_adapter", ROOT / "third_party/ByteTrack/cython_bbox.py")
        adapter = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(adapter)
        generator = np.random.default_rng(42)
        first = generator.uniform(-5, 15, (12, 4))
        second = generator.uniform(-5, 15, (7, 4))
        first[:, 2:] = first[:, :2] + np.abs(first[:, 2:]) + 0.1
        second[:, 2:] = second[:, :2] + np.abs(second[:, 2:]) + 0.1
        expected = np.zeros((len(first), len(second)))
        for n, box in enumerate(first):
            for k, query in enumerate(second):
                iw = min(box[2], query[2]) - max(box[0], query[0]) + 1
                ih = min(box[3], query[3]) - max(box[1], query[1]) + 1
                if iw > 0 and ih > 0:
                    area = (box[2] - box[0] + 1) * (box[3] - box[1] + 1)
                    query_area = (query[2] - query[0] + 1) * (query[3] - query[1] + 1)
                    expected[n, k] = iw * ih / (area + query_area - iw * ih)
        np.testing.assert_allclose(adapter.bbox_overlaps(first, second), expected, rtol=1e-14)
        self.assertEqual(adapter.bbox_overlaps(np.empty((0, 4)), second).shape, (0, 7))

    def test_official_references_perfect_sequence_and_empty_frame(self):
        with tempfile.TemporaryDirectory() as scratch:
            folder = Path(scratch)
            detections = folder / "cache.csv"
            detections.write_text("frame,timestamp,x,y,w,h,score,class_id\n" +
                                  "\n".join(f"{frame},{(frame-1)/25},10,20,12,13,0.9,0" for frame in range(1, 5)) + "\n", encoding="utf-8")
            ground = folder / "gt.txt"
            ground.write_text("\n".join(f"{frame},1,10,20,12,13,1,1,1" for frame in range(1, 5)) + "\n", encoding="utf-8")
            for algorithm in ("ocsort", "bytetrack"):
                with self.subTest(algorithm=algorithm):
                    output = folder / f"{algorithm}.txt"
                    run_reference.main(["--algorithm", algorithm, "--detections", str(detections), "--output", str(output),
                                        "--frames", "5", "--fps", "25", "--min-hits", "1"])
                    result = evaluate_sequence(ground, output, 5)
                    self.assertAlmostEqual(result["HOTA"], 1, places=5)
                    self.assertAlmostEqual(result["IDF1"], 1)
                    self.assertEqual(result["IDSW"], 0)
                    manifest = json.loads(output.with_suffix(".manifest.json").read_text())
                    self.assertEqual(manifest["frames"], 5)
                    self.assertIn("not an end-to-end", manifest["timing_scope"])

    def test_bytetrack_official_lap_threshold_unmatched_semantics(self):
        sys.path.insert(0, str(ROOT / "third_party/ByteTrack"))
        from yolox.tracker import matching
        matches, left, right = matching.linear_assignment(np.array([[0.1, 0.9], [0.9, 0.81]]), 0.8)
        self.assertEqual(matches.tolist(), [[0, 0]])
        self.assertEqual(list(left), [1])
        self.assertEqual(list(right), [1])
        matches, left, right = matching.linear_assignment(np.empty((0, 3)), 0.8)
        self.assertEqual(matches.shape, (0, 2))
        self.assertEqual(list(right), [0, 1, 2])


if __name__ == "__main__":
    unittest.main()
