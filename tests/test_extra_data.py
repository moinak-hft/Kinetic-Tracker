"""Structural checks for the confirmation-only data preparation path."""
import importlib.util
import contextlib
import copy
import io
import json
import hashlib
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from prepare_extra_data import SELECTION, MAX_DOWNLOAD_BYTES, validate_response, parse_labels
from prepare_data import GMOT_TOTAL_BYTES
import refinement_study


class ConfirmationDataTests(unittest.TestCase):
    def test_exact_http_range_required(self):
        valid = {"Content-Range": f"bytes 10-19/{GMOT_TOTAL_BYTES}", "Content-Length": "10"}
        validate_response(206, valid, 10, 19)
        for status, headers in [(200, valid), (206, {"Content-Range": "bytes 10-19/999"}),
                                (206, {**valid, "Content-Length": "11"})]:
            with self.assertRaises(ValueError):
                validate_response(status, headers, 10, 19)

    def test_predeclared_nine_clips_within_download_limit(self):
        self.assertEqual(set(SELECTION), {"airplane-2", "ball-3", "balloon-2", "bird-2", "boat-2", "car-2", "fish-2", "person-0", "stock-1"})
        self.assertEqual(sum(item["frames"] for item in SELECTION.values()), 1781)
        self.assertLess(sum(item["end"] - item["start"] + 1 for item in SELECTION.values()), MAX_DOWNLOAD_BYTES)

    def test_identity_mapping_preserves_geometry_and_empty_frames(self):
        mapping = {0: 1, 1: 2, 2: 3}
        rows, identities = parse_labels("0,9,1,2,3,4\n2,4,5,6,7,8\n", mapping, 25, True)
        self.assertEqual(identities, {4: 1, 9: 2})
        self.assertEqual(rows, [(1, 2, 1.0, 2.0, 3.0, 4.0, 1, 1, 1),
                                (3, 1, 5.0, 6.0, 7.0, 8.0, 1, 1, 1)])

    def test_invalid_or_duplicate_annotations_rejected(self):
        bad = ["0,1,0,0,10,10\n0,1,1,1,10,10", "5,1,0,0,10,10",
               "0,1,0,0,0,10", "0,1,nan,0,10,10"]
        for text in bad:
            with self.subTest(text=text), self.assertRaises(ValueError):
                parse_labels(text, {0: 1}, 25, True)
        with self.assertRaises(ValueError):
            parse_labels("0,-1,0,0,10,10,1.1", {0: 1}, 25, False)

    def test_author_detector_confidence_and_coordinates_preserved(self):
        rows, identities = parse_labels("0,-1,-2,3,4,5,0.4\n", {0: 1}, 25, False)
        self.assertEqual(identities, {})
        self.assertEqual(rows, [(1, "0.00000000", -2.0, 3.0, 4.0, 5.0, 0.4, 0)])


class RefinementProtocolTests(unittest.TestCase):
    def selection_fixture(self, motion_raw):
        protocol = {"primary_output_policy": "active",
                    "profiles": {"two_stage": {"selection_eligible": True}, "motion_full": {"selection_eligible": True}},
                    "validation_groups": {"synthetic": {"weight": .5, "cases": [{"path": "data/synthetic/s0"}]},
                                          "raw_insect": {"weight": .5, "cases": [{"path": "data/gmot40/insect-1"}]}},
                    "selection": {"reference_profile": "two_stage", "raw_guard_group": "raw_insect", "max_raw_hota_drop": .005}}
        records = []
        for profile, values in (("two_stage", (.5, .2)), ("motion_full", (.9, motion_raw))):
            for case, score in zip([{"path": "data/synthetic/s0"}, {"path": "data/gmot40/insect-1"}], values):
                records.append({"case": case, "profile": profile, "output_policy": "active", "schedule_name": "every",
                                "metrics": {name: score for name in refinement_study.SCORES}})
        return protocol, records

    def test_raw_nonregression_guard_can_reject_better_balanced_candidate(self):
        protocol, records = self.selection_fixture(.19)
        selection = refinement_study.select_validation(protocol, records)
        self.assertEqual(selection["selected_profile"], "two_stage")
        self.assertNotIn("motion_full", selection["raw_guard_accepted_profiles"])
        protocol, records = self.selection_fixture(.196)
        self.assertEqual(refinement_study.select_validation(protocol, records)["selected_profile"], "motion_full")

    def test_incomplete_validation_cannot_create_selection(self):
        protocol, records = self.selection_fixture(.2)
        with self.assertRaisesRegex(ValueError, "incomplete validation"):
            refinement_study.select_validation(protocol, records[:-1])

    def test_fixed2_selection_reuses_same_objective_without_mutating_protocol(self):
        protocol, records = self.selection_fixture(.196)
        original = copy.deepcopy(protocol)
        for record in records:
            record["schedule_name"] = "fixed2"
        result = refinement_study.select_validation(protocol, records, schedule_name="fixed2")
        self.assertEqual(result["selected_profile"], "motion_full")
        self.assertEqual(result["selection"]["schedule"], "fixed2")
        self.assertEqual(protocol, original)

    def test_confirmation_fails_before_freeze_without_scoring_or_outputs(self):
        with tempfile.TemporaryDirectory(prefix="kinetic-freeze-guard-") as temporary:
            output = Path(temporary) / "unfrozen_study"
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as failed:
                refinement_study.main(["--phase", "confirmation", "--output", str(output)])
            self.assertEqual(failed.exception.code, 2)
            self.assertFalse(output.exists())

    def test_runner_rejects_source_directory_outputs_before_mkdir_or_execution(self):
        with tempfile.TemporaryDirectory(prefix="kinetic-output-guard-") as temporary:
            root = Path(temporary)
            sequence = root / "data/case"
            sequence.mkdir(parents=True)
            (root / "work").mkdir()
            (root / "work/fake_tracker").write_bytes(b"never execute this fixture")
            metadata = {"name": "case", "num_frames": 1, "fps": 25,
                        "detections": "detections.csv", "ground_truth": "gt.txt",
                        "timestamp_file": "frames.csv", "images": "img1"}
            (sequence / "metadata.json").write_text(json.dumps(metadata))
            for name in ("detections.csv", "gt.txt", "frames.csv"):
                (sequence / name).write_bytes(("protected source " + name).encode())
            original = {name: (sequence / name).read_bytes() for name in ("metadata.json", "detections.csv", "gt.txt", "frames.csv")}
            protocol = {"executables": {"current": "work/fake_tracker"}, "shared_tracker": {},
                        "profiles": {"two_stage": {"executable": "current", "args": {"association": "two_stage"}}}}
            with patch.object(refinement_study, "ROOT", root), patch.object(refinement_study.subprocess, "run") as execute, \
                    patch.object(Path, "mkdir") as mkdir, self.assertRaisesRegex(ValueError, "protected input"):
                refinement_study.run_one(protocol, {"path": "data/case"}, "two_stage", "every", {"schedule": "every"},
                                         "active", sequence, "validation")
            execute.assert_not_called()
            mkdir.assert_not_called()
            self.assertEqual(original, {name: (sequence / name).read_bytes() for name in original})
            self.assertFalse((sequence / "metrics.json").exists())

    def test_summary_artifact_guard_preserves_protocol_and_hardlinked_input_bytes(self):
        with tempfile.TemporaryDirectory(prefix="kinetic-artifact-guard-") as temporary:
            root = Path(temporary)
            protocol = root / "freeze.json"
            protocol.write_bytes(b"protected protocol bytes")
            with self.assertRaisesRegex(ValueError, "protected input"):
                refinement_study.guard_file_writes([root / "freeze.json"], [protocol])
            self.assertEqual(protocol.read_bytes(), b"protected protocol bytes")
            source = root / "ground_truth.txt"
            source.write_bytes(b"protected GT bytes")
            alias = root / "aggregate.json"
            os.link(source, alias)
            with self.assertRaisesRegex(ValueError, "protected input"):
                refinement_study.guard_file_writes([alias], [source])
            self.assertEqual(source.read_bytes(), b"protected GT bytes")
            self.assertEqual(alias.read_bytes(), b"protected GT bytes")


class ActualImageIntegrityTests(unittest.TestCase):
    def image_fixture(self, root):
        sequence = root / "data/case"
        (sequence / "img1").mkdir(parents=True)
        (sequence / "img1/000001.jpg").write_bytes(b"first tiny byte fixture")
        (sequence / "img1/000002.jpg").write_bytes(b"second tiny byte fixture")
        digest = hashlib.sha256(b"first tiny byte fixturesecond tiny byte fixture").hexdigest()
        metadata = {"name": "case", "num_frames": 2, "fps": 25, "images": "img1", "image_content_sha256": digest,
                    "detections": "detections.csv", "ground_truth": "gt.txt", "timestamp_file": "frames.csv"}
        (sequence / "metadata.json").write_text(json.dumps(metadata))
        for name in ("detections.csv", "gt.txt", "frames.csv"):
            (sequence / name).write_bytes(b"hash-only fixture, not evaluated")
        return sequence, metadata

    def test_count_and_escaped_directory_rejected_without_decoding(self):
        with tempfile.TemporaryDirectory(prefix="kinetic-image-integrity-") as temporary:
            root = Path(temporary) / "project"
            sequence, metadata = self.image_fixture(root)
            with patch.object(refinement_study, "ROOT", root):
                checked = refinement_study.verify_image_content(sequence, metadata)
                self.assertEqual(checked["sha256"], metadata["image_content_sha256"])
                with self.assertRaisesRegex(ValueError, "file count"):
                    refinement_study.verify_image_content(sequence, {**metadata, "num_frames": 3})
                with self.assertRaisesRegex(ValueError, "inside the project"):
                    refinement_study.verify_image_content(sequence, {**metadata, "images": "../../../outside"})

    def test_confirmation_catches_changed_jpeg_with_unchanged_metadata(self):
        with tempfile.TemporaryDirectory(prefix="kinetic-confirmation-integrity-") as temporary:
            root = Path(temporary)
            sequence, metadata = self.image_fixture(root)
            (root / "tools").mkdir()
            (root / "tools/evaluate.py").write_bytes(b"hash-only evaluator fixture, never executed")
            (root / "work").mkdir()
            executable = root / "work/fake_tracker"
            executable.write_bytes(b"hash-only executable fixture, never executed")
            case = {"path": "data/case", "source": "provided"}
            protocol = {"executables": {"current": "work/fake_tracker"}, "validation_groups": {}, "confirmation_cases": [case]}
            protocol_path = root / "protocol.json"
            protocol_path.write_text(json.dumps(protocol))
            output = root / "runs/study"
            output.mkdir(parents=True)
            with patch.object(refinement_study, "ROOT", root):
                frozen = {"frozen": True, "protocol": protocol,
                          "protocol_sha256": refinement_study.sha256(protocol_path),
                          "runner_sha256": refinement_study.sha256(Path(refinement_study.__file__)),
                          "evaluator_sha256": refinement_study.sha256(root / "tools/evaluate.py"),
                          "executable_hashes": {"current": refinement_study.sha256(executable)},
                          "confirmation_inputs": {refinement_study.case_key(case): refinement_study.case_inputs(case)[-1]},
                          "image_integrity": {refinement_study.case_key(case): refinement_study.verify_image_content(sequence, metadata)}}
                (output / "freeze.json").write_text(json.dumps(frozen))
                refinement_study.check_freeze(protocol_path, output)
                original_metadata = (sequence / "metadata.json").read_bytes()
                image = sequence / "img1/000002.jpg"
                before = image.stat()
                image.write_bytes(b"Second tiny byte fixture")
                os.utime(image, ns=(before.st_atime_ns, before.st_mtime_ns + 1_000_000_000))
                self.assertEqual((sequence / "metadata.json").read_bytes(), original_metadata)
                with self.assertRaisesRegex(ValueError, "Actual ordered image bytes"):
                    refinement_study.check_freeze(protocol_path, output)


if __name__ == "__main__":
    unittest.main()
