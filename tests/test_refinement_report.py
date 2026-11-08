"""Protect scientific comparison pairing in the refinement report."""
import copy
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
from refinement_report import budget_confirmation_rows, comparable_metrics, SCORES, COUNTS


class BudgetPairingTests(unittest.TestCase):
    def test_repeated_controls_ignore_destination_path_but_preserve_method_and_score(self):
        original = {'HOTA': .5, 'metadata': {'pred_path': 'studyA/mot.txt', 'pred_sha256': 'same', 'threshold': .5}}
        repeat = copy.deepcopy(original)
        repeat['metadata']['pred_path'] = 'studyB/mot.txt'
        self.assertEqual(comparable_metrics(original), comparable_metrics(repeat))
        self.assertEqual(original['metadata']['pred_path'], 'studyA/mot.txt')
        for changed in [{'HOTA': .6}, {'metadata': {**repeat['metadata'], 'threshold': .6}}]:
            self.assertNotEqual(comparable_metrics(original), comparable_metrics({**repeat, **changed}))

    def fixture(self):
        rows = []
        for name, schedule, calls in [('baseline_v2', 'fixed2', 50), ('motion', 'fixed2', 50), ('motion', 'adaptive', 33)]:
            rows.append({'phase': 'confirmation', 'case': {'path': 'data/category/case'}, 'profile': name,
                         'schedule_name': schedule, 'output_policy': 'active', 'inputs': {'detections_sha256': 'same'},
                         'metadata': {'name': 'case'}, 'summary': {'frames': 100, 'detector_calls': calls},
                         'metrics': {**{k: .5 for k in SCORES}, **{k: 3 for k in COUNTS}}})
        return rows

    def test_complete_policy_keeps_different_calls_and_source_separate(self):
        rows = self.fixture()
        before = copy.deepcopy(rows)
        result = budget_confirmation_rows(rows, 'motion', 'adaptive')
        self.assertEqual(result[0]['budget_adaptive_calls'], 33)
        self.assertEqual(result[0]['baseline_fixed2_calls'], 50)
        self.assertIn('not an equal-call', result[0]['scope'])
        self.assertEqual(rows, before)

    def test_missing_comparator_is_an_error(self):
        with self.assertRaisesRegex(ValueError, 'Incomplete budget'):
            budget_confirmation_rows(self.fixture()[1:], 'motion', 'adaptive')

    def test_changed_detection_stream_is_an_error(self):
        rows = self.fixture()
        rows[2]['inputs'] = {'detections_sha256': 'different'}
        with self.assertRaisesRegex(ValueError, 'changed source'):
            budget_confirmation_rows(rows, 'motion', 'adaptive')


if __name__ == '__main__':
    unittest.main()
