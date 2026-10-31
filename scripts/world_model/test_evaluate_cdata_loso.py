"""Focused checks for fold isolation and explicit observation masks."""
import unittest
from dataclasses import replace

import numpy as np

from cdata_loader import STATE_KEYS, build_dataset
from evaluate_cdata_loso import DIM, features, fold_dataset, real_rows


class EvaluationProtocolTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source = build_dataset()

    def test_held_subject_does_not_affect_training_moments(self):
        held = self.source.subjects[0]
        ids = self.source.subjects[1:]
        fold = fold_dataset(self.source, ids)
        corrupted = replace(self.source, measures={k: {sid: v.copy() for sid, v in table.items()}
                                                   for k, table in self.source.measures.items()})
        for key in STATE_KEYS:
            if held in corrupted.measures[key]:
                corrupted.measures[key][held] = {col: 1e9 for col in corrupted.measures[key][held]}
        changed = fold_dataset(corrupted, ids)
        self.assertEqual(fold.m_mean, changed.m_mean)
        self.assertEqual(fold.m_std, changed.m_std)
        self.assertEqual({r['sid'] for r in real_rows(fold)}, set(ids))

    def test_missing_input_has_zero_value_and_zero_mask(self):
        sid = self.source.subjects[0]
        fold = fold_dataset(self.source, self.source.subjects[1:])
        # HF is not measured for G4/G5.
        x = features(fold, sid, 4, 0)
        hf = STATE_KEYS.index('HF')
        self.assertEqual(x[hf], 0.0)
        self.assertEqual(x[DIM + hf], 0.0)
        self.assertTrue(np.isfinite(x).all())
        rows = [r for r in real_rows(replace(fold, subjects=[sid])) if r['group'] == 4]
        self.assertTrue(all(r['valid'][hf] == 0 for r in rows))


if __name__ == '__main__':
    unittest.main()
