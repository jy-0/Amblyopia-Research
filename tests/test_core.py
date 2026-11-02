import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core import fit_head, load_head, metrics, pair_features, preprocess, read_manifest, score_embeddings


class CoreTests(unittest.TestCase):
    def test_pair_shape_and_exchange(self):
        x = np.random.default_rng(42).normal(size=(5, 2, 384)).astype(np.float32)
        self.assertEqual(pair_features(x).shape, (5, 768))
        np.testing.assert_array_equal(pair_features(x), pair_features(x[:, ::-1]))
        with self.assertRaises(ValueError): pair_features(np.zeros((3, 2, 3)))

    def test_preprocessing_and_invalid_crop(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / 'synthetic.png'
            Image.new('RGB', (20, 40), 'white').save(p)
            x = preprocess(p)
            self.assertEqual(tuple(x.shape), (3, 392, 392))
            self.assertEqual(float(x.min()), -1.)
            self.assertEqual(float(x.max()), 1.)
            with self.assertRaises(ValueError): preprocess(p, [0, 0, 30, 40])

    def test_manifest_has_unique_patient_ids(self):
        p = Path(__file__).resolve().parents[1] / 'examples/manifest.example.csv'
        self.assertEqual(len(read_manifest(p)), 6)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'bad.csv'
            path.write_text(p.read_text().replace('example_val_001', 'example_train_001'))
            with self.assertRaises(ValueError): read_manifest(path)

    def test_test_targets_cannot_change_fits(self):
        eyes = np.random.default_rng(4).normal(size=(24, 2, 384)).astype(np.float32)
        y = np.tile([0, 1], 12)
        split = np.array(['train'] * 12 + ['val'] * 6 + ['test'] * 6)
        changed = y.copy(); changed[split == 'test'] = 1 - changed[split == 'test']
        with patch('core.C_GRID', (.01,)), patch('core.GAMMA_GRID', ('scale',)):
            for classifier in ('logistic', 'rbf_svm'):
                a, ra, sa = fit_head(eyes, y, split, classifier)
                b, rb, sb = fit_head(eyes, changed, split, classifier)
                self.assertEqual(ra['parameters'], rb['parameters'])
                self.assertEqual(ra['thresholds'], rb['thresholds'])
                np.testing.assert_array_equal(sa, sb)

    def test_published_heads_and_decision_scores(self):
        root = Path(__file__).resolve().parents[1]
        eyes = np.zeros((2, 2, 384), np.float32)
        for classifier in ('logistic', 'rbf_svm'):
            artifact = load_head(root / f'weights/{classifier}_seed42.joblib')
            self.assertEqual(artifact['metadata']['seed'], 42)
            score = score_embeddings(artifact, eyes)
            self.assertEqual(score.shape, (2,))
            self.assertTrue(np.isfinite(score).all())


if __name__ == '__main__':
    unittest.main()
