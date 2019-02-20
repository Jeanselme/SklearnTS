import unittest
import numpy as np
import pandas as pd
from utils.ts_transformation import *
from sklearn.decomposition import PCA
from normalization.global_norm import GlobalNormalizationZScore

class TestTsTransformation(unittest.TestCase):

    def setUp(self):
        self.number_points = 100
        self.dim = 10
        self.ts = pd.DataFrame(np.random.rand(self.number_points, self.dim), index = np.arange(self.number_points))
        self.data = {j: pd.DataFrame(np.random.rand(np.random.randint(10, 100), self.dim)) for j in range(self.number_points)}

    def test_pushZeroTime(self):
        res = pushZeroTime(self.ts)
        self.assertEqual(res.index.max(), self.ts.index.max())
        self.assertEqual(res.index.min(), self.ts.index.min())

    def test_computeMeanStdCount(self):
        res = computeMeanStdCount([self.ts, self.ts])
        self.assertEqual(len(res), len(self.ts))
        self.assertAlmostEqual(res['mean'].mean(), self.ts.mean(axis = 1).mean())

    def test_encapsulation(self):
        pca = Transformation(PCA(2))
        pca_res = pca.fit_transform_dict(self.data)
        self.assertEqual(len(pca_res), len(self.data))
        self.assertEqual(pca_res[0].shape[1], 2)

    def test_accumulator(self):
        acc = Accumulator([Transformation(PCA(2)), GlobalNormalizationZScore()])
        acc.fit_transform_dict(self.data)

if __name__ == '__main__':
    unittest.main()