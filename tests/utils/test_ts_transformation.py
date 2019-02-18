import unittest
import numpy as np
import pandas as pd
from utils.ts_transformation import *

class TestTsTransformation(unittest.TestCase):

    def setUp(self):
        self.number_points = 100
        self.dim = 10
        self.ts = pd.DataFrame(np.random.rand(self.number_points, self.dim), index = np.arange(self.number_points))

    def test_pushZeroTime(self):
        res = pushZeroTime(self.ts)
        self.assertEqual(res.index.max(), self.ts.index.max())
        self.assertEqual(res.index.min(), self.ts.index.min())

    def test_computeMeanStdCount(self):
        res = computeMeanStdCount([self.ts, self.ts])
        self.assertEqual(len(res), len(self.ts))
        self.assertAlmostEqual(res['mean'].mean(), self.ts.mean(axis = 1).mean())

if __name__ == '__main__':
    unittest.main()