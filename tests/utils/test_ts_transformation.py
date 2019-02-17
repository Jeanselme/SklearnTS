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
        self.assertEqual(res.index.max(), - self.ts.index.min())
        self.assertEqual(res.index.min(), - self.ts.index.max())

if __name__ == '__main__':
    unittest.main()