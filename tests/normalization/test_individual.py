import unittest
import numpy as np
import pandas as pd
from normalization.individual_norm import *

class TestIndividual(unittest.TestCase):

    def setUp(self):
        self.number_points = 10
        self.dim = 10
        self.data = {}
        for i in range(self.number_points):
            data = np.random.rand(np.random.randint(10, 50), self.dim)
            self.data[i] = pd.DataFrame(data, index = pd.to_datetime(np.arange(len(data)), unit = "s"))

    def test_IndividualNormalizationZScore(self):
        norm = IndividualNormalizationZScore(window = '3s') # Normalization given last 3 seconds

        norm.fit(self.data)
        resT = norm.transform_dict(self.data)
        resFT = norm.fit_transform_dict(self.data)

        for i in range(self.number_points):
            self.assertEqual(resT[i].index[0].second, 1)
            self.assertEqual(resT[i].index[-1].second, len(self.data[i]) - 1)
            self.assertListEqual(resT[i].values.tolist(), resFT[i].values.tolist())

    def test_IndividualNormalizationZScoreRobust(self):
        norm = IndividualNormalizationZScoreRobust(window = '3s') # Normalization given last 3 seconds

        norm.fit(self.data)
        resT = norm.transform_dict(self.data)
        resFT = norm.fit_transform_dict(self.data)

        for i in range(self.number_points):
            self.assertEqual(resT[i].index[0].second, 1)
            self.assertListEqual(resT[i].values.tolist(), resFT[i].values.tolist())

if __name__ == '__main__':
    unittest.main()