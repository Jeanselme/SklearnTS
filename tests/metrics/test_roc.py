import unittest
import numpy as np
import pandas as pd
from metrics.roc import *

class TestROC(unittest.TestCase):

    def setUp(self):
        self.dim = 10
        self.number_points = 100
        self.number_classes = 2
        self.predictions = {j: np.random.randint(self.number_classes, size=np.random.randint(10,100)) for j in range(self.number_points)}
        self.labels = {j: np.random.randint(self.number_classes, size=len(self.predictions[j])) for j in range(self.number_points)}

        self.temporal_labels = [(pd.to_timedelta(i, unit='s'), {j: self.labels[j].copy() for j in self.labels}) for i in range(5, 50)]
        for i, labels in self.temporal_labels:
            for j in labels:
                labels[j][-i.seconds:] = -1

    def test_rocAUC(self):
        self.assertAlmostEqual(aucCompute(self.predictions, self.labels), 0.5, 2)
        self.assertAlmostEqual(aucCompute(self.labels, self.labels), 1.)

if __name__ == '__main__':
    unittest.main()