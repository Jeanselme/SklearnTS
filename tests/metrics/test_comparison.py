import unittest
import numpy as np
import pandas as pd
from metrics.comparison import *

class TestComparison(unittest.TestCase):

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

    def test_rocCompare(self):
        rocCompare([("random", self.predictions), ("perfect", self.labels)], self.labels)

    def test_histCompare(self):
        histCompare([("random", self.predictions), ("perfect", self.labels)], self.labels)

    def test_calibrationCompare(self):
        calibrationCompare([("random", self.predictions), ("perfect", self.labels)], self.labels)

    def test_aucEvolutionCompare(self):
        aucEvolutionCompare([("random", self.predictions), ("perfect", self.labels)], self.temporal_labels, classes = {'+':1, '-':0})

    def test_featuresImportanceCompare(self):
        featuresImportanceCompare([("random", np.random.random(size = self.dim)), ("perfect", np.random.random(size = self.dim))], np.arange(self.dim))

if __name__ == '__main__':
    unittest.main()