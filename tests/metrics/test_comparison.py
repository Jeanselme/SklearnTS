import unittest
import numpy as np
from metrics.comparison import *

class TestComparison(unittest.TestCase):

    def setUp(self):
        self.number_points = 100
        self.number_classes = 2
        self.predictions = {j: np.random.randint(self.number_classes, size=np.random.randint(10,100)) for j in range(self.number_points)}
        self.labels = {j: np.random.randint(self.number_classes, size=len(self.predictions[j])) for j in range(self.number_points)}

    def test_rocCompare(self):
        rocCompare([("random", self.predictions), ("perfect", self.labels)], self.labels)

    def test_histCompare(self):
        histCompare([("random", self.predictions), ("perfect", self.labels)], self.labels)

    def test_calibrationCompare(self):
        calibrationCompare([("random", self.predictions), ("perfect", self.labels)], self.labels)

if __name__ == '__main__':
    unittest.main()