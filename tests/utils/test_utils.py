import unittest
import numpy as np
from utils.utils import *

class TestUtils(unittest.TestCase):

    def setUp(self):
        self.number_points = 10
        self.number_classes = 3
        self.dim = 10
        self.data = {j: np.random.rand(np.random.randint(10, 100), self.dim) for j in range(self.number_points)}
        self.labels = {j: np.random.randint(self.number_classes, size=len(self.data[j])) for j in range(self.number_points)}

    def test_flatten(self):
        flatdata, flatlabels = flatten(self.data, self.labels)
       
        self.assertEqual(flatdata.shape[0], np.sum([len(self.data[d]) for d in self.data]))
        self.assertEqual(flatdata.shape[1], self.dim)

        self.assertEqual(len(flatlabels), len(flatdata))

if __name__ == '__main__':
    unittest.main()