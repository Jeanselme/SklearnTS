import unittest
import numpy as np
from utils.utils import *

class TestUtils(unittest.TestCase):

    def setUp(self):
        self.number_points = 10
        self.number_classes = 3
        self.dim = 10
        self.data = {j: np.random.rand(np.random.randint(10, 100), self.dim) for j in range(self.number_points)}
        self.labels = {j: np.random.randint(self.number_classes, size=len(self.data[j])) for j in range(self.number_points - 1)}
        self.labels.update({(self.number_points - 1): [self.number_classes] * len(self.data[self.number_points - 1])}) # Add a constant point with not same label

    def test_flatten(self):
        flatdata, flatlabels = flatten(self.data, self.labels)
       
        self.assertEqual(flatdata.shape[0], np.sum([len(self.data[d]) for d in self.data]))
        self.assertEqual(flatdata.shape[1], self.dim)

        self.assertEqual(len(flatlabels), len(flatdata))

        # Test on flatten data
        flatdata, flatlabels = flatten(flatdata, flatlabels)
       
        self.assertEqual(flatdata.shape[0], np.sum([len(self.data[d]) for d in self.data]))
        self.assertEqual(flatdata.shape[1], self.dim)

        self.assertEqual(len(flatlabels), len(flatdata))

    def test_selection(self):
        selecteddata, selectedlabels = selection(self.data, self.labels, {"+": 1, "-": 0})
        flatdata, flatlabels = flatten(selecteddata, selectedlabels)

        self.assertEqual(flatdata.shape[1], self.dim)
        self.assertEqual(len(flatlabels), len(flatdata))
        self.assertNotIn(self.number_classes - 1, flatlabels)

        selecteddata, selectedlabels = selection(self.data, self.labels, [0, 1])
        flatdata, flatlabels = flatten(selecteddata, selectedlabels)

        self.assertEqual(flatdata.shape[1], self.dim)
        self.assertEqual(len(flatlabels), len(flatdata))
        self.assertNotIn(self.number_classes - 1, flatlabels)

if __name__ == '__main__':
    unittest.main()