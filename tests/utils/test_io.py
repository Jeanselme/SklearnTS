import unittest
import numpy as np
import pandas as pd
from utils.io import *

class TestIo(unittest.TestCase):

    def setUp(self):
        self.number_points = 10
        self.number_classes = 3
        self.dim = 10
        self.data = {str(j): pd.DataFrame(np.random.rand(np.random.randint(10, 100), self.dim)) for j in range(self.number_points)}
        self.path = "tests/fake_data/"

    def test_writeParallel(self):
        writeParallel(self.data, self.path, 3)

    def test_readParallel(self):
        writeParallel(self.data, self.path, 3)
        data = readParallel(self.path, 3)
        self.assertEqual(len(data), len(self.data))
        for d in data:
            self.assertEqual(len(data[d]), len(self.data[d]))

    def test_extractParallel(self):
        writeParallel(self.data, self.path, 3)
        data = extractParallel(self.path, lambda x: x.pow(2), 3)
        self.assertEqual(len(data), len(self.data))
        for d in data:
            self.assertEqual(len(data[d]), len(self.data[d]))

if __name__ == '__main__':
    unittest.main()