import unittest
import numpy as np
from models.modelts import *
from models.cross_validation import *
from normalization.individual_norm import IndividualNormalizationZScore
from sklearn.linear_model import LogisticRegression

class TestCrossValidation(unittest.TestCase):

    def setUp(self):
        self.number_points = 10
        self.number_classes = 3
        self.dim = 10
        self.data = {j: np.random.rand(np.random.randint(10, 100), self.dim) for j in range(self.number_points)}
        self.labels = {j: np.random.randint(self.number_classes, size=len(self.data[j])) for j in range(self.number_points)}

    def test_ModelS(self):
        model = ModelDictionary(LogisticRegression())

        n_split = 5
        split = int(self.number_points / n_split)
        predictions, _ = cross_validation(model, self.data, self.labels, {j: range(j*split, (j+1)*split) for j in range(n_split)})

        self.assertEqual(len(predictions), len(self.data))
        
        for d in self.data:
            self.assertEqual(len(self.labels[d]), len(predictions[d]))

        # Ignore label 2
        predictions, _ = cross_validation(model, self.data, self.labels, {j: range(j*split, (j+1)*split) for j in range(n_split)}, [0, 1])

        self.assertEqual(len(predictions), len(self.data))
        
        for d in self.data:
            self.assertEqual(len(self.labels[d]), len(predictions[d]))

        # With normalizer
        ind = IndividualNormalizationZScore(5)
        predictions, labels = cross_validation(model, self.data, self.labels, {j: range(j*split, (j+1)*split) for j in range(n_split)}, [0, 1], normalizer = ind)

        self.assertEqual(len(predictions), len(self.data))
        
        for d in self.data:
            self.assertEqual(len(labels[d]), len(predictions[d]))

if __name__ == '__main__':
    unittest.main()