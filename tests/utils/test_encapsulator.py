import unittest
import numpy as np
import pandas as pd
from utils.encapsulator import *
from normalization.global_norm import GlobalNormalizationZScore
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression

class TestEncapsulator(unittest.TestCase):

    def setUp(self):
        self.number_points = 10
        self.number_classes = 3
        self.dim = 10
        self.data = {j: pd.DataFrame(np.random.rand(np.random.randint(10, 100), self.dim)) for j in range(self.number_points)}
        self.labels = {j: pd.Series(np.random.randint(self.number_classes, size=len(self.data[j]))) for j in range(self.number_points)}

    def test_Model(self):
        model = Model(LogisticRegression())
       
        model.fit_dict(self.data, self.labels)
        predictions = model.predict_dict(self.data)
        self.assertEqual(len(predictions), len(self.data))

        predictions_proba = model.predict_proba_dict(self.data)
        self.assertEqual(len(predictions), len(self.data))
        
        for d in self.data:
            self.assertEqual(len(self.labels[d]), len(predictions[d]))
            self.assertEqual(len(self.labels[d]), predictions_proba[d].shape[0])
            self.assertEqual(self.number_classes, predictions_proba[d].shape[1])

    def test_Transformation(self):
        pca = Transformation(PCA(2))
        pca_res = pca.fit_transform_dict(self.data)
        self.assertEqual(len(pca_res), len(self.data))
        self.assertEqual(pca_res[0].shape[1], 2)

    def test_Accumulator(self):
        acc = Accumulator([Transformation(PCA(2)), GlobalNormalizationZScore()])
        acc_res = acc.fit_transform_dict(self.data)
        self.assertEqual(len(acc_res), len(self.data))
        self.assertEqual(acc_res[0].shape[1], 2)

if __name__ == '__main__':
    unittest.main()