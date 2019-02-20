import unittest
import numpy as np
import pandas as pd
from models.modelts import *
from sklearn.linear_model import LogisticRegression

class TestModelTS(unittest.TestCase):

    def setUp(self):
        self.number_points = 10
        self.number_classes = 3
        self.dim = 10
        self.data = {j: pd.DataFrame(np.random.rand(np.random.randint(10, 100), self.dim)) for j in range(self.number_points)}
        self.labels = {j: np.random.randint(self.number_classes, size=len(self.data[j])) for j in range(self.number_points)}

    def test_ModelDictionary(self):
        model = ModelDictionary(LogisticRegression())
       
        model.fit(self.data, self.labels)
        model.score(self.data, self.labels)
        predictions = model.predict(self.data)
        self.assertEqual(len(predictions), len(self.data))
        
        for d in self.data:
            self.assertEqual(len(self.labels[d]), len(predictions[d]))

if __name__ == '__main__':
    unittest.main()