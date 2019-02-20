"""
    Normalize one time series given all the other
"""
from utils.ts_transformation import Transformation

class GlobalNormalizationZScore(Transformation):

    def fit(self, ts):
        self.mean = ts.mean(axis = "rows")
        self.std = ts.std(axis = "rows")
        self.std[self.std == 0] = 1 
        return self

    def transform(self, ts):
        return ((ts - self.mean) / self.std).dropna()