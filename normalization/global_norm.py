"""
    Normalize one time series given all the other
"""
from utils.encapsulator import Transformation

class GlobalNormalizationZScore(Transformation):

    def fit(self, ts, tsLabels = None):
        self.mean = ts.mean(axis = "rows")
        self.std = ts.std(axis = "rows")
        self.std[self.std == 0] = 1 
        return self

    def transform(self, ts):
        return ((ts - self.mean) / self.std).dropna()

class GlobalNormalizationZScoreRobust(GlobalNormalizationZScore):

    def fit(self, ts, tsLabels = None):
        self.mean = ts.median(axis = "rows")
        medianDeviation = (ts - self.mean).abs()
        self.std = medianDeviation.median(axis = "rows")
        self.std[self.std == 0] = 1 
        return self