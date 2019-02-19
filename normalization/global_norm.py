"""
    Normalize one time series given all the other
"""
import pandas as pd
from normalization.norm import Normalization

class GlobalNormalizationZScore(Normalization):

    def fit(self, tsDict):
        data = pd.concat([tsDict[ts] for ts in tsDict])
        self.mean = data.mean(axis = "rows")
        self.std = data.std(axis = "rows")
        self.std[self.std == 0] = 1 
        return self

    def transform_single(self, ts):
        return ((ts - self.mean) / self.std).dropna()