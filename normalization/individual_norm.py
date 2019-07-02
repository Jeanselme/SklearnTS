"""
    Normalize data of one time series through time
    Use a given period of time preceding t and normalize the data
"""
import numpy as np
from utils.encapsulator import Transformation

class IndividualNormalizationZScore(Transformation):

    def __init__(self, window):
        Transformation.__init__(self)
        self.window = window

    def transform(self, ts):
        roll = ts.rolling(self.window, min_periods = 1)
        std = roll.std()
        std[std == 0] = 1 
        return ((ts - roll.mean()) / std).dropna()

class IndividualNormalizationZScoreRobust(IndividualNormalizationZScore):

    def transform(self, ts):
        roll = ts.rolling(self.window, min_periods = 1)
        median = roll.median()
        medianDeviation = (ts - median).abs().rolling(self.window, min_periods = 1)
        medianDeviation = medianDeviation.median()

        # Avoid division by 0 and change median
        median[median.index.difference(medianDeviation.index)] = np.nan

        return ((ts - median) / medianDeviation).dropna()
