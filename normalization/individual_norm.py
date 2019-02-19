"""
    Normalize data of one time series through time
    Use a given period of time preceding t and normalize the data
"""
from normalization.norm import Normalization

class IndividualNormalizationZScore(Normalization):

    def __init__(self, window):
        self.window = window

    def transform_single(self, ts):
        roll = ts.rolling(self.window, min_periods = 1)
        std = roll.std()
        std[std == 0] = 1 
        return ((ts - roll.mean()) / std).dropna()