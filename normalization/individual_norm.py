"""
    Normalize data of one time series through time
    Use a given period of time preceding t and normalize the data
"""
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