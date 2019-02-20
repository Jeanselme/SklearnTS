import pandas as pd

class Transformation:
    """
        Object transforming data
    """
    def __init__(self, encapsulation = None):
        """
            Allow to encapsulate a function
        """
        self.encapsulation = encapsulation

    def fit(self, ts):
        if self.encapsulation:
            self.encapsulation.fit(ts)
        return self

    def transform(self, ts):
        if self.encapsulation:
            return pd.DataFrame(self.encapsulation.transform(ts), index = ts.index)
        raise Exception("Not implemented")

    def fit_transform(self, ts):
        if self.encapsulation:
            return pd.DataFrame(self.encapsulation.fit_transform(ts), index = ts.index)
        else:
            self.fit(ts)
            return self.transform(ts)

    def fit_dict(self, tsDict):
        return self.fit(pd.concat([tsDict[ts] for ts in tsDict]))

    def transform_dict(self, tsDict):
        if isinstance(tsDict, dict):
            return {d: self.transform_dict(tsDict[d]) for d in tsDict}
        elif isinstance(tsDict, pd.DataFrame):
            return self.transform(tsDict)
        else:
            return self.transform(pd.DataFrame(tsDict))

    def fit_transform_dict(self, tsDict):
        self.fit_dict(tsDict)
        return self.transform_dict(tsDict)

class Accumulator(Transformation):
    """
        Object allowing to use several transforamtion on the data
    """

    def __init__(self, transformationList):
        """
            Arguments:
                transformationList {List of transformation} -- Order is crucial
        """

        self.transformationList = transformationList

    def fit(self, ts):
        self.fit_transform(ts)
        return self

    def transform(self, ts):
        for transformation in self.transformationList:
            ts = transformation.transform(ts)
        return ts

    def fit_transform(self, ts):
        for transformation in self.transformationList:
            ts = transformation.fit_transform(ts)
        return ts

def pushZeroTime(ts, deep = False, invert = True):
    """
    Make the time series start or end at zero

    Arguments:
        ts {Dataframe / Time Series} -- The time series to invert
        invert {bool} --  Invert the time axis, ie the last point of the time series become zero
                        And all points before are positive times
                        Otherwise start at 0

    Returns:
        The inverted time series
    """
    res = ts.copy(deep)
    if invert:
        res.index =  max(ts.index) - ts.index
    else:
        res.index =  ts.index - min(ts.index) 
    return res

def computeMeanStdCount(tsList):
    """
    Computes the mean and std of a list of time series 
    (potentially of different size)
    
    Arguments:
        tsList {[type]} -- Dataframe or Series with time index

    Returns:
        Dataframes with 
        index           -- Time 
        mean            -- Mean of the ts
        std             -- Std of the ts
        number_points   -- Number points which allow to compute those values
    """
    result = pd.concat(tsList, axis=1)
    result["mean"] = result.mean(axis = "columns")
    result["std"] = result.std(axis = "columns")
    result["count"] = result.count(axis = "columns")
    return result