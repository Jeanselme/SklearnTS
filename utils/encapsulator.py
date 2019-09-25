import pandas as pd

class Encapsulator:

    def __init__(self, encapsulation = None):
        """
            Allow to encapsulate a function
        """
        self.encapsulation = encapsulation

class Model(Encapsulator):

    def fit(self, ts, tsLabels, **params):
        if self.encapsulation is not None:
            self.encapsulation.fit(ts, tsLabels, **params)
        return self

    def predict_proba(self, ts):
        if self.encapsulation:
            return pd.DataFrame(self.encapsulation.predict_proba(ts), index = ts.index)
        raise Exception("Not implemented")

    def predict(self, ts):
        if self.encapsulation:
            return pd.DataFrame(self.encapsulation.predict(ts), index = ts.index)
        raise Exception("Not implemented")

    def fit_dict(self, tsDict, tsLabelsDict, **params):
        keys = list(tsDict.keys())
        return self.fit(pd.concat([tsDict[ts] for ts in keys]), pd.concat([tsLabelsDict[ts] for ts in keys]), **params)

    def predict_dict(self, tsDict):
        if isinstance(tsDict, dict):
            return {d: self.predict_dict(tsDict[d]) for d in tsDict}
        elif isinstance(tsDict, pd.DataFrame):
            return self.predict(tsDict)
        else:
            return self.predict(pd.DataFrame(tsDict))
    
    def predict_proba_dict(self, tsDict):
        if isinstance(tsDict, dict):
            return {d: self.predict_proba_dict(tsDict[d]) for d in tsDict}
        elif isinstance(tsDict, pd.DataFrame):
            return self.predict_proba(tsDict)
        else:
            return self.predict_proba(pd.DataFrame(tsDict))

class Transformation(Encapsulator):
    """
        Object in order to encapsulate Sklearn model and transformation
    """

    def fit(self, ts, tsLabels = None):
        if self.encapsulation:
            self.encapsulation.fit(ts, tsLabels)
        return self

    def transform(self, ts):
        if self.encapsulation:
            transformed = self.encapsulation.transform(ts.copy())
            return pd.DataFrame(transformed, index = ts.index[-len(transformed):])
        raise Exception("Not implemented")

    def fit_transform(self, ts, tsLabels = None):
        if self.encapsulation:
            transformed = self.encapsulation.fit_transform(ts, tsLabels)
            return pd.DataFrame(transformed, index = ts.index[-len(transformed):])
        else:
            self.fit(ts, tsLabels)
            return self.transform(ts)

    def fit_dict(self, tsDict, tsLabelsDict = None):
        return self.fit(pd.concat([tsDict[ts] for ts in tsDict]), 
            pd.concat([tsLabelsDict[ts] for ts in tsDict]) if tsLabelsDict else None)

    def transform_dict(self, tsDict):
        if isinstance(tsDict, dict):
            return {d: self.transform_dict(tsDict[d]) for d in tsDict}
        elif isinstance(tsDict, pd.DataFrame):
            return self.transform(tsDict)
        else:
            return self.transform(pd.DataFrame(tsDict))

    def fit_transform_dict(self, tsDict, tsLabelsDict = None):
        self.fit_dict(tsDict, tsLabelsDict)
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

    def fit(self, ts, tsLabels = None):
        self.fit_transform(ts, tsLabels)
        return self

    def transform(self, ts):
        for transformation in self.transformationList:
            ts = transformation.transform(ts)
        return ts

    def fit_transform(self, ts, tsLabels = None):
        for transformation in self.transformationList:
            ts = transformation.fit_transform(ts, tsLabels)
        return ts
