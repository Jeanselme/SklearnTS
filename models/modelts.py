import pandas as pd
from utils.utils import flatten

class ModelDictionary():
    """
        Encapsulation of sklearn model
        This model is useful you have a dictionary of different
        sets of datapoints (different sizes allowed -> TS)
    """

    def __init__(self, model):
        """
            Initialize the model computation
            
            Arguments:
                model {Sklearn Model fit / score} -- Model to use
        """
        self.model = model

    def fit(self, trainingData, trainingLabels):
        """
        Fit the given model on the data
        
        Arguments:
            trainingData {Dict/List Features} -- Training time series
            trainingLabels {Dict/List Boolean} -- Labels associated
        """
        flatData, flatLabel = flatten(trainingData, trainingLabels)
        self.model = self.model.fit(flatData, flatLabel)
        return self

    def score(self, data, labels):
        flatData, flatLabel = flatten(data, labels)
        return self.model.score(flatData, flatLabel)

    def predict(self, data):
        if isinstance(data, dict):
            return {d: self.predict(data[d]) for d in data}
        elif isinstance(data, pd.DataFrame):
            return pd.DataFrame(self.model.predict(data), index = data.index, columns = ["Predictions"])
        else:
            return self.model.predict(data)