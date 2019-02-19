import pandas as pd

class Normalization:

    # Abstract
    def fit(self, tsDict):
        return self

    # Abstract
    def transform_single(self, ts):
        raise Exception("Not implemented")

    def transform(self, tsDict):
        if isinstance(tsDict, dict):
            return {d: self.transform(tsDict[d]) for d in tsDict}
        elif isinstance(tsDict, pd.DataFrame):
            return self.transform_single(tsDict)
        else:
            return self.transform_single(pd.DataFrame(tsDict))

    def fit_transform(self, tsDict):
        self.fit(tsDict)
        return self.transform(tsDict)