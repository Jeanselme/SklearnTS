import pandas as pd

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
        res.index =  ts.index.max() - ts.index
    else:
        res.index =  ts.index - ts.index.min() 
    return res.sort_index()

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