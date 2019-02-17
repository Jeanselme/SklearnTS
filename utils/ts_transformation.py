def pushZeroTime(ts, deep = False):
    """
    Invert the time axis, ie the last point of the time series become zero
    And all points before are negative times
    
    Arguments:
        ts {Dataframe / Time Series} -- The time series to invert

    Returns:
        The inverted time series
    """
    res = ts.copy(deep)
    res.index = ts.index - max(ts.index)
    return res