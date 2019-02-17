def pushZeroTime(ts, deep = False):
    """
    Invert the time axis, ie the last point of the time series become zero
    And all points before are positive times
    
    Arguments:
        ts {Dataframe / Time Series} -- The time series to invert

    Returns:
        The inverted time series
    """
    res = ts.copy(deep)
    res.index =  max(ts.index) - ts.index
    return res