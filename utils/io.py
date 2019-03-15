import os
import pandas as pd
from joblib import Parallel, delayed

def readParallel(directory, processor, files = None, **args):
    """
        Reads files located in the given directory in parallel
        
        Arguments:
            directory {str} -- Path where to read the data
            processor {int} -- Number of processor to use (see joblib Parallel for more info)
            files {List of str} -- Names of the files to read (default {None} - Open all files in directory)
            **args -- Args to give to pandas.read_csv

        Returns:
            Dict: Keys are file names, Values are pandas objects
    """
    if files is None:
        files = [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f)) and '.csv' in f]
    res = Parallel(n_jobs = processor)(delayed(pd.read_csv)(os.path.join(directory, f), **args) for f in files)
    return {f[:f.index('.csv')]: res[i].sort_index() for i, f in enumerate(files)}

def writeParallel(timeseries, directory, processor, **args):
    """
        Saves the different time series in the given directory
        
        Arguments:
            timeseries {Dict of pandas dataframe} -- Data to save (keys will be the name of the file to save)
            directory {str} -- Path where to save the data
            processor {int} -- Number of processor to use (see joblib Parallel for more info)
    """
    def write(df, path, **args):
        df.to_csv(path, **args)
    Parallel(n_jobs = processor)(delayed(write)(timeseries[ts], os.path.join(directory, ts if '.csv' in ts else ts + '.csv'), **args) for ts in timeseries)