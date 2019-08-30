import os
import pandas as pd
from joblib import Parallel, delayed

def readParallel(directory, processor, files = None, read_function = pd.read_csv, **args):
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
    res = Parallel(n_jobs = processor)(delayed(read_function)(os.path.join(directory, f), **args) for f in files)
    return {f[:f.rindex('.')]: res[i].sort_index() for i, f in enumerate(files)}

def extractParallel(directory, feature_function, processor, files = None, features_args = {}, read_function = pd.read_csv, reading_args = {}):
    """
        Reads files located in the given directory 
        And applies the features extraction in parallel

        Arguments:
            directory {str} -- Path where to read the data
            feature_function {function} -- Function to apply on the data
            processor {int} -- Number of processor to use (see joblib Parallel for more info)
            files {List of str} -- Names of the files to read (default {None} - Open all files in directory)
            features_args -- Args to give to feature_function
            reading_args -- Args to give to pandas.read_csv

        Returns:
            Dict: Keys are file names, Values are pandas objects
    """
    if files is None:
        files = [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f)) and '.csv' in f]
    res = Parallel(n_jobs = processor)(delayed(lambda path: feature_function(read_function(path, **reading_args), **features_args))(os.path.join(directory, f)) for f in files)
    return {f[:f.rindex('.')]: res[i].sort_index() for i, f in enumerate(files)}

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