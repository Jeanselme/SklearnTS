import numpy as np
import pandas as pd

def check_size(data, labels):
    """
    Verifies the size of the given data
    
    Arguments:
        data {Dict of times series} -- Data to use
        labels {Dict of labels} -- Labels to use
    """
    if isinstance(data, dict):
        for d in data:
            assert d in labels, "Datapoint {} has no associated labels".format(d)
            assert len(np.array(data[d]).shape) <= 2
            assert len(data[d]) == len(labels[d]), "Inconstitancy in the shape of data and label for {} expected {} labels but received {}".format(d, len(data[d]), len(labels[d]))
            assert len(np.array(labels[d]).flatten()) == len(labels[d]), "Expect one dimension labels"

def flatten(data, labels):
    """
    Transforms a dictionary into an array
    
    Arguments:
        data {Dict/List of times series} -- Time series
        labels {Dict/List labels} -- Labels associated
    """
    check_size(data, labels)
    if isinstance(data, dict):
        patients = list(data.keys())
        data = np.concatenate([np.array(data[p]).reshape((len(data[p]), -1)) for p in patients])
        labels = np.concatenate([np.array(labels[p]) for p in patients])
    if isinstance(data, pd.Series):
        data = data.values
    if isinstance(labels, pd.Series):
        labels = labels.values
    return data, labels

def selection(data, labels, classes = None):
    """
    Selects from data the points that are in classes
    
    Arguments:
        data {Dict/List of times series} -- Time series
        labels {Dict/List labels} -- Labels associated
        classes {Dict/List labels} -- Classes to select
    """
    if classes is None:
        if isinstance(data, dict):
            data_res, labels_res = {}, {}
            for d in data:
                data_res[d], labels_res[d] = selection(data[d], labels[d])
        elif isinstance(data, pd.DataFrame) or isinstance(data, pd.Series):
            index = data.index.intersection(labels.index)
            data_res, labels_res = data.loc[index].copy(), labels.loc[index].copy()
        else:
            data_res, labels_res = data.copy(), labels.copy()
        return data_res, labels_res

    if isinstance(classes, dict):
        classes = [classes[c] for c in classes]

    if isinstance(data, dict):
        check_size(data, labels)
        data_res = {d: data[d][np.isin(labels[d], classes)] for d in data if np.any(np.isin(labels[d], classes))}
        labels_res = {d: labels[d][np.isin(labels[d], classes)] for d in labels if np.any(np.isin(labels[d], classes))}
    else:
        data_res = data[np.isin(labels, classes)]
        labels_res = labels[np.isin(labels, classes)]

    return data_res, labels_res

def extractLabels(data, label_column):
    """
        Extracts the labels from the data
        
        Arguments:
            data {Dict/List of times series} -- Time series
            label_column {str} -- Name of the column to extract
    """
    if isinstance(data, dict):
        labels = {d: data[d][label_column] for d in data}
        data = {d: data[d][[c for c in data[d].columns if c != label_column]] for d in data}
    else:
        labels = data[label_column]
        data = data[[c for c in data.columns if c != label_column]]

    return data, labels

def discretization(data, threshold):
    if isinstance(data, dict):
        data = {d: discretization(data[d], threshold) for d in data}
    else:
        data = data > threshold
    return data