import numpy as np

def check_size(data, labels):
    """
    Verifies the size of the given data
    
    Arguments:
        data {Dict of times series} -- Data to use
        labels {Dict of labels} -- Labels to use
    """
    if isinstance(data, dict):
        dim = None
        for d in data:
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
        labels = np.concatenate([np.array(labels[p]).reshape((len(labels[p]), 1)) for p in patients])
    return data, labels