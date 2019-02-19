import numpy as np

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
        return data, labels

    if isinstance(classes, dict):
        classes = [classes[c] for c in classes]

    if isinstance(data, dict):
        check_size(data, labels)
        data = {d: data[d][np.isin(labels[d], classes)] for d in data if np.any(np.isin(labels[d], classes))}
        labels = {d: labels[d][np.isin(labels[d], classes)] for d in labels if np.any(np.isin(labels[d], classes))}
    else:
        data = data[np.isin(labels, classes)]
        labels = labels[np.isin(labels, classes)]

    return data, labels

def discretization(data, threshold):
    if isinstance(data, dict):
        data = {d: discretization(data[d], threshold) for d in data}
    else:
        data = data > threshold
    return data