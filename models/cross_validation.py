import numpy as np
import pandas as pd
from utils.utils import flatten, selection

def cross_validation(model, data, labels, folds, classes = None, transform = None, proba = True, nested_cv = False):
    """
        Computes the cross valdiation on the data
        Given the folds indicated in folds
        
        Arguments:
            model {model} -- Model to cross validate
            data {Dict} -- Data to split 
            labels {Dict} -- Labels of the data (keys have to match)
            folds {Dict: (key : fold, values: key of data and labels)} -- Folds in order to split the data
            transform {Transform Object} -- Compute the transformation on train and apply on train and test

        Returns:
            Predictions by the model on cross validated data (Dict with same keys than data)
    """
    predictions, labels_res = {}, labels.copy()
    for k in folds:
        data_fold, labels_fold = selection({d: data[d] for d in data if d not in folds[k]}, {d: labels[d] for d in data if d not in folds[k]}, classes)
        data_test = {d: data[d] for d in folds[k]}
        
        if transform is not None:
            data_fold = transform.fit_transform_dict(data_fold, labels_fold)
            data_test = transform.transform_dict(data_test)

            # Because Normalization can impact labeling
            data_fold = {d: data_fold[d] for d in data_fold if len(data_fold[d]) > 0} 
            labels_fold = {d: labels[d][data_fold[d].index] for d in data_fold if len(data_fold[d]) > 0} 

            data_test = {d: data_test[d] for d in data_test if len(data_test[d]) > 0}
            labels_res.update({d: labels[d][data_test[d].index] for d in folds[k] if len(data_test[d]) > 0})

        if nested_cv:
            groups = [i for i in data_fold for _ in range(len(data_fold[i]))]
            model.fit_dict(data_fold, labels_fold, groups = groups)
        else:
            model.fit_dict(data_fold, labels_fold)

        if proba:
            predictions.update(model.predict_proba_dict(data_test))
        else:
            predictions.update(model.predict_dict(data_test))

    return predictions, labels_res