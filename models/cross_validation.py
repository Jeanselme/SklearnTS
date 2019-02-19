from utils.utils import flatten, selection

def cross_validation(model, data, labels, folds, classes = None, normalizer = None):
    """
        Computes the cross valdiation on the data
        Given the folds indicated in folds
        
        Arguments:
            model {model} -- Model to cross validate
            data {Dict} -- Data to split 
            labels {Dict} -- Labels of the data (keys have to match)
            folds {Dict: (key : fold, values: key of data and labels)} -- Folds in order to split the data

        Returns:
            Predictions by the model on cross validated data (Dict with same keys than data)
    """
    predictions, labels_res = {}, labels.copy()
    for k in folds:
        data_fold, labels_fold = selection({d: data[d] for d in data if d not in folds[k]}, {d: labels[d] for d in data if d not in folds[k]}, classes)
        data_test = {d: data[d] for d in folds[k]}
        
        if normalizer is not None:
            data_fold = normalizer.fit_transform(data_fold)
            data_test = normalizer.transform(data_test)

            # Because Normalization can impact labeling
            labels_fold = {d: labels[d][data_fold[d].index] for d in labels_fold} 
            labels_res.update({d: labels[d][data_test[d].index] for d in folds[k]})

        data_fold, labels_fold = flatten(data_fold, labels_fold)
        model.fit(data_fold, labels_fold)
        predictions.update(model.predict(data_test))
    return predictions, labels_res
