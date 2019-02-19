import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

from utils.utils import flatten, selection

def confusionPlot(predictions, truth, classes, percentage = True):
    """
        Computes the confusion matrix of the given model
        
        Arguments:
            predictions {Dict / List} -- Label predictions
            truth {Dict / List} -- Ground truth
            classes {Dict "+":int, "-":int} -- Classes to consider to plot
    """
    predictions, truth = selection(predictions, truth, classes)
    predictions, truth = flatten(predictions, truth)

    classes_list = np.array(list(classes.keys()))
    confusion = confusion_matrix(truth, predictions, labels=[classes[c] for c in classes_list])
    notNull = confusion.sum(axis = 0) != 0

    if percentage:
        confusion = confusion / confusion.sum(axis = 1, keepdims = True)

    sns.heatmap(confusion[:, notNull], xticklabels = classes_list[notNull], yticklabels = classes_list, annot = True, vmin = 0, vmax = 1 if percentage else None)
    plt.xlabel("Predicted")
    plt.ylabel("Ground truth")