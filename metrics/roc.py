import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import auc, roc_curve

from utils.utils import flatten, selection

def rocPlot(predictions, truth, classes = None, label = "Model", newFigure = None, reverse = False):
    """
        Computes the roc with confidence bounds for the given model
        
        Arguments:
            predictions {Dict / List} -- Label predictions
            truth {Dict / List} -- Ground truth
            classes {Dict "+":int, "-":int} -- Classes to consider to plot {Default None ie {+":1, "-":0}}
        
        Keyword Arguments:
            label {str} -- Legend to plot (default: {"Model"})
            newFigure {str} -- Display on a given figure (default: {None} - Create new figure)
            reverse {bool} -- Plot the reverse ROC useful for analyzing TNR (default: {False})
    """
    predictions, truth = selection(predictions, truth, classes)
    predictions, truth = flatten(predictions, truth)
    global_fpr, global_tpr, _ = roc_curve(truth, predictions)
    if reverse:
        x, y = 1 - global_tpr, 1 - global_fpr # FNR, TNR
        x, y = x[::-1], y[::-1]
        minx = 1. / np.sum(truth == 1)
        str_print = "TNR @{:.2f}% FNR : {:.2f}".format(minx*100, np.interp(minx, x, y))
    else:
        x, y = global_fpr, global_tpr
        minx = 1. / np.sum(truth == 0)
        str_print = "TPR @{:.2f}% FPR : {:.2f}".format(minx*100, np.interp(minx, x, y))

            
    if newFigure is not None:
        plt.figure(newFigure)
    else:
        plt.plot(np.linspace(0, 1, 100), np.linspace(0, 1, 100), 'k--', label="Random")
        if reverse:
            plt.xlabel('False negative rate')
            plt.ylabel('True negative rate')
            plt.title('Reverse ROC curve')
        else:
            plt.xlabel('False positive rate')
            plt.ylabel('True positive rate')
            plt.title('ROC curve')
            
    newx = np.linspace(minx, 1, 1000)
    y = np.interp(newx, x, y)
    wilson = 1.96 * np.sqrt(y * (1 - y)/len(predictions))
    print(str_print + " +/- {:.2f}".format(np.interp(0.01, newx, wilson)))
    upper = np.minimum(y + wilson, 1)
    lower = np.maximum(y - wilson, 0)
    plRoc = plt.plot(newx, y, label=label + " ({:.2f} +/- {:.2f})".format(auc(newx, y), (auc(newx, upper) - auc(newx, lower))/2.))
    plt.fill_between(newx, lower, upper, color=plRoc[0].get_color(), alpha=.2)