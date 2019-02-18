import numpy as np
import matplotlib.pyplot as plt

from utils.ts_transformation import pushZeroTime

def evolutionPlot(predictions, truth, label = "Model", newFigure = None, clusters = None):
    """
        Computes the evolution of the mean evolution for each class
        With separation in clusters

        Arguments:
            predictions {Dict / List} -- Label predictions
            truth {Dict / List} -- Ground truth (one label per time series)
        
        Keyword Arguments:
            label {str} -- Legend to plot (default: {"Model"})
            newFigure {str} -- Display on a given figure (default: {None} - Create new figure)
            clusters {int / List of int} -- Number to clsuter to form or list of time bounds (default: {None} - No cluster)
    """
    if clusters is None:
        clusters = [0, max([len(p) for p in predictions])]
    elif type(clusters) is int:
        # Finds the different boundaries to divide the array in equal parts
        clusters = [0] + [a[-1] for a in np.split([len(p) for p in predictions], clusters)]
    assert len(clusters) > 1, "Bounds has to have more than one element"

    for i in range(len(clusters) - 1):
        selection = [pushZeroTime(p) for p in predictions if clusters[i] < len(predictions[p]) and len(predictions[p]) <= clusters[i+1]]

    global_fpr, global_tpr, _ = roc_curve(truth, predictions)
    if reverse:
        x, y = 1 - global_tpr, 1 - global_fpr # FNR, TNR
        x, y = x[::-1], y[::-1]
        minx = 1. / np.sum(truth == 1)
        print("TNR @{:.2f}% FNR : {:.2f}".format(minx*100, np.interp(minx, x, y)), end = ' ')
    else:
        x, y = global_fpr, global_tpr
        minx = 1. / np.sum(truth == 0)
        print("TPR @{:.2f}% FPR : {:.2f}".format(minx*100, np.interp(minx, x, y)), end = ' ')

            
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
    print("+/- {:.2f}".format(np.interp(0.01, newx, wilson)))
    upper = np.minimum(y + wilson, 1)
    lower = np.maximum(y - wilson, 0)
    plRoc = plt.plot(newx, y, label=label + " ({:.2f} +/- {:.2f})".format(auc(newx, y), (auc(newx, upper) - auc(newx, lower))/2.))
    plt.fill_between(newx, lower, upper, color=plRoc[0].get_color(), alpha=.2)