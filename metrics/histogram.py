import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from utils.utils import flatten

def histPlot(predictions, truth, label = "Model", newFigure = None, splitPosNeg = False, kde = False):
    """
        Computes the histograms of a binary predictions
        
        Arguments:
            predictions {Dict / List} -- Label predictions
            truth {Dict / List} -- Ground truth
        
        Keyword Arguments:
            label {str} -- Legend to plot (default: {"Model"})
            newFigure {str} -- Display on a given figure (default: {None} - Create new figure)
            splitPosNeg {bool} -- Split between positive and negative (default: {False})
            kde {bool} -- Computes the kde of the histogram (default: {False})
    """

    predictions, truth = flatten(predictions, truth)
    bins = np.linspace(0, 1, 20)

    if newFigure is not None:
        plt.figure(newFigure)
    else:
        plt.xlabel('Predicted Probability')
        plt.ylabel('Frequency')
        plt.title('Histogram Probabilities')

    if splitPosNeg:
        sns.distplot(predictions[truth == 1], label=label + " Positive", kde = kde, bins = bins)
        sns.distplot(predictions[truth == 0], label=label + " Negative", kde = kde, bins = bins)
    else:
        sns.distplot(predictions, label=label, kde = kde, bins = bins)