import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import brier_score_loss
from sklearn.calibration import calibration_curve

from utils.utils import flatten, selection

def calibrationPlot(predictions, truth, classes = None, label = "Model", newFigure = None, n_bins = 5):
    """
        Computes the roc with confidence bounds for the given model
        
        Arguments:
            predictions {Dict / List} -- Label predictions
            truth {Dict / List} -- Ground truth
            classes {Dict "+":int, "-":int} -- Classes to consider to plot {Default None ie {+":1, "-":0}}
        
        Keyword Arguments:
            label {str} -- Legend to plot (default: {"Model"})
            newFigure {str} -- Display on a given figure (default: {None} - Create new figure)
            n_bins {int} -- Numbre of bins for the calibration (default: {5})
    """
    predictions, truth = selection(predictions, truth, classes)
    predictions, truth = flatten(predictions, truth)
    predictions = ((predictions - predictions.min()) / (predictions.max() - predictions.min()))
    fraction_of_positives, mean_predicted_value = calibration_curve(truth, predictions, n_bins = n_bins)
    bins = np.linspace(0., 1. + 1e-8, n_bins + 1)
    binids = np.digitize(predictions, bins) - 1
    bin_sums = np.bincount(binids, minlength=len(bins))
    bin_sums = bin_sums[bin_sums != 0] * 500 / np.sum(bin_sums)

    if newFigure is not None:
        plt.figure(newFigure)
    else:
        plt.xlabel('Mean Predicted Value')
        plt.ylabel('Fraction Positive')
        plt.title('Calibration')

    p = plt.plot(mean_predicted_value, fraction_of_positives, alpha = 0.5, ls=':')
    plt.scatter(mean_predicted_value, fraction_of_positives, s = bin_sums, label = label + " ({:.2f})".format(brier_score_loss(truth, predictions)), color = p[0].get_color(), alpha = 0.5)