import numpy as np
import matplotlib.pyplot as plt

from metrics.histogram import histPlot
from metrics.calibration import calibrationPlot
from metrics.roc import rocPlot, aucEvolutionPlot

def rocCompare(listModels, truth, classes = None):
    """
        Plots the different roc for different models
        
        Arguments:
            listModels {List of (name, predictions)*} -- Models to display
            truth {Dict / List of true labels} -- Ground truth
            classes {Dict "+":int, "-":int} -- Classes to consider to plot {Default None ie {+":1, "-":0}}
    """
    for reverse in [False, True]:
        for log in [False, True]:
            plt.figure("Roc")
            plt.plot(np.linspace(0, 1, 100), np.linspace(0, 1, 100), 'k--', label="Random")
            if reverse:
                plt.xlabel('False negative rate')
                plt.ylabel('True negative rate')
                plt.title('Reverse ROC curve')
            else:
                plt.xlabel('False positive rate')
                plt.ylabel('True positive rate')
                plt.title('ROC curve')
            for (name, predictions) in listModels:
                rocPlot(predictions, truth, classes, name, "Roc", reverse)
            plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15))
            if log:
                plt.xscale('log')
            plt.show()

def histCompare(listModels, truth, classes = None, splitPosNeg = False, kde = False):
    """
        Plots the different histogram of predictions

        Arguments:
            listModels {List of (name, predictions)*} -- Models to display
            truth {Dict / List of true labels} -- Ground truth
            classes {Dict "+":int, "-":int} -- Classes to consider to plot {Default None ie {+":1, "-":0}}
    """
    plt.figure("Histogram Probabilities")
    plt.xlabel('Predicted Probability')
    plt.ylabel('Frequency')
    plt.title('Histogram Probabilities')
    for (name, predictions) in listModels:
        histPlot(predictions, truth, classes, name, "Histogram Probabilities", splitPosNeg, kde)
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15))
    plt.show()

def calibrationCompare(listModels, truth, classes = None, n_bins = 5):
    """
        Plots the different histogram of predictions

        Arguments:
            listModels {List of (name, predictions)*} -- Models to display
            truth {Dict / List of true labels} -- Ground truth
            classes {Dict "+":int, "-":int} -- Classes to consider to plot {Default None ie {+":1, "-":0}}
    """
    plt.figure("Calibration")
    plt.xlabel('Mean Predicted Value')
    plt.ylabel('Fraction Positive')
    plt.title('Calibration')
    for (name, predictions) in listModels:
        calibrationPlot(predictions, truth, classes, name, "Calibration", n_bins)
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15))
    plt.show()

def aucEvolutionCompare(listModels, temporalListLabels, classes):
    """
        Plots the different histogram of predictions

        Arguments:
            listModels {List of (name, predictions)*} -- Models to display
            temporalListLabels {Dict {time: true labels}} -- Ground truth
            classes {Dict "+":int, "-":int} -- Classes to consider to plot {Default None ie {+":1, "-":0}}
    """ 
    plt.figure("Evolution AUC")
    plt.xlabel('Time before event (in minutes)')
    plt.ylabel('AUC')
    plt.title('Evolution AUC')
    plt.plot([min(temporalListLabels)[0].seconds / 60., max(temporalListLabels)[0].seconds / 60.], [0.5, 0.5], 'k--', label="Random Model")
    for (name, predictions) in listModels:
        aucEvolutionPlot(temporalListLabels, predictions,classes, name, "Evolution AUC")
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15))
    plt.show()