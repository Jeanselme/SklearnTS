import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import auc, precision_recall_curve, average_precision_score

from utils.utils import flatten, selection

def averagePrecisionRecallCompute(predictions, truth, classes = None):
    """
        Computes AUC of the given predictions
        
        Arguments:
            predictions {Dict / List} -- Label predictions
            truth {Dict / List} -- Ground truth
        
        Keyword Arguments:
            classes {Dict "+":int, "-":int} -- Classes to consider to plot {Default None ie {+":1, "-":0}}
    
        Returns:
            float -- Estimation by pooling of auc
    """
    predictions, truth = selection(predictions, truth, classes)
    predictions, truth = flatten(predictions, truth)
    return average_precision_score(truth, predictions)

def precisionRecallPlot(predictions, truth, classes = None, label = "Model", newFigure = None, reverse = False, percentage = None):
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
    precision, recall, _ = precision_recall_curve(truth, predictions)
            
    if newFigure is not None:
        plt.figure(newFigure)
    else:
        plt.xlabel('Precision')
        plt.ylabel('Recall')
        plt.title('Precision Recall curve')

    plt.plot(precision, recall, label=label + " ({:.2f})".format(averagePrecisionRecallCompute(predictions, truth, classes)), ls = '--' if "train" in label.lower() else '-')

def computeEvolutionAPR(temporalListLabels, predictions, classes = None, percentage = 0.001):
    """
        Compute the evolution of the average Precision Recall 
        
        Arguments:
            temporalListLabels {List of (time, labels)*} -- Ground truth labels
            predictions {Dict / List of labels} -- Predicitons (same format than labels in temporalListLabels)
            classes {Dict} -- Classes to consider to plot (key: Name to display, Value: label)
            percentage {float} -- Evaluate the TPR and TNR at this given value of FNR and FPR
    """
    apr = {time: averagePrecisionRecallCompute(predictions, labels, classes) for time, labels in temporalListLabels          }
    return pd.DataFrame.from_dict(apr, orient = "index")

def aprEvolutionPlot(temporalListLabels, predictions, classes = None, label = "Model", newFigure = None):
    """
        Plots the evolution of the auc 
        
        Arguments:
            temporalListLabels {List of (time, labels)*} -- Ground truth labels
            predictions {Dict / List of labels} -- Predicitons (same format than labels in temporalListLabels)
            classes {Dict} -- Classes to consider to plot (key: Name to display, Value: label)
    """
    apr = computeEvolutionAPR(temporalListLabels, predictions, classes)

    if newFigure is not None:
        plt.figure(newFigure)
    else:
        plt.figure("Evolution AUC")
        plt.xlabel('Time before event (in minutes)')
        plt.ylabel('AUC')

    plt.plot(apr.index.total_seconds() / 60., apr.values, label = label, ls = '--' if "train" in label.lower() else '-')

