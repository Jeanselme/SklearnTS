import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import auc, roc_curve, roc_auc_score

from utils.utils import flatten, selection

def aucCompute(predictions, truth, classes = None):
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
    return roc_auc_score(truth, predictions)

def rocPlot(predictions, truth, classes = None, label = "Model", newFigure = None, reverse = False, percentage = None):
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
        if percentage is None:
            percentage = minx
        str_print = "TNR @{:.2f}% FNR : {:.2f}".format(percentage*100, np.interp(percentage, x, y))
    else:
        x, y = global_fpr, global_tpr
        minx = 1. / np.sum(truth == 0)
        if percentage is None:
            percentage = minx
        str_print = "TPR @{:.2f}% FPR : {:.2f}".format(percentage*100, np.interp(percentage, x, y))

            
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
    plRoc = plt.plot(newx, y, label=label + " ({:.2f} +/- {:.2f})".format(aucCompute(predictions, truth, classes), (auc(newx, upper) - auc(newx, lower))/2.), ls = '--' if "train" in label.lower() else '-')
    plt.fill_between(newx, lower, upper, color=plRoc[0].get_color(), alpha=.2)

def computeEvolutionRoc(temporalListLabels, predictions, classes = None, percentage = 0.001):
    """
        Plots the evolution of the auc 
        
        Arguments:
            temporalListLabels {List of (time, labels)*} -- Ground truth labels
            predictions {Dict / List of labels} -- Predicitons (same format than labels in temporalListLabels)
            classes {Dict} -- Classes to consider to plot (key: Name to display, Value: label)
            percentage {float} -- Evaluate the TPR and TNR at this given value of FNR and FPR
    """
    aucs = {}
    for time, labels in temporalListLabels:
        pred_time, labels_time = selection(predictions, labels, classes)
        pred_time, labels_time = flatten(pred_time, labels_time)
        fpr, tpr, _ = roc_curve(labels_time, pred_time)
        fnr, tnr = (1 - tpr)[::-1], (1 - fpr)[::-1]
        auc_time = auc(fpr, tpr)
        wilson_tpr = 1.96 * np.sqrt(tpr * (1 - tpr)/len(predictions))
        wilson_tnr = 1.96 * np.sqrt(tnr * (1 - tnr)/len(predictions))

        aucs[time] = {
                        "auc": auc_time, 
                        "lower": auc(fpr, tpr - wilson_tpr), 
                        "upper": auc(fpr, tpr + wilson_tpr), 

                        "tpr": np.interp(percentage, fpr, tpr),
                        "tpr_wilson" : np.interp(percentage, fpr, wilson_tpr),

                        "tnr": np.interp(percentage, fnr, tnr),
                        "tnr_wilson" : np.interp(percentage, fnr, wilson_tnr),
                     }
                     
    return pd.DataFrame.from_dict(aucs, orient = "index")

def rocEvolutionPlot(temporalListLabels, predictions, classes = None, label = "Model", newFigure = None):
    """
        Plots the evolution of the auc 
        
        Arguments:
            temporalListLabels {List of (time, labels)*} -- Ground truth labels
            predictions {Dict / List of labels} -- Predicitons (same format than labels in temporalListLabels)
            classes {Dict} -- Classes to consider to plot (key: Name to display, Value: label)
    """
    aucs = computeEvolutionRoc(temporalListLabels, predictions, classes)

    if newFigure is not None:
        plt.figure(newFigure)
    else:
        plt.figure("Evolution AUC")
        plt.xlabel('Time before event (in minutes)')
        plt.ylabel('AUC')

    plAuc = plt.plot(aucs.index.total_seconds() / 60., aucs["auc"].values, label = label, ls = '--' if "train" in label.lower() else '-')
    plt.fill_between(aucs.index.total_seconds() / 60., aucs["lower"], aucs["upper"], color=plAuc[0].get_color(), alpha=.2)

