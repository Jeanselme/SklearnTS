import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from metrics.histogram import histPlot
from metrics.calibration import calibrationPlot
from metrics.roc import rocPlot, computeEvolutionRoc

def rocCompare(listModels, truth, classes = None, **arg_roc):
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
                rocPlot(predictions, truth, classes, name, "Roc", reverse, **arg_roc)
            plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15))
            if log:
                plt.xscale('log')
            plt.ylim(-0.1, 1.1)
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
    plt.plot(np.linspace(0, 1, 100), np.linspace(0, 1, 100), 'k--', label="Random")
    for (name, predictions) in listModels:
        calibrationPlot(predictions, truth, classes, name, "Calibration", n_bins)
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15))
    plt.show()

def rocEvolutionCompare(listModels, temporalListLabels, classes, percentage = 0.001):
    """
        Plots the different histogram of predictions

        Arguments:
            listModels {List of (name, predictions)*} -- Models to display
            temporalListLabels {Dict {time: true labels}} -- Ground truth
            classes {Dict "+":int, "-":int} -- Classes to consider to plot {Default None ie {+":1, "-":0}}
    """ 
    aucs = {}
    for (name, predictions) in listModels:
        aucs[name] = computeEvolutionRoc(temporalListLabels, predictions, classes, percentage)
    
    # AUC
    plt.figure("Evolution")
    plt.xlabel('Time before event (in minutes)')
    plt.ylabel('Evolution')
    plt.title('Evolution AUC')
    plt.plot([min(temporalListLabels)[0].total_seconds() / 60., max(temporalListLabels)[0].total_seconds() / 60.], [0.5, 0.5], 'k--', label="Random Model")
    for name in aucs:
        plAuc = plt.plot(aucs[name].index.total_seconds() / 60., aucs[name]["auc"].values, label = name, ls = '--' if "train" in name.lower() else '-')
        plt.fill_between(aucs[name].index.total_seconds() / 60., aucs[name]["lower"], aucs[name]["upper"], color=plAuc[0].get_color(), alpha=.2)
    plt.gca().invert_xaxis()
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15))
    plt.ylim(0.4, 1.1)
    plt.show()
    
    # TPR
    for typePlot in ["tnr", "tpr"]:
        plt.figure("Evolution {}".format(typePlot))
        plt.xlabel('Time before event (in minutes)')
        plt.ylabel('Evolution')
        plt.title('Evolution {} @{:.2f}% {}'.format(typePlot, percentage * 100, "fnr" if typePlot == "tnr" else "fpr"))
        plt.plot([min(temporalListLabels)[0].total_seconds() / 60., max(temporalListLabels)[0].total_seconds() / 60.], [0, 0], 'k--', label="Random Model")
        for name in aucs:
            plAuc = plt.plot(aucs[name].index.total_seconds() / 60., aucs[name][typePlot].values, label = name, ls = '--' if "train" in name.lower() else '-')
            plt.fill_between(aucs[name].index.total_seconds() / 60., aucs[name][typePlot].values - aucs[name][typePlot + '_wilson'], aucs[name][typePlot].values + aucs[name][typePlot + '_wilson'], color=plAuc[0].get_color(), alpha=.2)
        plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15))
        plt.gca().invert_xaxis()
        plt.ylim(-0.1, 1.1)
        plt.show()

def featuresImportanceCompare(listModels, featuresNames, top = None):
    """
        Plots the importance that each model assign to each features
        
        Arguments:
            listModels {List of (name, features_weights)*} -- Models to display
            featuresNames {str list} -- Same size than features_weights
    """
    weights_model = {}
    for (name, weights) in listModels:
        weights_model[name] = {f: w for w, f in zip(weights / np.max(np.abs(weights)), featuresNames)}
    weights_model = pd.DataFrame.from_dict(weights_model)

    # Sort by mean value of features
    weights_model = weights_model.reindex(weights_model.abs().mean(axis = "columns").sort_values().index, axis = 0)
    if top is not None:
        weights_model = weights_model.iloc[-top:]
    plt.figure("Features importance", figsize=(8, max(4.8, len(weights_model) / 5)))
    plt.xlabel('Weights')
    plt.ylabel('Features')
    plt.title('Features importance')
    weights_model.plot.barh(ax = plt.gca())
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15))
    plt.show()
