import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from utils.ts_transformation import pushZeroTime, computeMeanStdCount

def evolutionPlot(predictions, truth, classes, clusters = None, invert = True, colors = {}):
    """
        Computes the evolution of the mean evolution for each class
        With separation in clusters

        Arguments:
            predictions {Dict / List} -- Label predictions
            truth {Dict / List} -- Ground truth (one label per time series)
            classes {Dict} -- Classes to consider to plot (key: Name to display, Value: label)
        
        Keyword Arguments:
            label {str} -- Legend to plot (default: {"Model"})
            newFigure {str} -- Display on a given figure (default: {None} - Create new figure)
            clusters {int / List of deltatime} -- Number to clsuter to form or list of time bounds (default: {None} - No cluster)
    """
    # Computes duration
    duration = {p: (max(predictions[p].index) - min(predictions[p].index)) for p in predictions}

    if clusters is None:
        clusters = [pd.to_timedelta(0, unit='m'), max([duration[p] for p in predictions])]
    elif type(clusters) is int:
        # Finds the different boundaries to divide the array in equal parts
        duration_list = np.sort([duration[p] for p in predictions]).tolist()
        length = int(len(duration) /  (clusters + 1))
        clusters = [pd.to_timedelta(0, unit='m')] + [duration_list[i * length] for i in range(1, clusters)] + [duration_list[-1]]
    assert len(clusters) > 1, "Bounds has to have more than one element"

    # Creates the subplots
    fig, axes = plt.subplots(len(clusters) - 1, 1, sharex = True, sharey = True, squeeze = False, figsize = (8, 10))

    for i in range(len(clusters) - 1):
        axes[i,0].set_title("Duration {} to {}".format(clusters[i], clusters[i+1]))
        # Select time series in the range
        selection = [p for p in predictions if ((clusters[i] < duration[p]) and (duration[p] <= clusters[i+1]))]

        # Push all points on a similar scale
        ts = {c : [pushZeroTime(predictions[p], invert = invert) for p in selection if classes[c] == truth[p]] for c in classes}
        for c in classes:
            if len(ts[c]) > 1:
                res = computeMeanStdCount(ts[c])
                res = res[res["count"] > 1]
                
                # Mean
                axes[i, 0].scatter(res.index.total_seconds() / 3600., res["mean"].values, c = colors[c] if c in colors else None, alpha = 0.5)
                plMean = axes[i, 0].plot(res.index.total_seconds() / 3600., res["mean"], label = c + " ({})".format(len(ts[c])), c = colors[c] if c in colors else None, alpha = 0.5)
                
                # Confidence bounds
                wilson = 1.96 * np.sqrt(res["mean"] * (1 - res["mean"])/res["count"])
                axes[i,0].fill_between(res.index.total_seconds() / 3600., res["mean"] + wilson, res["mean"] - wilson, color = plMean[0].get_color(), alpha = 0.25)


            # Legend
            axes[i,0].legend(loc='upper right', bbox_to_anchor=(1.3, 1))

    # Create legends 
    axes[0,0].invert_xaxis()
    fig.add_subplot(111, frameon = False)
    plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
    if invert:
        plt.xlabel('Time to event (in hour)')
    else:
        plt.xlabel('Time after event (in hour)')
    plt.ylabel('Predictions')
    plt.grid(alpha = .2)

    return fig, axes