import matplotlib.pyplot as plt
import numpy as np
import math

colours = [
    'tab:blue',
    'tab:orange',
    'tab:green',
    'tab:red',
    'tab:purple',
    'tab:brown',
    'tab:pink',
    'tab:gray',
    'tab:olive',
    'tab:cyan',
]
n_bins = 10


def plot(*data):
    fig, (ax0, ax1) = plt.subplots(nrows=1, ncols=2)

    ax0.set_xlabel('Quantized (0-255)')
    ax0.set_ylabel('count', color="black")
    ax0.hist(
        data[0],
        n_bins,
        density=False,
        histtype='bar',
        color=colours[0],
    )

    ax1.set_xlabel('No Transforms')
    ax1.set_ylabel('count', color="black")
    ax1.hist(
        data[1],
        n_bins,
        density=False,
        histtype='bar',
        color=colours[0],
    )
    fig.tight_layout()
    plt.show()
