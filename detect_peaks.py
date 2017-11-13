"""Detect peaks in data based on their amplitude and other features."""

from __future__ import division, print_function
import numpy as np

__author__ = "Felipe Montefuscolo"
__version__ = "0.1"


def strong_peak(x, y, min_hight=10., show=False, ax=None, verbose=False):
    """
    Detect max peak of periodograms of real data. It's not suitable for general purpose data.
    It assumes x[i] > 0 for all i, there are no nan and no inf.
    :param x: 1D array_like data. First and Last elements are not considered
    :param min_hight: times more than average to be considered peak
    :param show: bool, optional (default = False)
        if True (1), plot data in matplotlib figure.
    :param ax: a matplotlib.axes.Axes instance, optional (default = None).
    :param verbose: print info
    :return: index of the peak in `x`, or None
    """
    min_hight = float(min_hight)
    y = np.atleast_1d(y).astype('float64')
    n = len(y)
    if n < 3:
        return None
    if np.isnan(y).any():
        raise ValueError("input can not contain nan values")
    z = y[1:-1]
    mean = np.mean(z)
    imax = np.argmax(z) + 1
    if y[imax] < min_hight * mean:
        if verbose:
            print("Returning None because peak=%s < min_hight=%s * mean=%s" % (str(y[imax]), str(min_hight), str(mean)))
        result = None
    else:
        if verbose:
            print("index=%s, period=%s, mean=%s, min_hight=%s" % (str(imax), str(x[imax]), str(mean), str(min_hight)))
        result = imax

    if show and result:
        _plot(x, y, result, mean, ax)

    return result


def _plot(x, y, imax, mean, ax):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print('matplotlib is not available.')
    else:
        if ax is None:
            _, ax = plt.subplots(1, 1, figsize=(8, 4))

        ax.plot(x, y, 'b', lw=1)
        ax.plot(x[imax], y[imax], 'ro')
        n = len(x)
        ax.plot(x, [mean]*n, '--')

        # plt.grid()
        plt.show()
