import argparse
import sys
import os
import pandas as pd

from segmented_ols import seg_lin_reg, plot_segments
import numpy as np
from matplotlib.pylab import *
from statsmodels.graphics.tsaplots import utils
from matplotlib.finance import candlestick_ohlc

import plotly.plotly as py
from plotly.tools import FigureFactory as FF
from datetime import datetime

def get_args():
    parser = argparse.ArgumentParser(description='statistics properties of a file')
    parser.add_argument('-f', '--file', type=str, help='csv filename with candles data', required=True)
    parser.add_argument('-b', '--begin', type=str, help='begin time')
    parser.add_argument('-e', '--end', type=str, help='end time')

    args = parser.parse_args()

    if not os.path.isfile(args.file):
        raise ValueError("invalid file {}".format(args.file))

    if args.begin:
        args.begin = pd.Timestamp(args.begin)
    if args.end:
        args.end = pd.Timestamp(args.end)

    return args


def read_data(args):
    timeparser = lambda s: pd.datetime.strptime(str(s), '%Y-%m-%dT%H:%M:%S')
    Y = pd.DataFrame(pd.read_csv(args.file, parse_dates=True, index_col='time', date_parser=timeparser))

    if args.begin and args.end:
        Y = Y.loc[args.begin:args.end]
    elif args.begin:
        Y = Y.loc[args.begin:]
    elif args.end:
        Y = Y.loc[:args.end]
    return Y


def plot_slope_duration(segs, ax=None):
    # type: (list, AxesSubplot) -> None
    fig, ax = utils.create_mpl_ax(ax)

    y = [seg[0].stop - seg[0].start for seg in segs]
    x = [seg[2] for seg in segs]
    ax.vlines(x, [0], y, 'r')
    ax.set_xlabel('slope')
    ax.set_ylabel('duration')

    return fig


def candle_grew(data, j):
    return data['close'].iloc[j]- data['open'].iloc[j] > 0


def summary_grow(limit_time_ahead, data):
    x = np.arange(len(data.index))

    predicted_right = 0
    predicted_wrong = 0

    grows = []

    # the hypothesis is that the price will fall below a certain values after the price grows n_conseq time

    for i in range(0, len(x)):    #  V ^ ^ (^) X X X
        if candle_grew(data, i):
            continue
        duration = 0

        for j in range(i+1, len(x)):
            if not candle_grew(data, j):
                break
            duration += 1

        if duration == 0:
            continue

        grouth = data['close'].iloc[i + duration] - data['open'].iloc[i]

        lower_price = 999999999.9999999
        for j in range(i+1, min(len(x), i+limit_time_ahead+1)):
            if data.iloc[j]['low'] < lower_price:
                lower_price = data['low'].iloc[j]

        grows += [(duration, grouth, lower_price)]


def validate_ohlc(open, high, low, close, data):
    ix = []
    for lst in [open, low, close]:
        for index in range(len(high)):
            if high[index] < lst[index]:
                ix += [index]
    print(data.iloc[ix])

def main():
    args = get_args()

    data = read_data(args)
    x = np.arange(len(data.index))
    y = data['close'].values

    ax = subplots()

    #print(summary_grow(100, data))

    validate_ohlc(data['open'], data['high'], data['low'], data['close'], data)
    fig = FF.create_candlestick(data['open'], data['high'], data['high'] * 0., data['close'], dates=data.index)

if __name__ == '__main__':
    sys.exit(main())
