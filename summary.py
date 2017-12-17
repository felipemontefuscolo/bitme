import argparse
import sys
import os
import pandas as pd
from collections import defaultdict

from segmented_ols import seg_lin_reg, plot_segments
import numpy as np
from matplotlib.pylab import *
from statsmodels.graphics.tsaplots import utils
from matplotlib.finance import candlestick_ohlc

import plotly
import plotly.plotly as py
import plotly.figure_factory as pff
# from plotly.tools import FigureFactory as FF
import plotly.figure_factory as FF
from datetime import datetime
from plotly.offline import plot, iplot
from matplotlib.finance import candlestick2_ohlc


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
    return data['close'].iloc[j] - data['open'].iloc[j] > 0


def candle_fell(data, j):
    return data['close'].iloc[j] - data['open'].iloc[j] < 0


def sfloat(x):
    # return str(x[0]) + '\xc2\xb1' + str(x[1])
    return '{0:0.1f}'.format(x[0]) + '\xc2\xb1' + '{0:0.1f}'.format(x[1])


def update_best_picks(best_picks, n_bars, n_occurs, fall_or_growth, recover_or_retraction, freq):
    profit = n_occurs * fall_or_growth * recover_or_retraction * (2.*freq - 100.)
    best_picks[profit] = (n_bars, n_occurs, fall_or_growth, recover_or_retraction, freq)
    if len(best_picks) > 10:
        best_picks.pop(min(best_picks))
    pass


def update_best_picks2(best_picks, profit, n_bars, change_ratio, reversal_ratio):
    best_picks[profit] = (n_bars, '{0:.2f}%'.format((100 * change_ratio)), '{0:.2f}%'.format((100 * reversal_ratio)))
    if len(best_picks) > 5:
        best_picks.pop(min(best_picks))
    pass


def trend_reversal_analysis(time_ahead, data, direction=-1, change_ratio_margins=None, max_reversal_margins=None, show_as_percentage=True):
    x = np.arange(len(data.index))

    if change_ratio_margins is None:
        change_ratio_margins = np.linspace(0., 0.01, 15)
    else:
        change_ratio_margins = np.array(sorted(change_ratio_margins))

    if max_reversal_margins is None:
        max_reversal_margins = np.linspace(0., 1., 11)
    else:
        max_reversal_margins = np.array(sorted(max_reversal_margins))

    # n_bars -> matrix(fall_ratio; max_recover_ratio)
    P = defaultdict(lambda: np.zeros((len(change_ratio_margins), len(max_reversal_margins))).astype('float64'))
    # for nomalization purposes. note: P( f >= F | n == N, g >= G) = Q(f >= F, n == N, g >= G) / Q(n == N, g >= G)
    # Q = quantity
    Q = defaultdict(lambda: np.zeros(len(change_ratio_margins)).astype('float64'))

    # S = simulate a profit
    S = defaultdict(lambda: np.zeros((len(change_ratio_margins), len(max_reversal_margins))).astype('float64'))

    print('running with limit_time_ahead = {}'.format(time_ahead))

    for i in range(0, len(x)):
        if not direction * (data['close'].iloc[i] - data['open'].iloc[i]) > 0:
            continue

        n_bars = 1
        for j in reversed(range(0, i)):
            if not direction * (data['close'].iloc[j] - data['open'].iloc[j]) > 0:
                break
            n_bars += 1

        open = data['open'].iloc[i - n_bars + 1]
        possible_close = data['close'].iloc[i]
        liquidation_price = data['close'].iloc[min(i + time_ahead, len(x)-1)]
        change_ratio = abs(float(possible_close - open)) / open  # unsigned

        future_price_peak = direction * float('inf')
        for j in range(i + 1, min(i + time_ahead + 1, len(x))):
            # note: min(a,b) = -max(-a,-b)
            candle_j_peak = -direction * max(-direction * data['open'].iloc[j], -direction * data['close'].iloc[j])
            future_price_peak = -direction * max(-direction * future_price_peak,
                                                 -direction * candle_j_peak)
        if future_price_peak == direction * float('inf'):
            future_price_peak = possible_close
        else:
            assert -direction * future_price_peak >= -direction * possible_close

        max_reversal_ratio = (possible_close - future_price_peak) / (possible_close - open)
        assert max_reversal_ratio >= 0.

        for change_margin in range(len(change_ratio_margins)):
            for reversal_margin in range(len(max_reversal_margins)):
                if max_reversal_ratio >= max_reversal_margins[reversal_margin] and change_ratio >= change_ratio_margins[change_margin]:
                    P[n_bars][change_margin][reversal_margin] += 1.
                    S[n_bars][change_margin][reversal_margin] += abs(max_reversal_margins[reversal_margin]*(possible_close - open))
                if max_reversal_ratio < max_reversal_margins[reversal_margin] and change_ratio >= change_ratio_margins[change_margin]:
                    S[n_bars][change_margin][reversal_margin] -= abs(possible_close - liquidation_price)

            if change_ratio >= change_ratio_margins[change_margin]:
                Q[n_bars][change_margin] += 1

    # Profit = N * (recover * fall * open) * freq
    # best_picks: profit -> pars
    best_picks = {}

    for n_bars in P.keys():
        print
        print('given n_bars = ' + str(n_bars))
        data = P[n_bars]
        counts = Q[n_bars]

        if show_as_percentage:
            data = 100 * data / (counts.reshape((len(data), 1)) + 1e-15) + 0.5

        # update best pick
        # for i in range(len(change_ratio_margins)):
        #     for j in range(len(max_reversal_margins)):
        #         update_best_picks(best_picks,
        #                           n_bars, counts[i], change_ratio_margins[i], max_reversal_margins[j], data[i][j])
        profits = S[n_bars]
        # update best pick
        for i in range(len(change_ratio_margins)):
            for j in range(len(max_reversal_margins)):
                update_best_picks2(best_picks, profits[i][j], n_bars, change_ratio_margins[i], max_reversal_margins[j])

        if False:  # print P
            data2 = np.zeros((data.shape[0], data.shape[1] + 1))
            data2[:, 1:] = data
            data2[:, 0] = Q[n_bars]
            data2 = data2.astype('int')
            #df = pd.DataFrame(data=data, index=growth_ratio_margins, columns=max_fall_margins)
            df = pd.DataFrame(data=data2)
            # df.sort_index(inplace=True)
            # df = df.reindex(sorted(df.columns), axis=1)

            #to_return = pd.DataFrame(df)

            #columns = ['{}%'.format(int(100 * x)) if x != (-float('inf')) else 'total' for x in max_fall_margins]
            columns = ['count'] + ['{}%'.format(int(100 * x)) for x in max_reversal_margins]
            #columns[1] += '(trend\'s close)'
            #columns[-1] += '(trend\' open)'
            df.columns = columns
            df.index = ['{0:.2f}%'.format((100 * x)) for x in change_ratio_margins]
            df.index.name = 'given $ change >='
            df.columns.name = 'obtained reversal $ >='
            print(df.to_string())

        if True:  # print S
            # df = pd.DataFrame(data=data, index=growth_ratio_margins, columns=max_fall_margins)
            df = pd.DataFrame(data=S[n_bars])
            # df.sort_index(inplace=True)
            # df = df.reindex(sorted(df.columns), axis=1)

            # to_return = pd.DataFrame(df)

            # columns = ['{}%'.format(int(100 * x)) if x != (-float('inf')) else 'total' for x in max_fall_margins]
            columns = ['{}%'.format(int(100 * x)) for x in max_reversal_margins]
            # columns[1] += '(trend\'s close)'
            # columns[-1] += '(trend\' open)'
            df.columns = columns
            df.index = ['{0:.2f}%'.format((100 * x)) for x in change_ratio_margins]
            df.index.name = 'given $ change >='
            df.columns.name = 'obtained reversal $ >='
            print(df.to_string())
        #break

    print("")
    for p in reversed(sorted(best_picks.iteritems())):
        print p


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
    data = data[:].iloc[0:-1]
    x = np.arange(len(data.index))
    y = data['close'].values

    trend_reversal_analysis(15, data, direction=-1,
                            change_ratio_margins=np.concatenate([np.linspace(0.0005, 0.005, 10), np.linspace(0.006, 0.035, 9)]),
                            max_reversal_margins=np.linspace(0.1, 1., 10),
                            show_as_percentage=True)

    # validate_ohlc(data['open'], data['high'], data['low'], data['close'], data)
    if False:
        fig, ax = plt.subplots()
        candlestick2_ohlc(ax, data['open'], data['high'], data['low'], data['close'], width=0.6)
        show()

    validate_ohlc(data['open'], data['high'], data['low'], data['close'], data)
    fig = FF.create_candlestick(data['open'], data['high'], data['high'] * 0., data['close'], dates=data.index)


if __name__ == '__main__':
    sys.exit(main())

 # profit =