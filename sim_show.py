import argparse
import os

import datetime
import matplotlib.pyplot as plt
import pandas as pd
import sys

from matplotlib import ticker
from matplotlib.dates import date2num
from matplotlib.finance import candlestick2_ohlc, candlestick_ohlc
from plotly.offline.offline import matplotlib

from utils import read_data
import numpy as np


def get_args():
    parser = argparse.ArgumentParser(description='statistics properties of a file')
    parser.add_argument('-l', '--log_dir', type=str, help='sim output directory', required=True)
    parser.add_argument('-b', '--begin', type=str, help='begin time')
    parser.add_argument('-e', '--end', type=str, help='end time')

    args = parser.parse_args()

    if not os.path.isdir(args.log_dir):
        raise ValueError("invalid directory {}".format(args.log_dir))

    if args.begin:
        args.begin = pd.Timestamp(args.begin)
    if args.end:
        args.end = pd.Timestamp(args.end)

    return args


def add_time(df):
    df['time'] = [date2num(ii) for ii in df.index]
    return df


args = get_args()
fills_file = os.path.join(args.log_dir, 'fills.csv')
orders_file = os.path.join(args.log_dir, 'orders.csv')
pnl_file = os.path.join(args.log_dir, 'pnl.csv')
parameters_used = eval(open(os.path.join(args.log_dir, 'parameters_used'), 'r').readline())
candles_file = parameters_used[parameters_used.index('-f') + 1]

candles = read_data(candles_file)
ts_to_idx = dict(zip(candles.index, range(len(candles))))
idx_to_ts = dict(zip(range(len(candles)), candles.index))


def reset_idx(df):
    df.index = [ts_to_idx[ii] for ii in df.index]
    return df


#  fig, axes = plt.subplots(2, 1, figsize=(17, 9))  # bug in MacOX
fig = plt.figure(figsize=(17, 9))
axes = [fig.add_subplot(2, 1, 1),
        fig.add_subplot(2, 1, 2)]
candlestick2_ohlc(axes[0], candles['open'], candles['high'], candles['low'], candles['close'], width=0.6)
# axes[0].xaxis.set_major_locator(ticker.MaxNLocator(6))
# axes[0].xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: str(data.index[int(pos)])))

# data['time'] = [date2num(i) for i in data.index]
# ohlc = data[['time', 'open', 'high', 'low', 'close']]
# candlestick_ohlc(axes[0], ohlc.values, width=0.6)

fills = add_time(read_data(fills_file))
pnl = add_time(read_data(pnl_file))

# joining fills to candles so we align their timestamps
# fills = fills.reset_index().drop(columns=['time']).join(pnl.reset_index().drop(columns=['time']), on='symbol')
# joined = data.join(fills)
# joined['step'] = range(len(joined))
buys = reset_idx(fills.loc[fills['side'] == 'buy'][['price']])
sells = reset_idx(fills.loc[fills['side'] == 'sell'][['price']])
liqs = reset_idx(fills.loc[fills['type'] == 'market'][['price']])

axes[0].plot(buys['price'], linestyle='', color='g', marker='o')
axes[0].plot(sells['price'], linestyle='', color='b', marker='o')
axes[0].plot(liqs['price'], linestyle='', color='y', marker='o', markersize=10, markerfacecolor='none')
axes[0].set_title('fills')

cum_pnls = reset_idx(pnl[['cum_pnl']])
axes[1].plot(cum_pnls['cum_pnl'], '-o', markersize=3)
axes[1].set_title('cum pnl')

ax_pnl = axes[0].twinx()
ax_pnl.plot(cum_pnls['cum_pnl'], '-o', markersize=3)


def date_formatter(x, pos):
    try:
        return idx_to_ts[int(x)].strftime('%H:%M')
    except KeyError:
        return ''


for i in [0, 1]:
    y_formatter = matplotlib.ticker.ScalarFormatter(useOffset=False)
    axes[i].yaxis.set_major_formatter(y_formatter)
    axes[i].set_xlim(-1, len(candles))
    # axes[i].xaxis.set_major_locator(ticker.MaxNLocator(6))
    axes[i].xaxis.set_major_formatter(ticker.FuncFormatter(date_formatter))

# plt.tight_layout()
#fig.autofmt_xdate()
fig.set_tight_layout(True)
plt.show()

# fig = FF.create_candlestick(data['open'], data['high'], data['high'] * 0., data['close'], dates=data.index)
