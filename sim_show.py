import argparse
import os

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.finance import candlestick2_ohlc, candlestick_ohlc
from plotly.offline.offline import matplotlib


def get_args():
    parser = argparse.ArgumentParser(description='statistics properties of a file')
    parser.add_argument('-f', '--file', type=str, help='csv filename with candles data', required=True)
    parser.add_argument('-i', '--fills', type=str, help='fills file')
    parser.add_argument('-b', '--begin', type=str, help='begin time')
    parser.add_argument('-e', '--end', type=str, help='end time')

    args = parser.parse_args()

    if not os.path.isfile(args.file):
        raise ValueError("invalid file {}".format(args.file))

    if args.fills:
        if not os.path.isfile(args.fills):
            raise ValueError("invalid file {}".format(args.fills))

    if args.begin:
        args.begin = pd.Timestamp(args.begin)
    if args.end:
        args.end = pd.Timestamp(args.end)

    return args


def read_data(args, file):
    timeparser = lambda s: pd.datetime.strptime(str(s), '%Y-%m-%dT%H:%M:%S')
    Y = pd.DataFrame(pd.read_csv(file, parse_dates=True, index_col='time', date_parser=timeparser))

    if args.begin and args.end:
        Y = Y.loc[args.begin:args.end]
    elif args.begin:
        Y = Y.loc[args.begin:]
    elif args.end:
        Y = Y.loc[:args.end]
    return Y


args = get_args()
data = read_data(args, args.file)
fig, ax = plt.subplots()
candlestick2_ohlc(ax, data['open'], data['high'], data['low'], data['close'], width=0.6)

#candlestick_ohlc(ax, ohlc.values, width=0.6)

if args.fills:
    fills = read_data(args, args.fills)

    #data['buy'] = 0

    joined = data.join(fills)
    joined['step'] = range(len(joined))
    #joined.insert(0, 'time', range(len(joined.index)))
    buys = joined.loc[joined['side'] == 'buy']
    sells = joined.loc[joined['side'] == 'sell']
    liqs = joined.loc[joined['type'] == 'market']
    ax.plot(buys['step'], buys['price'], linestyle='', color='g', marker='o')
    ax.plot(sells['step'], sells['price'], linestyle='', color='b', marker='o')
    ax.plot(liqs['step'], liqs['price'], linestyle='', color='y', marker='o', markersize=10, markerfacecolor='none')


y_formatter = matplotlib.ticker.ScalarFormatter(useOffset=False)
ax.yaxis.set_major_formatter(y_formatter)
plt.show()

# fig = FF.create_candlestick(data['open'], data['high'], data['high'] * 0., data['close'], dates=data.index)
