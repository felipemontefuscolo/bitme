import argparse
import os

import matplotlib.pyplot as plt
import pandas as pd
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


args = get_args()
data = read_data(args)

fig, ax = plt.subplots()
candlestick2_ohlc(ax, data['open'], data['high'], data['low'], data['close'], width=0.6)
plt.show()

# fig = FF.create_candlestick(data['open'], data['high'], data['high'] * 0., data['close'], dates=data.index)
