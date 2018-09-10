import argparse
import sys

import pandas as pd

TRADE_COLS = ['timestamp', 'symbol', 'side', 'price', 'size', 'tickDirection']
QUOTE_COLS = ['timestamp', 'symbol', 'bidSize', 'bidPrice', 'askPrice', 'askSize']


# Normalize data gotten from https://public.bitmex.com/


def get_args(argv=None, namespace=None):
    parser = argparse.ArgumentParser(description="Get bitmex data")

    parser.add_argument('-b', '--begin-time', type=pd.Timestamp, help="Example: '2018-04-01T00:00:01'")

    parser.add_argument('-e', '--end-time', type=pd.Timestamp, help="Example: '2018-04-01T00:00:10'")

    parser.add_argument('-n', '--num-ticks', type=int, help="Number of ticks to get'")

    parser.add_argument('-s', '--symbol', type=str, default='XBTUSD', help='Instrument symbol')

    parser.add_argument('-o', '--output', type=str, required=True, help='Output template path, NO EXTENSION. E.g., '
                                                                        '"~/data/20180701-%%TYPE%%", where %%TYPE%% is '
                                                                        'a placeholder for "trades" & "quotes"')

    parser.add_argument('-t', '--trades', type=str, required=True, help='Trades filename')

    parser.add_argument('-q', '--quotes', type=str, required=True, help='Quotes filename')

    args = parser.parse_args(argv, namespace)

    return args


def create_1m_candles(normalized_trades: pd.DataFrame):
    print('creating ohlcv ...')

    r = normalized_trades.resample('1min').agg({'symbol': 'last', 'price': 'ohlc', 'size': 'sum'})
    r.columns = r.columns.get_level_values(1)
    return r


def common_process(args, table: pd.DataFrame):
    table['timestamp'] = table['timestamp'].apply(lambda s: pd.Timestamp(s.replace('D', 'T')))
    if args.begin_time:
        table = table[table['timestamp'] >= args.begin_time]
    if args.end_time:
        table = table[table['timestamp'] < args.end_time]

    table = table.query('symbol == "{}"'.format(args.symbol))

    return table


def read_trades(args) -> pd.DataFrame:
    print('reading trades ...')
    trades = pd.read_csv(args.trades)[TRADE_COLS]
    trades = common_process(args, trades)

    if args.symbol == 'XBTUSD':
        trades = trades[trades.price > 1]
    else:
        trades = trades[trades.price > 0.01]

    # combine ticks with same timestamp and price level
    def combine(x):
        y = x.iloc[0].copy()
        y['size'] = x['size'].sum()
        return y

    trades = trades.groupby(['timestamp', 'price']).apply(combine).reset_index('price', drop=True).drop(
        columns=['timestamp'])

    if args.num_ticks:
        trades = trades.iloc[:args.num_ticks]

    rename = {'ZeroMinusTick': '0-', 'MinusTick': '--', 'ZeroPlusTick': '0+', 'PlusTick': '++'}
    trades['tickDirection'] = trades['tickDirection'].transform(lambda x: rename[x])

    return trades


def read_quotes(args):
    print('reading quotes ...')
    quotes = pd.read_csv(args.quotes)[QUOTE_COLS]
    quotes = common_process(args, quotes)

    if args.symbol == 'XBTUSD':
        quotes = quotes[(quotes.bidPrice > 1) & (quotes.bidPrice < quotes.askPrice)]
    else:
        quotes = quotes[(quotes.bidPrice > 0.01) & (quotes.bidPrice < quotes.askPrice)]

    quotes.set_index('timestamp', inplace=True)

    if args.num_ticks:
        quotes = quotes.iloc[:args.num_ticks]

    return quotes


def main():
    args = get_args(None)

    trades = read_trades(args)
    quotes = read_quotes(args)
    ohlcv = create_1m_candles(trades)

    df_map = {'trades': trades, 'quotes': quotes, 'ohlcv': ohlcv}

    out_map = {}
    for name in df_map.keys():
        out_map[name] = args.output.replace("%TYPE%", name)

    if out_map['trades'] == out_map['quotes']:
        for name in out_map.keys():
            out_map[name] += '-' + name

    for name in out_map.keys():
        out_map[name] += '.csv.gz'

    for name, df in df_map.items():
        df.to_csv(out_map[name], compression='gzip')

    return 0


if __name__ == '__main__':
    sys.exit(main())
