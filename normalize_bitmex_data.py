import sys
import argparse
import pandas as pd

TRADE_COLS = ['timestamp', 'symbol', 'side', 'price', 'size', 'tickDirection']

# Normalize data gotten on https://public.bitmex.com/


def get_args(argv=None, namespace=None):
    parser = argparse.ArgumentParser(description="Get bitmex data")

    parser.add_argument('-b', '--begin-time', type=pd.Timestamp, help="Example: '2018-04-01T00:00:01'")

    parser.add_argument('-e', '--end-time', type=pd.Timestamp, help="Example: '2018-04-01T00:00:10'")

    parser.add_argument('-n', '--num-ticks', type=int, help="Number of ticks to get'")

    parser.add_argument('-s', '--symbol', type=str, default='XBTUSD', help='Instrument symbol')

    parser.add_argument('-o', '--output', type=str, required=True, help='Output filename')
    parser.add_argument('-i', '--input', type=str, required=True, help='Input filename')

    args = parser.parse_args(argv, namespace)

    return args


def main():
    args = get_args(None)

    r = pd.read_csv(args.input)[TRADE_COLS]

    if args.symbol:
        r = r.query('symbol == "{}"'.format(args.symbol))
        if args.symbol == 'XBTUSD':
            r = r[r.price > 1]

    r['timestamp'] = r['timestamp'].apply(lambda s: pd.Timestamp(s.replace('D', 'T')))
    if args.begin_time:
        r = r[r['timestamp'] >= args.begin_time]
    if args.end_time:
        r = r[r['timestamp'] < args.end_time]

    # combine ticks with same timestamp and price level
    def combine(x):
        y = x.iloc[0].copy()
        y['size'] = x['size'].sum()
        return y

    r = r.groupby(['timestamp', 'price']).apply(combine).reset_index('price', drop=True).drop(columns=['timestamp'])

    if args.num_ticks:
        r = r.iloc[:args.num_ticks]

    rename = {'ZeroMinusTick': '0-', 'MinusTick': '--', 'ZeroPlusTick': '0+', 'PlusTick': '++'}
    r['tickDirection'] = r['tickDirection'].transform(lambda x: rename[x])

    if '.gz' == args.output.lower()[-3:]:
        r.to_csv(args.output, compression='gzip')
    else:
        r.to_csv(args.output)

    return 0


if __name__ == '__main__':
    sys.exit(main())
