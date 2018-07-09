#!/usr/bin/env python
import sys
import time

import swagger_client
from swagger_client.rest import ApiException

from utils.utils import smart_open

import argparse
import pandas as pd

MAX_NUM_CANDLES_BITMEX = 500


def print_file(file_or_stdout, api_instance, bin_size, partial, symbol, reverse, start_time, end_time):
    chunks = split_in_chunks(start_time, end_time, MAX_NUM_CANDLES_BITMEX, bin_size)

    with smart_open(file_or_stdout) as fh:
        print("time,open,high,low,close,volume", file=fh)

        num_pages = len(chunks)
        for i in range(num_pages):
            chunk = chunks[i]
            s = chunk[0]
            e = chunk[1]

            count = (e - s) / pd.Timedelta(bin_size)

            page = api_instance.trade_get_bucketed(
                bin_size=bin_size,
                partial=partial,
                symbol=symbol,
                count=count,
                start=0.0,
                reverse=reverse,
                start_time=s,
                end_time=e)

            print("from {} to {}: {} candles downloaded".format(s, e, len(page)))

            #  TODO: bitmex has a bug where the high is not the highest value !!!!!
            for line in reversed(page):
                print(','.join([line.timestamp.strftime('%Y-%m-%dT%H:%M:%S'),
                                str(line.open),
                                str(max(line.high, line.open)),
                                str(min(line.low, line.open)),
                                str(line.close),
                                str(line.volume)]), file=fh)
            sys.stdout.write(
                "progress: completed %d out of %d pages (%.2f%%)   \r" %
                (i + 1, num_pages, 100 * float(i + 1) / num_pages))
            sys.stdout.flush()
            time.sleep(1.001)
        print("")


def split_in_chunks(start: pd.Timedelta, end: pd.Timedelta, chunk_size: int, bucket_size: str):
    i = start
    r = []
    dt = chunk_size * pd.Timedelta(bucket_size)
    while i <= end:
        r += [(i, min(end, i + dt))]
        i += dt
    return r


def get_args(args=None, namespace=None):
    parser = argparse.ArgumentParser(description="Get bitmex data")

    parser.add_argument('-b', '--begin-time', type=pd.Timestamp, required=True, help="Example: '2018-04-01T00:00:01'")

    parser.add_argument('-e', '--end-time', type=pd.Timestamp, required=True, help="Example: '2018-04-02T00:00:01'")

    parser.add_argument('-s', '--symbol', type=str, default='XBTUSD',
                        help='Instrument symbol. Send a bare series (e.g. XBU) to get data for the nearest expiring'
                             'contract in that series. You can also send a timeframe, e.g. `XBU:monthly`. '
                             'Timeframes are `daily`, `weekly`, `monthly`, `quarterly`, and `biquarterly`. (optional)')

    parser.add_argument('-z', '--bin-size', choices=('1m', '5m', '1h', '1d'), default='1m', type=str,
                        help='Time interval to bucket by')

    parser.add_argument('-o', '--file-or-stdout', type=str, required=True, help='Output filename or "-" for stdout')

    parser.add_argument('--partial', action='store_true', default=False, )

    args = parser.parse_args(args, namespace)

    return args


def main():
    args = get_args()

    # create an instance of the API class
    configuration = swagger_client.Configuration()
    configuration.host = 'https://www.bitmex.com/api/v1'
    api_instance = swagger_client.TradeApi(swagger_client.ApiClient(configuration))

    print("print to file " + (args.file_or_stdout if args.file_or_stdout is not '-' else 'std output'))
    try:
        print_file(file_or_stdout=args.file_or_stdout,
                   api_instance=api_instance,
                   bin_size=args.bin_size, partial=args.partial, symbol=args.symbol,
                   reverse=False,
                   start_time=args.begin_time, end_time=args.end_time)
    except ApiException as e:
        print("Exception when calling TradeApi->trade_get_bucketed: %s\n" % e)

    return 0


if __name__ == "__main__":
    sys.exit(main())
