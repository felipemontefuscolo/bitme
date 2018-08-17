#!/usr/bin/env python


# WARNING: the trades don't match their trades in one can get this data on https://public.bitmex.com/
# Possibly package loss?


import argparse
import sys
import time

import pandas as pd
import swagger_client

from dateutil.tz import tzutc

from utils.utils import smart_open

MAX_NUM_REQ_BITMEX = 500

TRADE_COLUMNS = ['timestamp', 'symbol', 'price', 'size', 'side']


# Note: one can get this data on https://public.bitmex.com/

def get_args(args=None, namespace=None):
    parser = argparse.ArgumentParser(description="Get bitmex data")

    parser.add_argument('-b', '--begin-time', type=pd.Timestamp, required=True, help="Example: '2018-04-01T00:00:01'")

    parser.add_argument('-e', '--end-time', type=pd.Timestamp, required=True, help="Example: '2018-04-01T00:00:10'")

    parser.add_argument('-s', '--symbol', type=str, default='XBTUSD',
                        help='Instrument symbol. Send a bare series (e.g. XBU) to get data for the nearest expiring'
                             'contract in that series. You can also send a timeframe, e.g. `XBU:monthly`. '
                             'Timeframes are `daily`, `weekly`, `monthly`, `quarterly`, and `biquarterly`. (optional)')

    parser.add_argument('-o', '--file-or-stdout', type=str, required=True, help='Output filename or "-" for stdout')

    args = parser.parse_args(args, namespace)

    args.begin_time = args.begin_time.to_pydatetime().astimezone(tzutc())
    args.end_time = args.end_time.to_pydatetime().astimezone(tzutc())

    return args


def main():
    args = get_args()

    # create an instance of the API class
    configuration = swagger_client.Configuration()
    configuration.host = 'https://www.bitmex.com/api/v1'
    trade_api = swagger_client.TradeApi(swagger_client.ApiClient(configuration))

    def print_progress(p):
        sys.stdout.write("progress: completed (%.2f%%)   \r" % p)
        sys.stdout.flush()

    with smart_open(args.file_or_stdout) as fh:
        print(','.join(TRADE_COLUMNS), file=fh)
        start = 0
        last_time_seen = args.begin_time
        while last_time_seen < args.end_time:
            page = trade_api.trade_get(symbol='XBTUSD',
                                       count=MAX_NUM_REQ_BITMEX,
                                       reverse='false',
                                       columns=','.join(TRADE_COLUMNS),
                                       start_time=args.begin_time,
                                       end_time=args.end_time,
                                       start=start
                                       )
            if page:
                last_time_seen = page[-1].timestamp
                start += len(page)
                for line in page:
                    if line.timestamp >= args.end_time:
                        break
                    print(','.join([str(line.timestamp)[:-9],
                                    str(line.symbol),
                                    str(line.price),
                                    str(-1 if line.side[0] == 'S' else +1),
                                    str(line.size)]), file=fh)
                progress = (min(last_time_seen, args.end_time) - args.begin_time) / (
                        args.end_time - args.begin_time) * 100
                print_progress(progress)
            else:
                break

            # we don't want to break the limit
            time.sleep(1.001)
        print_progress(100)


if __name__ == "__main__":
    sys.exit(main())
