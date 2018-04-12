#!/usr/bin/env python
import contextlib
import sys
import time

import swagger_client
from pandas import Timestamp, Timedelta
from swagger_client.rest import ApiException

from utils import smart_open

MAX_NUM_CANDLES_BITMEX = 500


def print_file(file_or_stdout, api_instance, bin_size, partial, symbol, reverse, start_time, end_time):
    chunks = split_in_chunks(start_time, end_time, MAX_NUM_CANDLES_BITMEX, bin_size)

    with smart_open(file_or_stdout) as fh:
        print >> fh, "time,open,high,low,close,volume"

        num_pages = len(chunks)
        for i in range(num_pages):
            chunk = chunks[i]
            s = chunk[0]
            e = chunk[1]

            count = (e - s) / Timedelta(bin_size)

            page = api_instance.trade_get_bucketed(
                bin_size=bin_size,
                partial=partial,
                symbol=symbol,
                count=count,
                start=0.0,
                reverse=reverse,
                start_time=s,
                end_time=e)

            print "from {} to {}: {} candles downloaded".format(s, e, len(page))

            #  TODO: bitmex has a bug where the high is not the highest value !!!!!
            for line in reversed(page):
                print >> fh, ','.join([line.timestamp.strftime('%Y-%m-%dT%H:%M:%S'),
                                       str(line.open),
                                       str(max(line.high, line.open)),
                                       str(min(line.low, line.open)),
                                       str(line.close),
                                       str(line.volume)])
            sys.stdout.write(
                "progress: completed %d out of %d pages (%.2f%%)   \r" %
                (i + 1, num_pages, 100 * float(i + 1) / num_pages))
            sys.stdout.flush()
            time.sleep(1.001)
        print ""


def split_in_chunks(start, end, chunk_size, bucket_size):
    # type: (Timestamp, Timestamp, int, str) -> list
    # bucket size options:  1m, 5m, 1h, 1d
    i = start
    r = []
    dt = chunk_size * Timedelta(bucket_size)
    while i <= end:
        r += [(i, min(end, i + dt))]
        i += dt
    return r


def main():
    start_time = Timestamp('2017-12-12T00:00:01')
    end_time = Timestamp('2018-01-12T00:00:01')

    file_or_stdout = 'data/bitmex_1month.csv'

    # create an instance of the API class
    configuration = swagger_client.Configuration()
    configuration.host = 'https://www.bitmex.com/api/v1'
    api_instance = swagger_client.TradeApi(swagger_client.ApiClient(configuration))
    bin_size = '1m'  # str | Time interval to bucket by. Available options: [1m,5m,1h,1d]. (optional) (default to 1m)
    partial = False  # bool | If true, will send in-progress (incomplete) bins for the current time period. (optional) (default to false)
    symbol = 'XBTUSD'  # str | Instrument symbol. Send a bare series (e.g. XBU) to get data for the nearest expiring contract in that series.  You can also send a timeframe, e.g. `XBU:monthly`. Timeframes are `daily`, `weekly`, `monthly`, `quarterly`, and `biquarterly`. (optional)
    reverse = False  # bool | If true, will sort results newest first. (optional) (default to false)

    print "print to file " + (file_or_stdout if file_or_stdout is not '-' else 'std output')
    try:
        print_file(file_or_stdout=file_or_stdout,
                   api_instance=api_instance,
                   bin_size=bin_size, partial=partial, symbol=symbol,
                   reverse=reverse,
                   start_time=start_time, end_time=end_time)
    except ApiException as e:
        print("Exception when calling TradeApi->trade_get_bucketed: %s\n" % e)

    return 0


if __name__ == "__main__":
    sys.exit(main())
