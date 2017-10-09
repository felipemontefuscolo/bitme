#!/usr/bin/env python
import contextlib
import datetime
import sys
import time
from optparse import OptionParser

import dateutil
import dateutil.parser
import swagger_client
from swagger_client.rest import ApiException

MAX_NUM_CANDLES_BITMEX = 500


@contextlib.contextmanager
def smart_open(filename=None):
    if filename and filename != '-':
        fh = open(filename, 'w')
    else:
        fh = sys.stdout

    try:
        yield fh
    finally:
        if fh is not sys.stdout:
            fh.close()


def get_duration_in_min(start, end):
    # type: (datetime.datetime, datetime.datetime) -> int
    c = end - start
    x = divmod(c.days * 86400 + c.seconds, 60)
    return x[0] + (x[1] > 0)


def print_file(file_or_stdout, api_instance, bin_size, partial, symbol, reverse, start_time, end_time):

    duration_min = get_duration_in_min(start_time, end_time)
    assert duration_min > 0

    tmp = divmod(duration_min, MAX_NUM_CANDLES_BITMEX)
    num_pages = tmp[0] + (tmp[1] > 0)  # extra page for left over

    with smart_open(file_or_stdout) as fh:
        print >> fh, "time,open,high,low,close,volume"

        count = tmp[1] + 1 if tmp[1] > 0 else MAX_NUM_CANDLES_BITMEX
        for i in reversed(range(num_pages)):
            sys.stdout.write(
                "progress: %d out of %d pages (%.2f%%)   \r" % (num_pages-i, num_pages, 100 * float(num_pages-i) / num_pages))
            sys.stdout.flush()

            page = api_instance.trade_get_bucketed(bin_size=bin_size, partial=partial, symbol=symbol,
                                                   count=count, start=i * MAX_NUM_CANDLES_BITMEX,
                                                   reverse=reverse, start_time=start_time, end_time=end_time)
            for line in reversed(page):
                print >> fh, ','.join([line.timestamp.strftime('%Y-%m-%dT%H:%M:%S'),
                                       str(line.open),
                                       str(line.high),
                                       str(line.low),
                                       str(line.close),
                                       str(line.volume)])
            time.sleep(1.001)
            count = MAX_NUM_CANDLES_BITMEX
        print ""


def main():
    start_time = dateutil.parser.parse('2017-10-01T19:26:00')  # datetime | Starting date filter for results. (optional)
    end_time = dateutil.parser.parse('2017-10-08T19:26:00')  # datetime | Ending date filter for results. (optional)
    file_or_stdout = 'data/bitmex_1week.csv'

    # create an instance of the API class
    configuration = swagger_client.Configuration()
    configuration.host = 'https://www.bitmex.com/api/v1'
    api_instance = swagger_client.TradeApi(swagger_client.ApiClient(configuration))
    bin_size = '1m'  # str | Time interval to bucket by. Available options: [1m,5m,1h,1d]. (optional) (default to 1m)
    partial = False  # bool | If true, will send in-progress (incomplete) bins for the current time period. (optional) (default to false)
    symbol = 'XBTUSD'  # str | Instrument symbol. Send a bare series (e.g. XBU) to get data for the nearest expiring contract in that series.  You can also send a timeframe, e.g. `XBU:monthly`. Timeframes are `daily`, `weekly`, `monthly`, `quarterly`, and `biquarterly`. (optional)
    reverse = True  # bool | If true, will sort results newest first. (optional) (default to false)

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
