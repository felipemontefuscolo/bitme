#!/usr/bin/env python
import math
import sys
import time
import contextlib
import GDAX
import re

from utils import Hour, get_current_ts, to_iso_utc, to_iso_local, Min, to_ts
from optparse import OptionParser

NUM_CANDLES_GDAX_LIMIT = 200


# granularity in sec
# start and end are timestamps
def download_and_print_candles(publicClient, granularity, begin_ts, end_ts, filename):
    n_candles = math.ceil((end_ts - begin_ts) / granularity)
    n_calls = int(math.ceil(n_candles / NUM_CANDLES_GDAX_LIMIT))
    print "getting candles ... 0%"
    with smart_open(filename) as fh:
        print >> fh, "time,low,high,open,close,volume"
        for i in range(0, n_calls):
            ss = begin_ts + i * NUM_CANDLES_GDAX_LIMIT * granularity
            ee = min(granularity * NUM_CANDLES_GDAX_LIMIT + ss, end_ts)
            candles_tmp = publicClient.getProductHistoricRates(granularity=granularity, start=to_iso_utc(ss),
                                                               end=to_iso_utc(ee))
            #print_candles(candles_tmp, '-')
            if 'message' in candles_tmp:
                raise Exception('invalid gdax message: ' + str(candles_tmp))

            candles_tmp = list(reversed(candles_tmp))
            candles_ts_to_utc(candles_tmp)
            for candle in candles_tmp:
                print >> fh, ','.join(str(e) for e in candle)
            time.sleep(1.005)
            print "getting candles ... " + str(int(float(i+1)/n_calls*100 + 0.5)) + "%"
            # print str((n_calls - i)/n_calls)*100 + '% done'
    print "done"


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


def candles_ts_to_utc(candles):
    last_ts = 0
    for candle in candles:
        if last_ts < int(candle[0]):
            last_ts = int(candle[0])
            candle[0] = to_iso_utc(candle[0])
        else:
            raise Exception(
                'invalid time stamp order: last = ' + str(to_iso_utc(last_ts)) + ', current = ' + to_iso_utc(candle[0]))


def check_candles_order(candles_utc):
    last_ts = 0
    for candle in candles_utc:
        current = to_ts(candle[0])
        if last_ts < current:
            last_ts = current
        else:
            raise Exception(
                'invalid time stamp order: last = ' + str(to_iso_utc(last_ts)) + ', current = ' + to_iso_utc(candle[0]))


def get_options():
    parser = OptionParser(usage="usage: %prog [options] filename (use - for stdout)",
                          version="%prog 1.0")
    parser.add_option("-d", "--duration", action="store", dest="duration", default="30m", help="number expression "
                                                                                               "followed by time unit")
    parser.add_option("-g", "--granularity", action="store", dest="granularity", default="5m", help="number "
                                                                                                    "expression "
                                                                                                    "followed by time "
                                                                                                    "unit")
    (options, args) = parser.parse_args()

    options.duration = parse_number(str(options.duration))
    options.granularity = parse_number(str(options.granularity))
    if len(args) < 1:
        parser.print_help()
        exit(-1)

    print "options: " + str({'duration(s)': options.duration, 'granularity(s)': options.granularity})
    print "filename: " + str(args[0])

    return options, args


def parse_number(x):
    # type: (str) -> int

    x = "".join(x.split())
    match = re.match(r"([0-9]+)([a-z]+)", x, re.I)
    unit = match.groups()[1]
    n = int(float(match.groups()[0]))
    if unit.lower() in ('s', 'sec', 'second', 'seconds'):
        y = n
    elif unit.lower() in ('m', 'min', 'minute', 'minutes'):
        y = Min(n).to_sec()
    elif unit.lower() in ('h', 'hour', 'hours'):
        y = Hour(n).to_sec()
    elif unit.lower() in ('d', 'day', 'days'):
        y = Hour(n).to_sec()
    else:
        raise Exception("invalid time unit")
    return y


def print_candles(candles, filename):
    with smart_open(filename) as fh:
        print >> fh, "time,low,high,open,close,volume"
        for candle in candles:
            print >> fh, ','.join(str(e) for e in candle)
    return


def main():
    # key = os.environ['GDAX_KEY']
    # b64secret = os.environ['GDAX_SECRET']
    # passphrase = os.environ['GDAX_PASS']

    (options, args) = get_options()

    filename = args[0]
    # how to print:
    # with smart_open(filename) as fh:
    #     print >> fh, 'some output'
    # return

    publicClient = GDAX.PublicClient(product_id="ETH-USD")
    # authClient = GDAX.AuthenticatedClient(key, b64secret, passphrase, product_id="BTC-USD")

    current_time = get_current_ts()
    print("current time is:\n"
          "ts: " + str(current_time) + "\n"
                                       "local: " + to_iso_local(current_time) + "\n"
                                                                                "utc: " + to_iso_utc(current_time))

    gran = options.granularity
    e = int(get_current_ts())
    e = e - e % gran
    b = e - options.duration

    print("getting candles for (utc) " + to_iso_utc(b) + " -- " + to_iso_utc(e))
    download_and_print_candles(publicClient, granularity=gran, begin_ts=b, end_ts=e, filename=filename)

    return 0


if __name__ == "__main__":
    sys.exit(main())
