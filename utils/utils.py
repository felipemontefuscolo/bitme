import calendar
import contextlib
import datetime
import time
import dateutil.parser
import math
import sys
from decimal import Decimal

import pandas as pd
from typing import Union


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


def smart_close(handler):
    if handler != sys.stdout:
        handler.close()


class Day:
    def __init__(self, x):
        self._x = x
        pass

    def to_sec(self):
        return self._x * 86400

    def to_min(self):
        return self._x * 1440

    def to_hour(self):
        return self._x * 24


class Hour:
    def __init__(self, x):
        self._x = x
        pass

    def to_sec(self):
        return self._x * 3600

    def to_min(self):
        return self._x * 60


class Min:
    def __init__(self, x):
        self._x = x
        pass

    def to_sec(self):
        return self._x * 60


def get_current_ts():
    return int(time.time())


def to_iso_utc(ts):
    return datetime.datetime.utcfromtimestamp(ts).isoformat()


def to_iso_local(ts):
    return datetime.datetime.fromtimestamp(ts).isoformat()


def to_ts(iso_time):
    # return time.mktime(dateutil.parser.parse(iso_time).timetuple())
    return calendar.timegm(dateutil.parser.parse(iso_time).timetuple())


def isclose(a, b, rel_tol=1e-09, abs_tol=0.0):
    return abs(a - b) <= max(rel_tol * max(abs(a), abs(b)), abs_tol)


def floor_5(x):
    # example 10.75 -> 10.5
    # 21.5 -> 21.0 -> 10.5
    if x > 0:
        return math.floor(2. * x) * 0.5
    else:
        return math.ceil(2. * x) * 0.5


def read_data(file: str, begin: pd.Timestamp = None, end: pd.Timestamp = None) -> pd.DataFrame:
    timeparser = lambda s: pd.datetime.strptime(str(s), '%Y-%m-%dT%H:%M:%S')
    data = pd.DataFrame(pd.read_csv(file, parse_dates=True, index_col='time', date_parser=timeparser))

    if begin and end:
        data = data.loc[begin:end]
    elif begin:
        data = data.loc[begin:]
    elif end:
        data = data.loc[:end]
    return data


def to_nearest(num, tick_size: Union[float, str]):
    """Given a number, round it to the nearest tick. Very useful for sussing float error
       out of numbers: e.g. toNearest(401.46, 0.01) -> 401.46, whereas processing is
       normally with floats would give you 401.46000000000004.
       Use this after adding/subtracting/multiplying numbers."""
    tick_dec = Decimal(str(tick_size))
    return float((Decimal(round(num / float(tick_size), 0)) * tick_dec))


def to_str(number, precision=2):
    return "%.{}f".format(precision) % round(float(number), precision)
