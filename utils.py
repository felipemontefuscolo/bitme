import calendar
import datetime
import time

import dateutil.parser


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
