import calendar
import datetime
import time

import dateutil.parser


class Day:
    def __init__(self):
        pass

    @staticmethod
    def to_sec(x):
        return x * 86400

    @staticmethod
    def to_min(x):
        return x * 1440

    @staticmethod
    def to_hour(x):
        return x * 24


class Hour:
    def __init__(self):
        pass

    @staticmethod
    def to_sec(x):
        return x * 3600

    @staticmethod
    def to_min(x):
        return x * 60


class Min:
    def __init__(self):
        pass

    @staticmethod
    def to_sec(x):
        return x*60


def get_current_ts():
    return int(time.time())


def to_iso_utc(ts):
    return datetime.datetime.utcfromtimestamp(ts).isoformat()


def to_iso_local(ts):
    return datetime.datetime.fromtimestamp(ts).isoformat()


def to_ts(iso_time):
    # return time.mktime(dateutil.parser.parse(iso_time).timetuple())
    return calendar.timegm(dateutil.parser.parse(iso_time).timetuple())
