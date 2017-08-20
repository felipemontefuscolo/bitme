import csv
import math

from utils import to_ts, to_iso_utc


class Candles:
    def __init__(self, ts_l, low_l, high_l, open_l, close_l, volume_l):
        self.ts_l = ts_l
        self.low_l = low_l
        self.high_l = high_l
        self.open_l = open_l
        self.close_l = close_l
        self.volume_l = volume_l
        pass

    def printf(self):
        print "time,low,high,open,close,volume"
        for t, l, h, o, c, v in zip(self.ts_l, self.low_l, self.high_l, self.open_l, self.close_l, self.volume_l):
            tt = to_iso_utc(t) if t is not None else "None"
            print(','.join([tt, str(l), str(h), str(o), str(c), str(v)]))

    @classmethod
    def fromfilename(cls, filename):
        """
        :param filename: 1 second granular candles
        :return: Candles
        """
        ts_l = []
        low_l = []
        high_l = []
        open_l = []
        close_l = []
        volume_l = []

        with open(filename, 'r') as csvfile:
            csvfile.readline()  # discard header
            reader = csv.reader(csvfile, delimiter=',')
            for row in reader:
                Candles._append(ts_l, low_l, high_l, open_l, close_l, volume_l, row)

        return cls(ts_l, low_l, high_l, open_l, close_l, volume_l)

    @staticmethod
    def _append(ts_l, low_l, high_l, open_l, close_l, volume_l, csv_row):
        try:
            last_ts = int(ts_l[-1])
        except IndexError:
            last_ts = int(to_ts(csv_row[0]))

        new_ts = int(to_ts(csv_row[0]))
        skip_l = [None] * (new_ts - last_ts - 1)

        ts_l.extend(skip_l)
        low_l.extend(skip_l)
        high_l.extend(skip_l)
        open_l.extend(skip_l)
        close_l.extend(skip_l)
        volume_l.extend(skip_l)

        ts_l.append(new_ts)
        low_l.append(float(csv_row[1]))
        high_l.append(float(csv_row[2]))
        open_l.append(float(csv_row[3]))
        close_l.append(float(csv_row[4]))
        volume_l.append(float(csv_row[5]))

    def get_candles(self, granularity, begin_ts, end_ts):
        """
        :param granularity:
        :param begin_ts:  including
        :param end_ts: excluding
        :return if (end_ts-being_ts) / granularity is not integer, it may skip some candles at the end
        """
        ts_l = []
        low_l = []
        high_l = []
        open_l = []
        close_l = []
        volume_l = []

        granularity = (granularity if granularity <= len(self.ts_l) else len(self.ts_l))

        begin_ts = max(begin_ts, self.ts_l[0])
        end_ts = min(end_ts, self.ts_l[-1])

        n = int(math.ceil((end_ts + 1 - begin_ts) / float(granularity)))
        n = (1 if n == 0 else n)
        new_beg_ts = end_ts - n * granularity
        for i in range(0, n):
            idx = new_beg_ts + i * granularity - self.ts_l[0]
            if idx < 0:
                continue
            ts_l.append(new_beg_ts + i * granularity)
            low_l.append(_min(self.low_l[idx:idx + granularity]))
            high_l.append(max(self.high_l[idx:idx + granularity]))
            open_l.append(_first_non_none(self.open_l[idx:idx + granularity]))
            close_l.append(_last_non_none(self.close_l[idx:idx + granularity]))
            volume_l.append(_sum(self.volume_l[idx: idx + granularity]))
        return Candles(ts_l, low_l, high_l, open_l, close_l, volume_l)


def _min(v):
    m = [x for x in v if x is not None]
    return min(m if len(m) > 0 else [None])


def _max(v):
    return max(x if x is not None else None for x in v)


def _first_non_none(v):
    return next((x for x in v if x is not None), None)


def _last_non_none(v):
    return next((x for x in v if x is not None), None)


def _sum(v):
    s = sum(x if x is not None else 0 for x in v)
    return s if s != 0.0 else None