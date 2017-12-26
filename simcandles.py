import csv
import math
import pandas as pd

from candles import Candles
from utils import to_ts, to_iso_utc


class CandlesViewIterator:
    def __init__(self, candles):
        # type: (SimCandles) -> ()
        self.candles = candles  # is this conversion inefficient?
        self.idx = int(0)
        self.candles_size = candles.size()
        pass

    def __iter__(self):
        return self

    def __next__(self):
        self.idx += 1
        if self.idx > self.candles_size:
            raise StopIteration
        return self.candles.data.iloc[0:self.idx]

    next = __next__  # python 2 compatibility


class SimCandles(Candles):
    def __init__(self, filename=None, data=None):
        # type: (str, pd.DataFrame) -> None
        Candles.__init__(self)
        if filename is not None and data is None:
            timeparser = lambda s: pd.datetime.strptime(str(s), '%Y-%m-%dT%H:%M:%S')
            self.data = pd.DataFrame(  # is this conversion inefficient?
                pd.read_csv(filename, parse_dates=True, index_col='time', date_parser=timeparser))
        elif data is not None and filename is None:
            self.data = data
        else:
            raise ValueError("XOR(filename==None, data==None) should be True")

    def at(self, index):
        return self.data.iloc[index]

    def views(self):
        return CandlesViewIterator(self)

    def size(self):
        return self.data.shape[0]

    def to_csv(self):
        return self.data.to_csv()

    def printf(self):
        print(self.data)

    def last_timestamp(self):
        # type: () -> pd.Timestamp
        return self.data.iloc[-1].name

    def last_price(self):
        # type: () -> float
        return self.data.iloc[-1].close

    def sample_candles(self, granularity, begin_ts, end_ts):
        # type: (pd.Timedelta, pd.Timestamp, pd.Timestamp) -> SimCandles
        '''
        :param granularity: example: pd.Timedelta(hours=1)'
        :param begin_ts: example: SimCandles().iloc[0].name - pd.Timedelta(hours=1)
        :param end_ts: example: SimCandles().iloc[0].name
        :return: sampled SimCandles
        '''
        sliced = self.data.loc[begin_ts:end_ts]

        l = sliced[['low']].resample(granularity).min()
        h = sliced[['high']].resample(granularity).max()
        o = sliced[['open']].resample(granularity).first()
        c = sliced[['close']].resample(granularity).last()
        v = sliced[['volume']].resample(granularity).sum()

        result = pd.concat([l, h, o, c, v], axis=1)

        return SimCandles(data=result)
