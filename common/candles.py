import pandas as pd
from pandas import Series, DataFrame, Timestamp


class Candles:
    def __init__(self, filename=None, data=None):
        # type: (str, DataFrame) -> None
        if filename is not None and data is None:
            timeparser = lambda s: pd.datetime.strptime(str(s), '%Y-%m-%dT%H:%M:%S')
            self.data = pd.DataFrame(  # is this conversion inefficient?
                pd.read_csv(filename, parse_dates=True, index_col='time', date_parser=timeparser))
        elif data is not None and filename is None:
            assert isinstance(data, DataFrame)
            self.data = data
        else:
            raise ValueError("XOR(filename==None, data==None) should be True")
        self.fix_bitmex_bug()

    def fix_bitmex_bug(self):
        idx = self.data['low'] > self.data['open']
        c = self.data.copy()
        c['low'][idx] = self.data['open']
        self.data = c

    def at(self, index):
        # type: (int) -> Series
        return self.data.iloc[index]

    def subset(self, idx_begin, idx_end):
        # type: (int, int) -> Candles
        return Candles(data=self.data.iloc[idx_begin:idx_end])

    def size(self):
        return self.data.shape[0]

    def to_csv(self):
        return self.data.to_csv()

    def last_timestamp(self):
        # type: () -> Timestamp
        return self.data.iloc[-1].name

    def last_trade_price(self):
        # type: () -> float
        return self.data.iloc[-1].close

    def sample_candles(self, granularity, begin_ts, end_ts):
        # type: (pd.Timedelta, Timestamp, Timestamp) -> Candles
        '''
        :param granularity: example: pd.Timedelta(hours=1)'
        :param begin_ts: example: Candles().iloc[0].name - pd.Timedelta(hours=1)
        :param end_ts: example: Candles().iloc[0].name
        :return: sampled Candles
        '''
        sliced = self.data.loc[begin_ts:end_ts]

        l = sliced[['low']].resample(granularity).min()
        h = sliced[['high']].resample(granularity).max()
        o = sliced[['open']].resample(granularity).first()
        c = sliced[['close']].resample(granularity).last()
        v = sliced[['volume']].resample(granularity).sum()

        result = pd.concat([l, h, o, c, v], axis=1)

        return Candles(data=result)

