import pandas as pd
from pandas import Series, DataFrame, Timestamp
from typing import Union


class Candles:
    def __init__(self, filename=None, data=None):
        self.data = self.to_ohlcv(data, filename)

    @staticmethod
    def to_ohlcv(data: Union[pd.DataFrame, list] = None, filename=None) -> pd.DataFrame:

        if filename is not None:
            if data is not None:
                raise ValueError("XOR(filename==None, data==None) should be True")
            data = pd.read_csv(filename)

        if isinstance(data, list):
            df = pd.DataFrame(data=data, dtype='float64')
        else:
            df = data
        if 'timestamp' in df.columns:
            df.set_index('timestamp', inplace=True)
        elif 'time' in df.columns:
            df.set_index('time', inplace=True)
        df = df[['open', 'high', 'low', 'close', 'volume']]
        df.index = pd.to_datetime(df.index)

        # dirty fix of bitmex bug
        idx = df['low'] > df['open']
        c = df.copy()
        c['low'][idx] = df['open']
        idx = df['high'] < df['open']
        c['high'][idx] = df['open']

        return c

    def _fix_bitmex_bug(self):
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

