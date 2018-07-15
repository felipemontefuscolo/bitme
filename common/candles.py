from typing import Union

import pandas as pd

REQUIRED_COLUMNS = ['open', 'high', 'low', 'close', 'volume']


def create_df_for_candles():
    return pd.DataFrame(columns=REQUIRED_COLUMNS, dtype=float, index=pd.DatetimeIndex(data=[], name='timestamp'))


def fix_bitmex_bug(df: pd.DataFrame) -> pd.DataFrame:
    idx = df['low'] > df['open']
    c = df.copy()
    c['low'][idx] = df['open']
    idx = df['high'] < df['open']
    c['high'][idx] = df['open']

    return c


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
    df.index.name = 'timestamp'

    return fix_bitmex_bug(df)


def resample_candles(data: pd.DataFrame, granularity: pd.Timedelta, begin_ts: pd.Timestamp, end_ts: pd.Timestamp):
    """
    :param data: pd.DataFrame, assumes that it has columns low, high, open, close, volume
    :param granularity:
    :param begin_ts:
    :param end_ts:
    :return:
    """
    sliced = data.loc[begin_ts:end_ts]

    lo = sliced[['low']].resample(granularity).min()
    hi = sliced[['high']].resample(granularity).max()
    op = sliced[['open']].resample(granularity).first()
    cl = sliced[['close']].resample(granularity).last()
    vo = sliced[['volume']].resample(granularity).sum()

    result = pd.concat([lo, hi, op, cl, vo], axis=1)

    return result
