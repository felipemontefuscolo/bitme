from pandas import Series


class Candles:
    def __init__(self):
        pass

    def at(self, index):
        # type: (int) -> Series
        raise AttributeError("interface class")

    def subset(self, idx_begin, idx_end):
        # type: (int, int) -> Candles
        raise AttributeError("interface class")

    def size(self):
        raise AttributeError("interface class")

    def to_csv(self):
        raise AttributeError("interface class")

    def last_timestamp(self):
        # type: () -> pd.Timestamp
        raise AttributeError("interface class")

    def last_price(self):
        # type: () -> float
        raise AttributeError("interface class")

    def sample_candles(self, granularity, begin_ts, end_ts):
        # type: (pd.Timedelta, pd.Timestamp, pd.Timestamp) -> Candles
        raise AttributeError("interface class")
