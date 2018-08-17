from common import Symbol
import pandas as pd


class Quote:
    def __init__(self,
                 timestamp: pd.Timestamp,
                 symbol: Symbol,
                 bid_size=None,
                 bid_price=None,
                 ask_size=None,
                 ask_price=None):
        self.timestamp = timestamp
        self.symbol = symbol
        self.bid_size = bid_size
        self.bid_price = bid_price
        self.ask_size = ask_size
        self.ask_price = ask_price

    @staticmethod
    def from_raw(raw: dict):
        return Quote(pd.Timestamp(raw['timestamp']),
                     Symbol[raw['symbol']],
                     raw['bidSize'],
                     raw['bidPrice'],
                     raw['askPrice'],
                     raw['askSize'])

    def update_from_bitmex(self, raw: dict):
        assert self.symbol.name == raw['symbol']
        self.timestamp = pd.Timestamp(raw['timestamp'])
        self.bid_size = raw['bidSize']
        self.bid_price = raw['bidPrice']
        self.ask_size = raw['askPrice']
        self.ask_price = raw['askSize']
        return self
