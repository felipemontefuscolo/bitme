from common import Symbol
import pandas as pd


class Quote:
    def __init__(self,
                 symbol: Symbol,
                 timestamp: pd.Timestamp = None,
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
        return Quote(symbol=Symbol[raw['symbol']],
                     timestamp=pd.Timestamp(raw['timestamp']),
                     bid_size=raw['bidSize'],
                     bid_price=raw['bidPrice'],
                     ask_size=raw['askSize'],
                     ask_price=raw['askPrice'])

    def update_from_bitmex(self, raw: dict):
        assert self.symbol.name == raw['symbol']
        self.timestamp = pd.Timestamp(raw['timestamp'])
        self.bid_size = raw['bidSize']
        self.bid_price = raw['bidPrice']
        self.ask_size = raw['askSize']
        self.ask_price = raw['askPrice']
        return self

    def w_mid(self):
        return (self.bid_size * self.bid_price + self.ask_size * self.ask_price) / (self.bid_size + self.ask_size)