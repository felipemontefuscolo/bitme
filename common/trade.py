from enum import Enum

import pandas as pd

from api import Symbol


class TickDirection(Enum):
    ZeroPlusTick = 'ZeroPlusTick'
    PlusTick = 'PlusTick'
    ZeroMinusTick = 'ZeroMinusTick'
    MinusTick = 'MinusTick'


class Trade:
    def __init__(self,
                 symbol: Symbol,
                 timestamp: pd.Timestamp = None,
                 side: int = None,
                 price: float = None,
                 size: float = None,
                 tick_direction: TickDirection = None):
        self.timestamp = timestamp
        self.symbol = symbol
        self.side = side
        self.price = price
        self.size = size
        self.tick_direction = tick_direction

    @staticmethod
    def from_raw(raw: dict):
        return Trade(timestamp=pd.Timestamp(raw['timestamp']),
                     symbol=Symbol[raw['symbol']],
                     side=+1 if raw['side'][0] == 'B' else -1,
                     price=raw['price'],
                     size=raw['size'],
                     tick_direction=TickDirection[raw['tickDirection']])

    def update_from_bitmex(self, raw):
        assert self.symbol.name == raw['symbol']
        self.timestamp = pd.Timestamp(raw['timestamp'])
        self.side = +1 if raw['side'][0] == 'B' else -1
        self.price = raw['price']
        self.size = raw['size']
        self.tick_direction = TickDirection[raw['tickDirection']]
        return self

    def __str__(self):
        return str(self.__dict__)