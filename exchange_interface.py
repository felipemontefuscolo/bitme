from enum import Enum
from pandas import Timestamp

from orders import Orders
from position import Position
from simcandles import SimCandles


class ExchangeCommon:
    def __init__(self):
        pass

    def get_candles1m(self):
        # type: (None) -> SimCandles
        raise AttributeError("interface class")

    def post_orders(self, orders):
        # type: (Orders) -> bool
        """
        :param orders:
        :return: True if any order was rejected
        """
        raise AttributeError("interface class")

    def current_time(self):
        # type: () -> Timestamp
        raise AttributeError("interface class")

    def current_price(self):
        # type: () -> float
        raise AttributeError("interface class")

    def get_position(self, symbol):
        # type: (Enum) -> Position
        raise AttributeError("interface class")

    def get_closed_positions(self, symbol):
        # type: (Enum) -> list(Position)
        raise AttributeError("interface class")

    def set_leverage(self, symbol, value):
        # type: (Enum, float) -> bool
        """
        :param symbol
        :param value Leverage value. Send a number between 0.01 and 100 to enable isolated margin with a fixed leverage.
               Send 0 to enable cross margin.
        :return: True if succeeded
        """
        raise AttributeError("interface class")

    def cancel_orders(self, orders, drop_canceled=True):
        # type: (Orders, bool) -> None
        raise AttributeError("interface class")

