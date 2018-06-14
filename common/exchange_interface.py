from abc import ABCMeta, abstractmethod
from enum import Enum
from pandas import Timestamp, DataFrame

from common.orders import Orders
from common.position import Position


class ExchangeInterface(metaclass=ABCMeta):
    def __init__(self):
        pass

    @abstractmethod
    def get_candles1m(self) -> DataFrame:
        raise AttributeError("interface class")

    @abstractmethod
    def post_orders(self, orders) -> bool:
        """
        :param orders:
        :return: True if any order was rejected
        """
        raise AttributeError("interface class")

    @abstractmethod
    def current_time(self) -> Timestamp:
        raise AttributeError("interface class")

    @abstractmethod
    def current_price(self) -> float:
        raise AttributeError("interface class")

    @abstractmethod
    def get_position(self, symbol: Enum) -> Position:
        raise AttributeError("interface class")

    @abstractmethod
    def get_closed_positions(self, symbol: Enum) -> Position:
        raise AttributeError("interface class")

    @abstractmethod
    def set_leverage(self, symbol: Enum, value: float) -> bool:
        """
        :param symbol
        :param value Leverage value. Send a number between 0.01 and 100 to enable isolated margin with a fixed leverage.
               Send 0 to enable cross margin.
        :return: True if succeeded
        """
        raise AttributeError("interface class")

    @abstractmethod
    def cancel_orders(self, orders: Orders, drop_canceled=True):
        raise AttributeError("interface class")

