from abc import ABCMeta, abstractmethod

import pandas as pd
from common.orders import Orders
from api.position_interface import PositionInterface
from api.symbol import Symbol


class ExchangeInterface(metaclass=ABCMeta):
    def __init__(self):
        pass

    @abstractmethod
    def get_candles1m(self) -> pd.DataFrame:
        raise AttributeError("interface class")

    @abstractmethod
    def post_orders(self, orders) -> bool:
        """
        :param orders:
        :return: True if any order was rejected
        """
        raise AttributeError("interface class")

    @abstractmethod
    def current_time(self) -> pd.Timestamp:
        raise AttributeError("interface class")

    @abstractmethod
    def get_tick_info(self, symbol: Symbol = None) -> dict:
        """
        :param symbol:
        :return: dict, example: {"buy": 6630.0, "last": 6633.0, "mid": 6630.0, "sell": 6630.5}
        """
        raise AttributeError("interface class")

    @abstractmethod
    def get_position(self, symbol: Symbol) -> PositionInterface:
        raise AttributeError("interface class")

    @abstractmethod
    def get_closed_positions(self, symbol: Symbol=None) -> PositionInterface:
        raise AttributeError("interface class")

    @abstractmethod
    def set_leverage(self, symbol: Symbol, value: float) -> bool:
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
