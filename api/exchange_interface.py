from abc import ABCMeta, abstractmethod

import pandas as pd
from typing import Iterable, List, Union

from api.position_interface import PositionInterface
from api.symbol import Symbol
from common.order import OrderCommon, OrderContainerType
from common.quote import Quote


class ExchangeInterface(metaclass=ABCMeta):
    def __init__(self):
        pass

    @abstractmethod
    def get_candles1m(self) -> pd.DataFrame:
        raise AttributeError("interface class")

    @abstractmethod
    def get_opened_orders(self, symbol=None) -> OrderContainerType:
        raise AttributeError("interface class")

    @abstractmethod
    def post_orders(self, orders: List[OrderCommon]) -> List[OrderCommon]:
        """
        :param orders:
        :return: list of orders that were successfully posted
        NOTE: it changes orders id
        """
        raise AttributeError("interface class")

    @abstractmethod
    def cancel_orders(self, orders: Union[OrderContainerType, List[OrderCommon], List[str]]) -> OrderContainerType:
        """
        :param orders: Dict of id -> order
        :return: list of cancelled orders
        """
        raise AttributeError("interface class")

    @abstractmethod
    def current_time(self) -> pd.Timestamp:
        raise AttributeError("interface class")

    @abstractmethod
    def get_quote(self, symbol: Symbol) -> Quote:
        raise AttributeError("interface class")

    @abstractmethod
    def get_position(self, symbol: Symbol) -> PositionInterface:
        raise AttributeError("interface class")

    @abstractmethod
    def get_closed_positions(self, symbol: Symbol = None) -> List[PositionInterface]:
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
