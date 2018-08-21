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
    def subscribe_to_trades(self, tactic):
        raise AttributeError("interface class")

    @abstractmethod
    def subscribe_to_quotes(self, tactic):
        raise AttributeError("interface class")

    @abstractmethod
    def get_candles1m(self) -> pd.DataFrame:
        raise AttributeError("interface class")

    @abstractmethod
    def get_opened_orders(self, symbol=None) -> OrderContainerType:
        raise AttributeError("interface class")

    @abstractmethod
    def send_orders(self, orders: List[OrderCommon]):
        """
        It does not send orders immediately. These orders goes to a queue. If a order is canceled, the method
        handle_cancel will be triggered.
        :param orders:
        :return: list of orders that were successfully posted
        NOTE: it changes orders id
        """
        raise AttributeError("interface class")

    @abstractmethod
    def cancel_orders(self, orders: Union[OrderContainerType, List[OrderCommon], List[str]]) -> OrderContainerType:
        """
        It does not cancel orders immediately. These orders goes to a queue.
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
