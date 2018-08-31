from abc import ABCMeta, abstractmethod
from typing import List, Union

import pandas as pd

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
    def get_opened_orders(self, symbol: Symbol) -> OrderContainerType:
        raise AttributeError("interface class")

    @abstractmethod
    def send_orders(self, orders: List[OrderCommon]):
        """
        Send orders
        :param orders:
        NOTE: it changes orders id
        """
        raise AttributeError("interface class")

    @abstractmethod
    def cancel_orders(self, orders: Union[OrderContainerType, List[OrderCommon], List[str]]):
        """
        Send cancel request.
        :param orders: Dict of id -> order
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
    def get_pnl_history(self, symbol: Symbol) -> List[float]:
        raise AttributeError("interface class")

    @abstractmethod
    def set_leverage(self, symbol: Symbol, leverage: float):
        """
        :param symbol
        :param value Leverage value. Send a number between 0.01 and 100 to enable isolated margin with a fixed leverage.
               Send 0 to enable cross margin.
        """
        raise AttributeError("interface class")

    @abstractmethod
    def get_balance_xbt(self) -> float:
        raise AttributeError("interface class")
