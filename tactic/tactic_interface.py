from abc import ABCMeta, abstractmethod

from common.fill import Fill
from common.order import OrderCommon
from api.symbol import Symbol
from api.exchange_interface import ExchangeInterface
import pandas as pd


class TacticInterface(metaclass=ABCMeta):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def init(self, exchange: ExchangeInterface, preferences: dict) -> None:
        raise AttributeError("interface class")

    @abstractmethod
    def get_symbol(self) -> Symbol:
        raise AttributeError("interface class")

    @abstractmethod
    def handle_1m_candles(self, candles1m: pd.DataFrame) -> None:
        raise AttributeError("interface class")

    @abstractmethod
    def handle_fill(self, fill: Fill) -> None:
        raise AttributeError("interface class")

    @abstractmethod
    def handle_cancel(self, order: OrderCommon) -> None:
        raise AttributeError("interface class")

    @abstractmethod
    def id(self) -> str:
        raise AttributeError("interface class")
