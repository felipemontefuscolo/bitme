from abc import ABCMeta, abstractmethod

from common.fill import Fill
from common.id_generator import IdGenerator
from common.order import OrderCommon
from api.symbol import Symbol
from api.exchange_interface import ExchangeInterface
import pandas as pd


class TacticInterface(IdGenerator, metaclass=ABCMeta):
    def __init__(self, use_uuid=False):
        super().__init__(use_uuid)

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

    def id(self) -> str:
        return self.__class__.__name__
