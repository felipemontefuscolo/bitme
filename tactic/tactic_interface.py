from abc import ABCMeta, abstractmethod

from common.fill import Fill
from common.order import OrderCommon
from api.symbol import Symbol
from api.exchange_interface import ExchangeInterface


class TacticInterface(metaclass=ABCMeta):
    def __init__(self):
        pass

    @abstractmethod
    def init(self, exchange: ExchangeInterface, preferences: dict) -> None:
        raise AttributeError("interface class")

    @abstractmethod
    def get_symbol(self) -> Symbol:
        raise AttributeError("interface class")

    @abstractmethod
    def handle_candles(self, exchange: ExchangeInterface) -> None:
        raise AttributeError("interface class")

    @abstractmethod
    def handle_submission_error(self, failed_order: OrderCommon) -> None:
        raise AttributeError("interface class")

    @abstractmethod
    def handle_fill(self, exchange: ExchangeInterface, fill: Fill) -> None:
        raise AttributeError("interface class")

    @abstractmethod
    def handle_cancel(self, exchange: ExchangeInterface, order: OrderCommon) -> None:
        raise AttributeError("interface class")

    def id(self):
        # type: () -> str
        return self.__class__.__name__
