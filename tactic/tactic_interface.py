from abc import ABCMeta, abstractmethod

from common.fill import Fill
from common.order import OrderCommon
from api.symbol import Symbol
from api.exchange_interface import ExchangeInterface
import pandas as pd

from common.quote import Quote
from common.trade import Trade


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
    def handle_trade(self, trade: Trade) -> None:
        raise AttributeError("interface class")

    @abstractmethod
    def handle_quote(self, quote: Quote) -> None:
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
