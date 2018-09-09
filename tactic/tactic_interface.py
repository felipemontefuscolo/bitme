import base64
import uuid
from abc import ABCMeta, abstractmethod

from common.fill import Fill
from common.order import OrderCommon
from api.symbol import Symbol
from api.exchange_interface import ExchangeInterface
import pandas as pd

from common.quote import Quote
from common.trade import Trade


class TacticInterface(metaclass=ABCMeta):
    ord_num = -1
    is_live = True

    def __init__(self):
        super().__init__()

    @abstractmethod
    def initialize(self, exchange: ExchangeInterface, preferences: dict) -> None:
        raise AttributeError("interface class")

    @abstractmethod
    def finalize(self) -> None:
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
    def handle_1m_candles(self, candles: pd.DataFrame) -> None:
        raise AttributeError("interface class")

    @abstractmethod
    def handle_cancel(self, order: OrderCommon) -> None:
        raise AttributeError("interface class")

    @abstractmethod
    def handle_liquidation(self, pnl: float):
        raise AttributeError("interface class")

    @abstractmethod
    def id(self) -> str:
        """
        IMPORTANT: can not contain underscores!
        :return:
        """
        raise AttributeError("interface class")

    def gen_order_id(self) -> str:
        if self.is_live:
            return "{}_{}".format(self.id(),
                                  base64.b64encode(uuid.uuid4().bytes).decode('utf8').rstrip('=\n'))
        self.ord_num += 1
        return "{}_{}".format(self.id(), self.ord_num)

    def does_own_order(self, order: OrderCommon):
        my_id = self.id()
        if not order.client_id:
            return False
        return order.client_id[:len(my_id)] == my_id
