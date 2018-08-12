import pandas as pd
import time

from api import ExchangeInterface, Symbol
from common import Fill, OrderCommon, OrderType, FillType
from tactic import TacticInterface


class TacticTest1(TacticInterface):
    exchange = None
    num_subs = 0

    def init(self, exchange: ExchangeInterface, preferences: dict) -> None:
        self.exchange = exchange
        pass

    def get_symbol(self) -> Symbol:
        return Symbol.XBTUSD
        pass

    def handle_1m_candles(self, candles1m: pd.DataFrame) -> None:
        if self.num_subs == 0:
            self.num_subs += 1
            orders = self.exchange.post_orders([OrderCommon(symbol=Symbol.XBTUSD,
                                                            type=OrderType.Market,
                                                            tactic=self,
                                                            signed_qty=+12)])


        pass

    def handle_fill(self, fill: Fill) -> None:

        if self.num_subs == 1 and fill.fill_type == FillType.complete:
            self.num_subs += 1
            time.sleep(5)
            orders = self.exchange.post_orders([OrderCommon(symbol=Symbol.XBTUSD,
                                                            type=OrderType.Market,
                                                            tactic=self,
                                                            signed_qty=-12)])

        pass

    def handle_cancel(self, order: OrderCommon) -> None:
        pass

    @staticmethod
    def id() -> str:
        return "TTest1"
