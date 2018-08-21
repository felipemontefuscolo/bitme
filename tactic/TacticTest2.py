import pandas as pd

from api import ExchangeInterface, Symbol
from common import Fill, OrderCommon, OrderType
from tactic import TacticInterface


class TacticTest2(TacticInterface):
    exchange = None
    num_subs = 0

    got_the_cancel = False

    def init(self, exchange: ExchangeInterface, preferences: dict) -> None:
        self.exchange = exchange
        pass

    def get_symbol(self) -> Symbol:
        return Symbol.XBTUSD
        pass

    def handle_1m_candles(self, candles1m: pd.DataFrame) -> None:
        if self.num_subs == 0:
            self.num_subs += 1
            quote = self.exchange.get_quote(self.get_symbol())
            price = (quote.bid_price + quote.ask_price) / 2.
            # this should be rejected
            orders = self.exchange.send_orders([OrderCommon(symbol=Symbol.XBTUSD,
                                                            price=2*price,
                                                            type=OrderType.Limit,
                                                            tactic=self,
                                                            signed_qty=+12)])

        pass

    def handle_fill(self, fill: Fill) -> None:
        raise AttributeError('The only order submitted should never be filled')

        pass

    def handle_cancel(self, order: OrderCommon) -> None:
        self.got_the_cancel = True
        pass

    @staticmethod
    def id() -> str:
        return "TTest2"
