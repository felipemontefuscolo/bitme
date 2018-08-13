import math
import pandas as pd

from api import ExchangeInterface, Symbol
from common import Fill, OrderCommon, OrderType
from tactic import TacticInterface


class TacticTest3(TacticInterface):
    exchange = None
    num_subs = 0
    my_orders = []
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
            price = self.exchange.get_tick_info()['buy']
            # this should be rejected
            self.my_orders = self.exchange.post_orders([OrderCommon(symbol=Symbol.XBTUSD,
                                                                    price=math.floor(price / 2),
                                                                    type=OrderType.Limit,
                                                                    tactic=self,
                                                                    signed_qty=+12)])
        else:
            r = self.exchange.cancel_orders(self.my_orders)
            assert r

        pass

    def handle_fill(self, fill: Fill) -> None:
        raise AttributeError('The only order submitted should never be filled')

        pass

    def handle_cancel(self, order: OrderCommon) -> None:
        self.got_the_cancel = True
        pass

    @staticmethod
    def id() -> str:
        return "TTest3"