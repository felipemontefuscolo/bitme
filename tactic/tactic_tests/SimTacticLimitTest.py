import logging
import time
import unittest

import pandas as pd

from api import ExchangeInterface, Symbol
from common import Fill, OrderCommon, OrderType, FillType, OrderCancelReason
from common.quote import Quote
from common.trade import Trade
from tactic import TacticInterface

logger = logging.getLogger('root')

assertEqual = unittest.TestCase().assertEqual
assertFalse = unittest.TestCase().assertFalse
assertTrue = unittest.TestCase().assertTrue


class SimTacticLimitTest(TacticInterface):
    """
    Limit order test; it is only for SIM
    """
    symbol = Symbol.XBTUSD
    exchange = None

    def __init__(self):
        """
        """
        self.n_quotes_seen = 0
        self.qty = 1000
        self.period = 100  # trades
        self.tick = 0
        self.orders_to_send = []

    def initialize(self, exchange: ExchangeInterface, preferences: dict) -> None:
        self.exchange = exchange
        pass

    def _create_orders(self):
        quote = self.exchange.get_quote(self.symbol)
        bid = quote.bid_price
        ask = quote.ask_price

        # sending two orders at same time to check if we get two different fills
        self.orders_to_send = [OrderCommon(symbol=self.symbol,
                                           type=OrderType.Limit,
                                           price=bid - 0.5,
                                           client_id=self.gen_order_id(),
                                           signed_qty=+self.qty),
                               OrderCommon(symbol=self.symbol,
                                           type=OrderType.Limit,
                                           price=ask + 0.5,
                                           client_id=self.gen_order_id(),
                                           signed_qty=-self.qty)]

    def finalize(self) -> None:
        pass

    def handle_trade(self, trade: Trade) -> None:

        if self.tick % self.period == 0:
            opened_orders = self.exchange.get_opened_orders(self.symbol, self.id())
            if len(opened_orders) > 0:
                self.exchange.cancel_orders(opened_orders)
            self._create_orders()
            self.exchange.send_orders(self.orders_to_send)
            # print("SENT {} {} {}".format(self.exchange.current_time(), self.orders_to_send[0].price, self.orders_to_send[1].price))
        # elif self.orders_to_send:
        #     print('PRICE = {},  self.orders_to_send={}'.format(trade.price, [i for i in self.orders_to_send]))
        self.tick += 1


        pass

    def handle_quote(self, quote: Quote) -> None:
        pass

    def handle_liquidation(self, pnl: float):
        pass

    def get_symbol(self) -> Symbol:
        return self.symbol
        pass

    def handle_1m_candles(self, candles1m: pd.DataFrame) -> None:
        pass

    def handle_fill(self, fill: Fill) -> None:
        pass

    def handle_cancel(self, order: OrderCommon) -> None:
        print("CANCELED")
        if not (order.status_msg == OrderCancelReason.cross_during_post_only or
                order.status_msg == OrderCancelReason.end_of_sim):
            raise AttributeError("Unexpected order cancel reason: {}".format(order.status_msg))
        self.orders_to_send = [o for o in self.orders_to_send if o.client_id != order.client_id]

    @staticmethod
    def id() -> str:
        return "SimTLT"
