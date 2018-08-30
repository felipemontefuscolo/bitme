import logging
import time
import unittest

import pandas as pd

from api import ExchangeInterface, Symbol
from common import Fill, OrderCommon, OrderType, FillType, OrderContainerType
from common.quote import Quote
from common.trade import Trade
from tactic import TacticInterface

logger = logging.getLogger('root')

assertEqual = unittest.TestCase().assertEqual
assertFalse = unittest.TestCase().assertFalse


class TacticLimitOrderTest(TacticInterface):
    symbol = Symbol.XBTUSD
    exchange = None
    qty = 123
    initial_pos = 0

    def __init__(self):
        logger.info("Starting {}".format(self.id()))
        self.sell_id = None

    def initialize(self, exchange: ExchangeInterface, preferences: dict) -> None:
        logger.info("Initializing {}".format(self.id()))
        self.exchange = exchange
        pass

    def finalize(self) -> None:
        pass

    def handle_trade(self, trade: Trade) -> None:
        pass

    def handle_quote(self, quote: Quote) -> None:
        pass

    def handle_liquidation(self, pnl: float):
        raise AttributeError("Didn't expect to get here")

    def get_symbol(self) -> Symbol:
        return self.symbol
        pass

    def handle_1m_candles(self, candles1m: pd.DataFrame) -> None:

        if not self.sell_id:
            # wait for quotes
            time.sleep(0.1)

            price = self.exchange.get_quote(self.symbol).ask_price
            if not price:
                raise AttributeError("We should have quote by now")

            orders = [
                # First order should be accepted
                OrderCommon(symbol=self.symbol,
                            type=OrderType.Limit,
                            client_id=self.gen_order_id(),
                            signed_qty=-self.qty,
                            price=price + 1000),

                # First order should be cancelled because we will try to sell bellow the bid
                # Note that our Limit orders should be post-only, i.e., ParticipateDoNotInitiate
                OrderCommon(symbol=self.symbol,
                            type=OrderType.Limit,
                            client_id=self.gen_order_id(),
                            signed_qty=-self.qty,
                            price=max(price - 500, 1)),
            ]

            self.sell_id = [o.client_id for o in orders]

            self.exchange.send_orders(orders)

        pass

    def handle_fill(self, fill: Fill) -> None:
        raise AttributeError("Didn't expect to get here")

    def handle_cancel(self, order: OrderCommon) -> None:
        if order.client_id != self.sell_id[1]:
            if order.client_id == self.sell_id[0]:
                raise AttributeError("Tactics should not need to handle their own cancels")
            raise AttributeError("Expecting to get id {}, got {} instead".format(self.sell_id[1], order.client_id))

        opened_orders = self.exchange.get_opened_orders(self.symbol)  # type: OrderContainerType
        if len(opened_orders) != 1:
            raise AttributeError("Expected to have exactly 1 order opened, but got ids: {}".format(opened_orders.keys()))

        try:
            opened_orders[self.sell_id[0]]
        except KeyError:
            raise AttributeError("Expected to have order {} opened".format(self.sell_id[0]))

        logger.info("order {} cancelled, now trying to cancel {}".format(self.sell_id[1], self.sell_id[0]))
        self.exchange.cancel_orders([self.sell_id[0]])
        time.sleep(1)

        opened_orders = self.exchange.get_opened_orders(self.symbol)

        if len(opened_orders) != 0:
            raise AttributeError("not expeting to have order opened")

    @staticmethod
    def id() -> str:
        return "TLOT"
