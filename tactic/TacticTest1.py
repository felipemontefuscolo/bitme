import logging
import time

import pandas as pd

from api import ExchangeInterface, Symbol
from common import Fill, OrderCommon, OrderType, FillType
from common.quote import Quote
from common.trade import Trade
from tactic import TacticInterface

logger = logging.getLogger('root')


class TacticTest1(TacticInterface):
    """
    Just send 1 market-buy followed buy 1 market-sell and check if everything is ok
    """
    symbol = Symbol.XBTUSD

    exchange = None

    buy_id = None
    sell_id = None

    qty = 12
    buy_leaves = qty
    sell_leaves = qty

    initial_pos = 0

    def initialize(self, exchange: ExchangeInterface, preferences: dict) -> None:
        self.exchange = exchange
        pass

    def finalize(self) -> None:
        print('finalizing tactic')
        assert self.buy_leaves == 0
        assert self.sell_leaves == 0

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

        if not self.buy_id:
            self.initial_pos = self.exchange.get_position(self.symbol).signed_qty

            self.buy_id = self.gen_order_id()
            logger.info("sending buy order {}".format(self.buy_id))
            self.exchange.send_orders([OrderCommon(symbol=self.symbol,
                                                   type=OrderType.Market,
                                                   client_id=self.buy_id,
                                                   signed_qty=+self.qty)])

        pass

    def handle_fill(self, fill: Fill) -> None:

        if fill.side.lower()[0] == 'b':
            logger.info('Filled buy order. Leaves qty: before={}, after={}'.format(self.buy_leaves,
                                                                                   self.buy_leaves - fill.qty))
            self.buy_leaves -= fill.qty
        else:
            if fill.side.lower()[0] != 's':
                raise AttributeError('side should be buy or sell, got {}'.format(fill.side))

            logger.info('Filled sell order. Leaves qty: before={}, after={}'.format(self.sell_leaves,
                                                                                    self.sell_leaves - fill.qty))
            self.sell_leaves -= fill.qty

        if fill.fill_type == FillType.complete:
            if self.sell_id is None:

                time.sleep(1)

                pos = self.exchange.get_position(self.symbol)
                if pos.signed_qty != self.qty + self.initial_pos:
                    raise AttributeError('current_pos={}, initial_pos={}, qty to fill={}'.format(pos.signed_qty,
                                                                                                 self.initial_pos,
                                                                                                 self.qty))

                self.sell_id = self.gen_order_id()
                logger.info("sending sell order {}".format(self.sell_id))
                self.exchange.send_orders([OrderCommon(symbol=self.symbol,
                                                       type=OrderType.Market,
                                                       client_id=self.gen_order_id(),
                                                       signed_qty=-self.qty)])
            else:
                time.sleep(.3)
                logger.info("checking position, it should be = initial position ...")
                pos = self.exchange.get_position(self.symbol)
                assert pos.signed_qty == self.initial_pos
                assert not pos.is_open

        pass

    def handle_cancel(self, order: OrderCommon) -> None:
        raise AttributeError("Didn't expect to get here")

    @staticmethod
    def id() -> str:
        return "TTest1"
