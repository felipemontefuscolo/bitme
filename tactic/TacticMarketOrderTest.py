import logging
import time
import unittest

import pandas as pd

from api import ExchangeInterface, Symbol
from common import Fill, OrderCommon, OrderType, FillType
from common.quote import Quote
from common.trade import Trade
from tactic import TacticInterface

logger = logging.getLogger('root')

assertEqual = unittest.TestCase().assertEqual
assertFalse = unittest.TestCase().assertFalse


class TacticMarketOrderTest(TacticInterface):
    """
    Just send n market-buys followed buy n market-sells and check if everything is ok
    """
    symbol = Symbol.XBTUSD
    exchange = None
    qty = 5
    initial_pos = 0
    n_closed_positions = 0

    def __init__(self, n_trades, n_positions):
        """
        :param n_trades: num buys (and sells)
        :param n_positions: number of times it will open a position
        """
        self.n_trades = n_trades
        self.buy_id = [None] * n_trades
        self.sell_id = [None] * n_trades
        self.buy_leaves = [self.qty] * n_trades
        self.sell_leaves = [self.qty] * n_trades
        self.next_action = 0
        self.n_positions = n_positions

    def initialize(self, exchange: ExchangeInterface, preferences: dict) -> None:
        self.exchange = exchange
        pass

    def finalize(self) -> None:
        print('finalizing tactic')
        assertEqual(self.buy_leaves, [0] * self.n_trades)
        assertEqual(self.sell_leaves, [0] * self.n_trades)

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

        if self.n_trades > 0 and not self.buy_id[0] and self.n_positions > 0:
            logger.info("opening a position")
            self.initial_pos = self.exchange.get_position(self.symbol).current_qty

            self.buy_id[0] = self.gen_order_id()
            self.next_action = 1
            logger.info("sending buy order {}".format(self.buy_id[0]))
            self.exchange.send_orders([OrderCommon(symbol=self.symbol,
                                                   type=OrderType.Market,
                                                   client_id=self.buy_id[0],
                                                   signed_qty=+self.qty)])

        pass

    def buy_id_to_index(self, oid: str):
        return self.id_to_index(oid, self.buy_id)

    def sell_id_to_index(self, oid: str):
        return self.id_to_index(oid, self.sell_id)

    @staticmethod
    def id_to_index(oid: str, ids: list):
        for j in range(len(ids)):
            if ids[j] == oid:
                return j
        return -1

    def expected_position(self):
        n_buys = min(self.next_action, self.n_trades)
        n_sells = max(self.next_action - self.n_trades, 0)
        return (n_buys - n_sells) * self.qty + self.initial_pos

    def handle_fill(self, fill: Fill) -> None:

        if fill.side.lower()[0] == 'b':
            i = self.buy_id_to_index(fill.order_id)
            if i == -1:
                raise AttributeError("We didn't send order with id {}".format(fill.order_id))
            self.buy_leaves[i] -= fill.qty

        else:
            if fill.side.lower()[0] != 's':
                raise AttributeError('side should be buy or sell, got {}'.format(fill.side))

            i = self.sell_id_to_index(fill.order_id)
            if i == -1:
                raise AttributeError("We didn't send order with id {}".format(fill.order_id))
            self.sell_leaves[i] -= fill.qty

        if fill.fill_type == FillType.complete:
            # we need this little delays because it seems that bitmex takes a while to update the position
            time.sleep(.3)
            pos = self.exchange.get_position(self.symbol)
            assertEqual(pos.current_qty, self.expected_position(), "n_buys={}, n_sells={}, init_pos={}".format(
                min(self.next_action, self.n_trades), max(self.next_action - self.n_trades, 0), self.initial_pos
            ))

            if self.next_action < self.n_trades:

                client_id = self.gen_order_id()
                self.buy_id[self.next_action] = client_id
                logger.info("sending buy order {}".format(self.buy_id[self.next_action]))
                self.next_action += 1
                self.exchange.send_orders([OrderCommon(symbol=self.symbol,
                                                       type=OrderType.Market,
                                                       client_id=client_id,
                                                       signed_qty=+self.qty)])

            elif self.n_trades <= self.next_action < 2 * self.n_trades:

                client_id = self.gen_order_id()
                self.sell_id[self.next_action - self.n_trades] = client_id
                self.next_action += 1
                logger.info("sending sell order {}".format(self.sell_id))
                self.exchange.send_orders([OrderCommon(symbol=self.symbol,
                                                       type=OrderType.Market,
                                                       client_id=client_id,
                                                       signed_qty=-self.qty)])
            else:
                time.sleep(.3)
                logger.info("checking position, it should be = initial position ...")
                pos = self.exchange.get_position(self.symbol)
                assertEqual(pos.current_qty, self.initial_pos)
                self.n_closed_positions += 1

                pnls = self.exchange.get_pnl_history(self.symbol)
                if len(pnls) != self.n_closed_positions:
                    raise AttributeError(
                        "Expected to have {} closed position, got {}".format(self.n_closed_positions,len(pnls)))

                self.n_positions -= 1
                if self.n_positions > 0:
                    self.__init__(self.n_trades, self.n_positions)
                    self.handle_1m_candles(None)

        pass

    def handle_cancel(self, order: OrderCommon) -> None:
        raise AttributeError("Didn't expect to get here")

    @staticmethod
    def id() -> str:
        return "TMOT"
