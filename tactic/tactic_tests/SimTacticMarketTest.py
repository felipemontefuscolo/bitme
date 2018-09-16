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


class SimTacticMarketTest(TacticInterface):
    """
    Market order test; it is only for SIM
    """
    symbol = Symbol.XBTUSD
    exchange = None
    initial_pos = 0
    n_closed_positions = 0
    candle_last_ts = None

    def __init__(self):
        """
        """
        self.orders_to_send = []  # list of list
        self.expected_position = []
        self.current_step = -1
        self.total_filled = 0
        self.cycle_num = 0
        self.trigger = [True, False, False]  # candle, trade, quote
        self.name_to_step = {'a': 0, 'b': 1, 'c': 1, 'd': 2, 'e': 3}

        self.n_trades_seen = 0
        self.n_quotes_seen = 0

    def initialize(self, exchange: ExchangeInterface, preferences: dict) -> None:
        self.exchange = exchange

        qty_pref = '/' + self.id() + '/qty_to_trade'
        if qty_pref not in preferences:
            raise ValueError('Pref "{}" must be specified'.format(qty_pref))
        self.qty = int(preferences[qty_pref])

        pass

    def gen_my_id(self, order_name):
        id = TacticInterface.gen_order_id(self)
        parts = id.split('_')
        return '{}_{}_{}'.format(parts[0], str(order_name), parts[1])

    def _create_orders(self):
        self.orders_to_send = []
        self.expected_position = []

        # step 0
        self.orders_to_send.append([OrderCommon(symbol=self.symbol,
                                                type=OrderType.Market,
                                                client_id=self.gen_my_id('a'),
                                                signed_qty=-self.qty)])
        self.expected_position.append(-self.qty)

        # step 1
        # sending two orders at same time to check if we get two different fills
        self.orders_to_send.append([OrderCommon(symbol=self.symbol,
                                                type=OrderType.Market,
                                                client_id=self.gen_my_id('b'),
                                                signed_qty=-self.qty),
                                    OrderCommon(symbol=self.symbol,
                                                type=OrderType.Market,
                                                client_id=self.gen_my_id('c'),
                                                signed_qty=-self.qty)])
        self.expected_position.append(-3 * self.qty)

        # step 2
        self.orders_to_send.append([OrderCommon(symbol=self.symbol,
                                                type=OrderType.Market,
                                                client_id=self.gen_my_id('d'),
                                                signed_qty=+4 * self.qty)])
        self.expected_position.append(+self.qty)

        # step 3
        self.orders_to_send.append([OrderCommon(symbol=self.symbol,
                                               type=OrderType.Market,
                                               client_id=self.gen_my_id('e'),
                                               signed_qty=-self.qty)])
        self.expected_position.append(0)

    def finalize(self) -> None:
        assert self.n_trades_seen > 0
        assert self.n_quotes_seen > 0
        pass

    def handle_trade(self, trade: Trade) -> None:
        self.n_trades_seen += 1
        if self.trigger[1]:
            self._start_cycle()
        pass

    def handle_quote(self, quote: Quote) -> None:
        self.n_quotes_seen += 1
        if self.trigger[2]:
            self._start_cycle()
        pass

    def handle_liquidation(self, pnl: float):
        raise AttributeError("Didn't expect to get here")

    def get_symbol(self) -> Symbol:
        return self.symbol
        pass

    def handle_1m_candles(self, candles1m: pd.DataFrame) -> None:
        if not self.candle_last_ts:
            self.candle_last_ts = candles1m.index[-1]
        elif not isinstance(candles1m, str):
            assert self.candle_last_ts < candles1m.index[-1]

        if self.trigger[0]:
            self._start_cycle()

        pass

    def _get_order(self, fill: Fill):
        for o_list in self.orders_to_send:  # type: list
            for o in o_list:
                if o.client_id == fill.order_id:  # type: OrderCommon
                    return o
        raise AttributeError('Could not find order for fill {}'.format(fill))

    def _get_step(self, order):
        return self.name_to_step[order.client_id.split('_')[1]]

    def handle_fill(self, fill: Fill) -> None:
        self.total_filled += fill.qty
        order = self._get_order(fill)
        order_name = order.client_id.split('_')[1]

        if fill.fill_type == FillType.complete:
            # time.sleep(.3)
            pos = self.exchange.get_position(self.symbol)
            step_from_order = self._get_step(self._get_order(fill))
            assertEqual(self.current_step, step_from_order)

            assertEqual(self.expected_position[self.current_step], pos.signed_qty, "at step {}".format(self.current_step))

            if step_from_order + 1 == 4:  # finish cycle
                self.current_step = -1
                self.cycle_num += 1
                assertEqual(self.cycle_num * self.qty * 8, self.total_filled)

                self.trigger = self.trigger[1:] + self.trigger[:1]
                if self.trigger[0]:
                    self.trigger = self.trigger[1:] + self.trigger[:1]

            elif order_name != 'b':
                self.current_step += 1
                self.exchange.send_orders(self.orders_to_send[self.current_step])
        pass

    def handle_cancel(self, order: OrderCommon) -> None:
        assertEqual(OrderCancelReason.end_of_sim, order.status_msg)

    def _start_cycle(self):
        if self.current_step == -1:
            self.current_step = 0
            self._create_orders()
            self.exchange.send_orders(self.orders_to_send[0])  # TODO: fix this

    @staticmethod
    def id() -> str:
        return "SimTMT"
