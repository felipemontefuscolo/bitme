import pandas as pd
from typing import List

from api import ExchangeInterface, Symbol
from common import Fill, OrderCommon, Quote, Trade, OrderType, FillType
from tactic import TacticInterface, PositionSim
import math


class TacticMakerV1(TacticInterface):

    def __init__(self):
        self.exchange = None  # type: ExchangeInterface
        self.max_qty = 1000
        self.min_liq = 1000  # doesn't make much sense when we don't have the full book available
        self.alpha = 100  # qty / price
        self.leverage = 1
        self.symbol = Symbol.XBTUSD
        self.risk = 0.001  # percent

        # buy, sell
        self.buy_order = None  # type: OrderCommon
        self.sell_order = None  # type: OrderCommon
        # lower, upper
        self.last_quote = None  # type: Quote
        self.rest = 0  # type: int
        self.learning = 0  # unit of position

        # this is to keep track of avg_entry_price
        self.position = PositionSim(self.symbol, on_position_close=None)
        # this should be None when self.position is not open
        self.liq_price = None

        """
        price = fcst - pos / a
        
        buy = min( buy_best + 0.5,  floor(fcst - 0.25))
        sel = min( sel_best - 0.5,  ceil(fcst + 0.25))
        
        qty_buy = min(max_qty, round((price - buy) * a))
        qty_sel = min(max_qty, round((sel - price) * a))
        
        risk:
        liq_price = e / (1 + side * risk)
        """

    @staticmethod
    def _floor(x):
        return math.floor(2. * x) / 2

    @staticmethod
    def _ceil(x):
        return math.ceil(2. * x) / 2

    def initialize(self, exchange: ExchangeInterface, preferences: dict) -> None:
        self.exchange = exchange

        def pref(x):
            return '/{}/{}'.format(self.id(), x)

        if pref('max_qty') in preferences:
            self.max_qty = int(preferences[pref('max_qty')])
        if pref('min_liq') in preferences:
            self.min_liq = int(preferences[pref('min_liq')])
        if pref('alpha') in preferences:
            self.alpha = float(preferences[pref('alpha')])
        pass

    def finalize(self) -> None:
        pass

    def get_symbol(self) -> Symbol:
        return self.symbol

    def _create_orders(self, learning: float, fcst: float, quote: Quote):

        # TODO: improve to use full book information
        buy_best = quote.bid_price if quote.bid_size >= self.min_liq else quote.bid_price - 0.5
        sell_best = quote.ask_price if quote.ask_size >= self.min_liq else quote.ask_price + 0.5

        price = fcst - learning / self.alpha

        buy = min(buy_best + 0.5, self._floor(price - 0.25))
        sell = max(sell_best - 0.5, self._ceil(price + 0.25))

        qty_buy = min(self.max_qty, round((price - buy) * self.alpha))
        qty_sel = min(self.max_qty, round((sell - price) * self.alpha))

        if qty_buy >= 1:
            self.buy_order = OrderCommon(symbol=self.symbol,
                                         type=OrderType.Limit,
                                         client_id=self.gen_order_id(),
                                         signed_qty=qty_buy,
                                         price=buy)

        if qty_sel >= 1:
            self.sell_order = OrderCommon(symbol=self.symbol,
                                          type=OrderType.Limit,
                                          client_id=self.gen_order_id(),
                                          signed_qty=-qty_sel,
                                          price=sell)

    def _send_orders(self):
        # TODO: create a way to reuse open orders
        orders = []
        if self.buy_order:
            orders.append(self.buy_order)
        if self.sell_order:
            orders.append(self.sell_order)

        self.exchange.send_orders(orders)

    def _liq_pos(self, pos: float):
        self.exchange.send_orders([OrderCommon(symbol=self.symbol,
                                               type=OrderType.Market,
                                               client_id=self.gen_order_id(),
                                               signed_qty=-pos)])

    # override
    def gen_order_id(self, suffix=None) -> str:
        if suffix:
            return super().gen_order_id() + suffix
        else:
            return super().gen_order_id()

    def handle_trade(self, trade: Trade) -> None:
        if not self.last_quote:
            return
        if self.rest > 0:
            self.rest -= 1
            return

        if not self.buy_order and not self.sell_order:
            fcst = (self.last_quote.bid_price + self.last_quote.ask_price) * 0.5
            self._create_orders(learning=self.position.signed_qty, fcst=fcst, quote=self.last_quote)
            self._send_orders()
        else:
            side = self.position.side
            should_liq = self.liq_price is not None and trade.price * side < self.liq_price * side

            if should_liq:
                self._cancel_all()
                self.exchange.close_position(self.symbol)
                self.rest = 3

        pass

    def _cancel_all(self):
        self.buy_order = None
        self.sell_order = None
        self.exchange.cancel_all_orders(self.symbol)

    def handle_quote(self, quote: Quote) -> None:
        self.last_quote = quote

    def handle_fill(self, fill: Fill) -> None:
        if fill.side[0] == 'B':
            side = +1
        elif fill.side[0] == 'S':
            side = -1
        else:
            raise ValueError("Invalid fill side: {}".format(fill.side))

        signed_qty = fill.qty * side
        self.position.update(signed_qty=signed_qty,
                             price=fill.price,
                             leverage=self.leverage,
                             current_timestamp=fill.fill_time,
                             fee=0.)

        if self.position.is_open:
            self.liq_price = self.position.avg_entry_price / (1 + self.position.side * self.risk)
        else:
            self.liq_price = None

        if fill.fill_type == FillType.complete:
            if side > 0:
                self.buy_order = None
            else:
                self.sell_order = None
        return

    def handle_1m_candles(self, candles: pd.DataFrame) -> None:
        pass

    def handle_cancel(self, order: OrderCommon) -> None:
        self._cancel_all()
        self.rest = 3

    def handle_liquidation(self, pnl: float):
        raise AttributeError("This tactic should liquidate before bitmex liquidation")

    @staticmethod
    def id() -> str:
        return 'TMV1'
