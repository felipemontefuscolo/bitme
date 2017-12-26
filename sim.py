# NOTES:
# timestamps are of type pd.Timestamp
# side are of type str ('buy' or 'sell')


import sys

from collections import defaultdict

import math
from enum import Enum

import utils
from orders import to_str, TWOPLACES, EIGHPLACES, OrderStatus, OrderSubmissionError, OrderType
from simcandles import SimCandles
from tactic_mm import *
from utils import sign


class Fill:
    def __init__(self, order, qty_filled, price_fill, fill_time):
        # type: (OrderCommon, float, float, pd.Timestamp) -> None
        self.order_id = order.id
        self.side = 'buy' if order.signed_qty > 0 else 'sell'
        self.qty = qty_filled  # signed
        self.price = price_fill
        self.order_type = order.type
        self.fill_time = fill_time

    def __repr__(self):
        return str(self.to_json())

    def __str__(self):
        return str(self.to_line())

    def to_json(self):
        params = {
            'time': str(self.fill_time),
            'order_id': self.order_id,
            'side': self.side,
            'price': to_str(self.price, TWOPLACES),  # USD
            'qty': to_str(self.qty, EIGHPLACES),  # BTC
            'type': self.order_type
        }
        return params

    def to_line(self):
        return ','.join([
            str(self.fill_time),
            str(self.order_id),
            str(self.side),
            str(to_str(self.price, TWOPLACES)),  # USD
            str(to_str(self.qty, EIGHPLACES)),  # BTC
            str(self.order_type)
        ])

    @staticmethod
    def get_header():
        return "time,order_id,side,price,qty,type"


class ExchangeCommon:
    def __init__(self):
        pass

    def get_candles1m(self):
        raise AttributeError("interface class")

    def send_orders(self, orders, tactic_id):
        raise AttributeError("interface class")

    def current_time(self):
        raise AttributeError("interface class")


#  only BTC is supported for now
class SimExchangeBitMex(ExchangeCommon):
    TAKER_FEE = 0.00075
    MAKER_FEE = -0.00025

    class Symbol(Enum):
        XBTUSD = 'XBTUSD'
        XBTZ17 = 'XBTZ17'
        __iter__ = Enum.__iter__

    def __init__(self, initial_balance, file_name, tactics):
        ExchangeCommon.__init__(self)
        self.xbt_initial_balance = initial_balance
        self.xbt_balance = initial_balance
        self.positions = dict(zip([i for i in self.Symbol], [0] * len(self.Symbol)))  # in dollars. 0 mean no position
        self.position_fills = dict(zip([i for i in self.Symbol], [[]] * len(self.Symbol)))  # fill for the opened
                                                                                             # positions only. Used to
                                                                                             # compute pnl
        self.position_liq_price = dict(zip([i for i in self.Symbol], [float('nan')] * len(self.Symbol)))  # liq. prices for opened positions
        self.closed_position = dict(zip([i for i in self.Symbol], [[]] * len(self.Symbol)))
        self.all_fills = []
        self.all_realized_pnls = []

        self.candles = SimCandles(file_name)  # all candles
        self.time_idx = 0

        self.tac_sent_orders = defaultdict(lambda: Orders())  # str(TacticID) -> Orders()
        self.tac_opened_orders = defaultdict(lambda: Orders())  # str(TacticID) -> Orders()

        self.fills_file = open('/Users/felipe/bitme/output.fills', 'w')
        self.orders_file = open('/Users/felipe/bitme/output.orders', 'w')
        self.fills_file.write(Fill.get_header() + '\n')
        self.orders_file.write(Orders().to_csv() + '\n')

        self.leverage = dict(zip([i for i in self.Symbol], [50.] * len(self.Symbol)))  # 1 means 1%

        self.tactics = tactics

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.fills_file.close()
        self.orders_file.close()

    def current_time(self):
        return self.candles.data.iloc[self.time_idx].name

    def is_position_open(self, symbol):
        return self.positions[symbol] != 0

    def set_leverage(self, symbol, value):
        # type: (SimExchangeBitMex.Symbol, float) -> bool
        """
        :param value Leverage value. Send a number between 0.01 and 100 to enable isolated margin with a fixed leverage.
               Send 0 to enable cross margin.
        :return: True if succeeded """
        self.leverage[symbol] = value
        return True

    def current_price(self):
        return self.candles.data.iloc[self.time_idx].close

    def get_candles1m(self):
        return SimCandles(data=self.candles.data.iloc[0:self.time_idx + 1])

    # for sims only
    def advance_time(self, print_progress=True):
        is_last_round = self._is_last_candle()
        if print_progress:
            sys.stdout.write(
                "progress: %d out of %d (%.4f%%)   \r" %
                (self.time_idx, self.candles.size(), 100 * float(self.time_idx) / self.candles.size()))
            sys.stdout.flush()
        for tactic in self.tactics:  # type: TaticInterface
            tactic.handle_candles(self, self.positions, self.xbt_balance)
        self._execute_all_orders()
        self.time_idx += 1
        if is_last_round:
            sys.stdout.write("progress: %d out of %d (%.4f%%)\n" % (self.time_idx, self.candles.size(), 100.))



    def is_open(self):
        return self.time_idx < len(self.candles.data)

    @staticmethod
    def _compute_xbt_cost(qty, price, margin, fee):
        return (1. + fee) * abs(qty) / (price * margin)

    @staticmethod
    def _max_qty_given_balance(balance, price, margin, fee):
        # NOTE: return a non signed quantity
        return balance * price * margin / (1. + fee)

    def send_orders(self, orders, tactic_id):
        # type: (Orders, str) -> bool
        # may change order status
        # return true if any submission failed
        contain_errors = False
        current_time = self.candles.data.iloc[self.time_idx].name
        self.tac_sent_orders[tactic_id].merge(orders)

        if self.time_idx == self.candles.size() - 1:
            #  this is the last candle, cancel all orders
            for o in orders:  # type: OrderCommon
                o.time_posted = current_time
                o.status = OrderStatus.canceled
                o.status_msg = OrderSubmissionError.end_of_sim
            return True

        # discard bad orders
        current_price = self.current_price()
        for o in orders:  # type: OrderCommon
            if o.order_type is OrderType.limit:
                if (o.is_sell() and o.price < current_price) or (o.is_buy() and o.price > current_price):
                    o.status = OrderStatus.canceled
                    o.status_msg = OrderSubmissionError.invalid_price
                    contain_errors = True

                # set fee as 0 to have a pessimist cost
                cost = self._compute_xbt_cost(o.signed_qty, o.price, self.leverage[o.symbol], 0.)
                if self.xbt_balance < cost:
                    o.status = OrderStatus.canceled
                    o.status_msg = OrderSubmissionError.insufficient_funds
                    contain_errors = True

        for o in orders:
            o.time_posted = current_time
            if o.status is OrderStatus.opened:
                self.tac_opened_orders[tactic_id].add(o)

        return contain_errors

    def close_position_after_fill(self, symbol):
        assert self.positions[symbol] == 0
        fills = self.position_fills[symbol]
        self.all_fills += fills
        buy_qty = 0.
        buy_vol = 0.
        sell_qty = 0.
        sell_vol = 0.
        for fill in fills:  # type:
            if fill.qty > 0:
                buy_qty += fill.qty
                buy_vol += fill.qty * fill.price
            else:
                sell_qty += fill.qty
                sell_vol += fill.qty * fill.price
        pnl_xbt = (buy_qty / buy_vol - sell_qty / sell_vol) * (buy_qty - sell_qty) * self.leverage[symbol]
        self.all_realized_pnls += [pnl_xbt]
        self.xbt_balance += pnl_xbt
        self.all_fills += fills
        self.position_fills[symbol] = dict(zip([i for i in self.Symbol], [[]] * len(self.Symbol)))
        self.position_liq_price = dict(zip([i for i in self.Symbol], [float('nan')] * len(self.Symbol)))  # liq. prices for opened positions
        current_time = self.candles.data.iloc[self.time_idx].name
        self.closed_position[symbol] += [(current_time, pnl_xbt)]

    def _estimate_current_price(self, current_candle):
        # type: (pd.Series) -> float
        high = current_candle.high
        low = current_candle.low
        close_p = current_candle.close
        open_p = current_candle.open
        return (3 * open_p + 2. * (low + high) + close_p) / 8.

    def _average_entry_price(self, symbol):
        buy_qty = 0.
        buy_vol = 0.
        for fill in self.position_fills[symbol]:  # type:
            if fill.qty > 0:
                buy_qty += fill.qty
                buy_vol += fill.qty * fill.price
        return buy_vol / buy_qty

    def _execute_all_orders(self):
        current_candle = self.candles.data.iloc[self.time_idx]  # type: pd.Series

        for tactic_id, orders in self.tac_opened_orders.iteritems():
            for o in orders:  # type: OrderCommon
                self._execute_order(order=o, current_candle=current_candle)

    def _execute_order(self, order, current_candle):
        current_time = current_candle.name  # pd.Timestamp
        high = current_candle.high
        low = current_candle.low
        volume = current_candle.volume

        price_fill = self._estimate_current_price(current_candle)

        qty_fill = None
        cost_xbt = float('nan')
        if order.order_type is OrderType.market:  # fills immediately
            cost_xbt = self._compute_xbt_cost(price_fill, order.signed_qty, self.leverage[order.symbol], self.TAKER_FEE)
            if cost_xbt > self.xbt_balance:
                qty_fill = self.xbt_balance / cost_xbt * order.signed_qty
                cost_xbt = self.xbt_balance
            else:
                qty_fill = order.signed_qty
        elif order.order_type is OrderType.limit:
            cost_xbt = self._compute_xbt_cost(order.price, order.signed_qty, self.leverage[order.symbol], self.MAKER_FEE)
            if high == low:
                if (order.is_sell() and order.price < high) or (order.is_buy() and order.price > low):
                    qty_fill = 0.5 * volume
            else:
                if order.is_sell() and order.price < high:
                    qty_fill = -((high - order.price) / (high - low)) * volume
                elif order.is_buy() and order.price > low:
                    qty_fill = +((low - order.price) / (low - high)) * volume

        # common part between market and limit
        if qty_fill is not None:
            order.fill(qty_fill)
            fill = Fill(order=order, qty_filled=qty_fill, price_fill=price_fill, fill_time=current_time)
            self.position_fills[order.symbol] += [fill]
            if self.positions[order.symbol] == 0 or utils.sign(qty_fill) == utils.sign(self.positions[order.symbol]):
                self.xbt_balance -= cost_xbt
                self.position_liq_price[order.symbol] = \
                    self._average_entry_price(order.symbol) * self.leverage[order.symbol] / \
                    (self.leverage[order.symbol] + utils.sign(qty_fill) * 0.47)

            self.positions[order.symbol] += qty_fill
            if self.positions[order.symbol] == 0:
                # closing position
                self.close_position_after_fill(order.symbol)

    def _execute_liquidation(self):
        current_candle = self.candles.data.iloc[self.time_idx]
        current_price = self._estimate_current_price(current_candle)
        for symbol in self.Symbol:
            liq_price = self.position_liq_price[symbol]
            if (current_price < liq_price and self.positions[symbol] > 0) or \
                    (current_price > liq_price and self.positions[symbol] < 0) or \
                    self._is_last_candle():
                order = OrderCommon(symbol=symbol, signed_qty=self.positions[symbol], order_type=OrderType.market)
                self._execute_order(order, current_candle)

    def _is_last_candle(self):
        return self.time_idx == len(self.candles.data) - 1

    def print_summary(self):
        print("position btc = " + str(self.xbt_balance))
        print("position usd = " + str(self.positions))
        print("close price = " + str(self.candles.data.iloc[-1].close))
        #print("optimist realized profit = " + str(position_usd + position_btc * close_p - initial_position_usd))


def main():
    print("starting sim")
    # candles = Candles.fromfilename('/Users/felipe/bitme/data/test')

    # file_name = '/Users/felipe/bitme/data/data1s.csv'
    file_name = '/Users/felipe/bitme/data/bitmex_1day.csv'
    # file_name = '/Users/felipe/bitme/data/test'
    tactics = [TacticForBitMex2(SimExchangeBitMex.Symbol.XBTUSD)]

    with SimExchangeBitMex(0.2, file_name, tactics) as exchange:

        while exchange.is_open():

            exchange.advance_time(print_progress=True)

        exchange.print_summary()

    return 0


if __name__ == "__main__":
    sys.exit(main())
