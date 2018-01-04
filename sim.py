# NOTES:
# timestamps are of type pd.Timestamp
# side are of type str ('buy' or 'sell')
import copy
import os
import sys
from collections import defaultdict

from enum import Enum
from numpy.core.umath import sign

from orders import to_str, TWOPLACES, OrderCancelReason, OrderStatus
from simcandles import SimCandles
from tactic_mm import *


class Fill:
    def __init__(self, order, qty_filled, price_fill, fill_time):
        # type: (OrderCommon, float, float, pd.Timestamp) -> None
        self.symbol = order.symbol
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
            'symbol': str(self.symbol.name),
            'order_id': self.order_id,
            'side': self.side,
            'price': to_str(self.price, TWOPLACES),  # USD
            'qty': str(int(self.qty)),  # USD
            'type': self.order_type.name
        }
        return params

    def to_line(self):
        return ','.join([
            str(self.fill_time.strftime('%Y-%m-%dT%H:%M:%S')),  # type: pd.Timestamp
            str(self.symbol.name),
            str(self.order_id),
            str(self.side),
            str(to_str(self.price, TWOPLACES)),  # USD
            str(int(self.qty)),  # USD
            str(self.order_type.name)
        ])

    @staticmethod
    def get_header():
        return "time,symbol,order_id,side,price,qty,type"


class ExchangeCommon:
    def __init__(self):
        pass

    def get_candles1m(self):
        raise AttributeError("interface class")

    def post_orders(self, orders):
        """
        :param orders:
        :return: True if any order was rejected
        """
        raise AttributeError("interface class")

    def current_time(self):
        raise AttributeError("interface class")


#  only BTC is supported for now
class SimExchangeBitMex(ExchangeCommon):
    FEE = {OrderType.limit: -0.00025, OrderType.market: 0.00075}

    class Symbol(Enum):
        XBTUSD = 'XBTUSD'
        XBTH18 = 'XBTH18'
        __iter__ = Enum.__iter__

    N_SYMBOLS = len(Symbol)
    SYMBOLS = list(Symbol)

    # reference: https://www.bitmex.com/app/riskLimits#instrument-risk-limits
    RISK_LIMITS = {Symbol.XBTUSD: 0.0015, Symbol.XBTH18: 0.0015}

    class Position:
        """
        Only isolated margin is supported (see isolated vs cross here: https://www.bitmex.com/app/isolatedMargin)
        It means that when a position is opened, a fixed amount is taken as collateral. Any gains are only credited
        after the position is closed.
        """

        def __init__(self):
            self.buy_qty = 0.
            self.buy_vol = 0.
            self.sell_qty = 0.  # always negative
            self.sell_vol = 0.
            self.realized_pnl = 0.
            self.liq_price = 0.
            self.close_ts = None
            self.side = None
            pass

        def close_position(self):
            assert self.is_closeable()
            pnl = self.realized_pnl
            self.__init__()
            return pnl

        def update(self, qty, price, multiplier, fee):
            if self.side is None:
                self.side = sign(qty)
            """
            should be updated on every fill
            """
            net_qty = self.net_qty()
            if net_qty * (qty + net_qty) < 0:  # changing position direction
                raise ValueError("provided qty changes position side. This case should be handled outside this method.")

            if qty > 0:
                self.buy_qty += qty
                self.buy_vol += qty * price * (1. + fee)
            else:
                self.sell_qty += qty
                self.sell_vol += qty * price * (1. - fee)
            self.liq_price = self.entry_price() * multiplier / (self.side * .75 + multiplier)
            self.realized_pnl = self.calc_realized_pnl(multiplier)

        def calc_realized_pnl(self, multiplier):
            # pnl = Contracts * Multiplier * (1/Entry Price - 1/Exit Price)

            return self.side * multiplier * min(abs(self.buy_qty), abs(self.sell_qty)) * \
                   (self.buy_qty/max(self.buy_vol, 1.e-8) - self.sell_qty/min(self.sell_vol, -1.e-8))

        def entry_price(self):
            # this is not exactly true, but it's a good approximation
            if self.side > 0:
                return self.buy_vol / self.buy_qty
            else:
                return self.sell_vol / self.sell_qty

        def net_qty(self):
            return self.buy_qty + self.sell_qty

        def position(self):
            return self.buy_qty + self.sell_qty

        def is_closeable(self):
            return abs(self.buy_qty + self.sell_qty) < 1.e-10

        def is_open(self):
            return self.side is not None

        def does_change_side(self, qty):
            net_qty = self.net_qty()
            return net_qty * (qty + net_qty) < 0

        def does_reduce_position(self, qty):
            return self.position_change(qty) < 0

        def position_change(self, signed_qty):
            # type: (float) -> float
            """
            :param signed_qty:
            :return: a positive number if signed_qty increase position (positively or negatively) and negative otherwise
            """
            return abs(self.buy_qty + self.sell_qty + signed_qty) - abs(self.buy_qty + self.sell_qty)

    def __init__(self, initial_balance, file_name, tactics):
        ExchangeCommon.__init__(self)
        self.xbt_initial_balance = initial_balance
        self.xbt_balance = initial_balance

        self.positions = defaultdict(self.Position)  # Symbol -> Position
        self.leverage = dict([(i, 100.) for i in self.SYMBOLS])  # 1 means 1%

        self.active_orders = Orders()

        # liq price for each position
        self.closed_positions_hist = defaultdict(list)  # Symbol -> list of Position
        self.fills_hist = []
        self.order_hist = Orders()

        self.candles = SimCandles(file_name)  # all candles
        self.time_idx = 0

        self.tactics = tactics

        self.n_cancels = defaultdict(int)
        self.n_liquidations = defaultdict(int)  # Symbol -> int

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

    def set_leverage(self, symbol, value):
        # type: (SimExchangeBitMex.Symbol, float) -> bool
        """
        :param symbol
        :param value Leverage value. Send a number between 0.01 and 100 to enable isolated margin with a fixed leverage.
               Send 0 to enable cross margin.
        :return: True if succeeded """
        self.leverage[symbol] = value
        return True

    def current_price(self):
        return self.candles.data.iloc[self.time_idx].close

    def current_time(self):
        return self.candles.data.iloc[self.time_idx].name

    def current_candle(self):
        return self.candles.data.iloc[self.time_idx]

    def get_candles1m(self):
        return SimCandles(data=self.candles.data.iloc[0:self.time_idx + 1])

    def is_last_candle(self):
        return self.time_idx == self.candles.size() - 1

    # for sims only
    def advance_time(self, print_progress=True):
        if print_progress:
            sys.stdout.write(
                "progress: %d out of %d (%.4f%%)   \r" %
                (self.time_idx, self.candles.size(), 100 * float(self.time_idx) / self.candles.size()))
            sys.stdout.flush()

        if self.is_last_candle():
            for symbol in self.SYMBOLS:
                self._execute_liquidation(symbol)
                self.time_idx += 1
                sys.stdout.write("progress: %d out of %d (%.4f%%)\n" % (self.time_idx, self.candles.size(), 100.))
                return
        else:
            for tactic in self.tactics:  # type: TaticInterface
                tactic.handle_candles(self, self.positions, self.xbt_balance)
            self.time_idx += 1
            self._execute_all_orders()
            current_price = self._estimate_price()
            for symbol in self.SYMBOLS:
                position = self.positions[symbol]
                if position.is_open():
                    side = position.side
                    liq_price = self.positions[symbol].liq_price
                    if (side > 0 and current_price < liq_price) or (side < 0 and current_price > liq_price):
                        self._execute_liquidation(symbol)

    def is_open(self):
        return self.time_idx < len(self.candles.data)

    def _execution_cost(self, order, qty_filled, price_filled, apply_fee=True):
        # type: (OrderCommon, float, float) -> float
        return (1. + self.FEE[order.type] * apply_fee) * abs(qty_filled) / (price_filled * self.leverage[order.symbol])

    def _execution_profit(self, order, qty_filled, price_filled):
        # type: (OrderCommon, float, float) -> float
        return (1. - self.FEE[order.type]) * abs(qty_filled) / (price_filled * self.leverage[order.symbol])

    def _limit_order_margin_cost_xbt(self, qty, price, symbol):
        # type: (float, float, self.Symbol) -> float
        """
        Simplified version of Bitmex cost. It assumes that abs(qty) < 100 XBT.
        Also, it ignores the mark price and risk price.
        """
        return abs(qty) / (price * self.leverage[symbol])

    def cancel_orders(self, orders, drop_canceled=True, status=OrderStatus.canceled, reason=OrderCancelReason.cancel_requested):
        for o in orders:
            self.active_orders[o.id].status = status
            self.active_orders[o.id].status_msg = reason
            self.n_cancels[reason.name] += 1
        if drop_canceled:
            orders.drop_closed_orders()
            self.active_orders.drop_closed_orders()  # in case orders does not refers to self.opened_orders

    @staticmethod
    def _reject_order(order, time_posted, reason):
        # type: (OrderCommon, pd.Timestamp, OrderCancelReason) -> None
        assert order.status is not OrderStatus.opened  # it can only reject order not yet posted
        order.time_posted = time_posted
        order.status = OrderStatus.canceled
        order.status_msg = reason

    def post_orders(self, orders):
        # type: (Orders, str) -> bool
        # may change order status
        # return true if any submission failed
        contain_errors = False
        current_time = self.current_time()
        self.order_hist.merge(orders)

        if self.time_idx == self.candles.size() - 1:
            #  this is the last candle, cancel all orders
            for o in orders:  # type: OrderCommon
                self._reject_order(o, current_time, OrderCancelReason.end_of_sim)
            return True

        # discard bad orders
        current_price = self.current_price()
        for o in orders:  # type: OrderCommon
            if o.type is OrderType.limit:
                crossed = (o.is_sell() and o.price < current_price) or (o.is_buy() and o.price > current_price)
                if crossed or o.price < 0:
                    # only post_only are supported, so don't let it cross
                    self._reject_order(o, current_time, OrderCancelReason.invalid_price)
                    contain_errors = True
                    continue
            elif o.type is not OrderType.market:
                raise ValueError("invalid order type %s" % str(o.type))

        for o in orders:  # type: OrderCommon
            o.time_posted = current_time
            if o.status == OrderStatus.opened:
                self.active_orders.add(o)

        return contain_errors

    def _estimate_price(self, current_candle=None):
        if current_candle is None:
            current_candle = self.current_candle()
        high = current_candle.high
        low = current_candle.low
        close_p = current_candle.close
        open_p = current_candle.open
        return (3 * open_p + 2. * (low + high) + close_p) / 8.

    def _execute_all_orders(self):
        assert self.time_idx < len(self.candles.data)
        for o in self.active_orders:  # type: OrderCommon
            self._execute_order(o)
        self.active_orders.drop_closed_orders()

    def _execute_order(self, order):
        # type: (OrderCommon) -> None
        assert self.time_idx < self.candles.size()
        current_candle = self.current_candle()
        current_time = current_candle.name  # pd.Timestamp
        high = current_candle.high
        low = current_candle.low
        open = current_candle.open
        close = current_candle.close

        position = self.positions[order.symbol]  # type: self.Position
        position_value = position.position()
        qty_fill = qty_to_close = outstanding_qty = None
        crossed = False

        if order.type is OrderType.market:
            crossed = True
            price_fill = self._estimate_price()
            qty_fill = order.signed_qty
        elif order.type is OrderType.limit:
            price_fill = order.price
            max_qty_fill = order.signed_qty - order.filled
            # clip fill
            if open <= order.price <= close:
                qty_fill = max_qty_fill
            elif high == low == order.price:
                qty_fill = round(0.5 * max_qty_fill)
            else:
                if low < order.price < high:
                    if order.is_sell():
                        factor = max((high - order.price) / (high - low), 0.50001)
                        assert factor >= 0
                    else:
                        factor = max((low - order.price) / (low - high), 0.50001)
                        assert factor >= 0
                    qty_fill = round(factor * max_qty_fill)
            if qty_fill is not None:
                crossed = True
        else:
            raise ValueError("order type " + str(order.type) + " not supported")

        if not crossed:
            return

        if position.does_change_side(qty_fill):
            qty_to_close = sign(qty_fill) * min(abs(position_value), abs(qty_fill))
            outstanding_qty = qty_fill - qty_to_close

        if order.fill(qty_fill) or order.type == OrderType.market:
            order.status = OrderStatus.filled
            order.fill_price = price_fill

        fee = self.FEE[order.type]

        if outstanding_qty:
            position.update(qty=qty_to_close,
                            price=price_fill,
                            multiplier=self.leverage[order.symbol],
                            fee=fee)
            assert position.is_closeable()
            self._close_position(order.symbol)
            position.update(qty=outstanding_qty,
                            price=price_fill,
                            multiplier=self.leverage[order.symbol],
                            fee=fee)
            assert not position.is_closeable()
        else:
            position.update(qty=qty_fill,
                            price=price_fill,
                            multiplier=self.leverage[order.symbol],
                            fee=fee)
            if position.is_closeable():
                self._close_position(order.symbol)

            fill = Fill(order=order,
                        qty_filled=qty_fill,
                        price_fill=price_fill,
                        fill_time=current_time)
            self.fills_hist += [fill]

    def _execute_liquidation(self, symbol):
        self.cancel_orders(self.active_orders.of_symbol(symbol), reason=OrderCancelReason.liquidation)
        position = self.positions[symbol]
        if not position.is_open():
            return
        order = OrderCommon(symbol=symbol, signed_qty=-position.position(), type=OrderType.market)
        self._execute_order(order)
        if position.is_open():
            raise AttributeError("position was not close during liquidation. position = %f" % position.position())
        if not self.is_last_candle():
            self.n_liquidations[symbol.name] += 1

    def _close_position(self, symbol, force_close=False):
        position = self.positions[symbol]
        assert position.is_closeable() or force_close
        position.close_ts = self.current_time()
        self.closed_positions_hist[symbol] += [copy.deepcopy(position)]
        self.xbt_balance += position.close_position()

    @staticmethod
    def _count_per_symbol(lista):
        count_per_symbol = defaultdict(int)
        for f in lista:
            count_per_symbol[f.symbol.name] += 1
        return dict(count_per_symbol)

    def print_summary(self):
        print("initial btc " + str(self.xbt_initial_balance))
        print("position btc = " + str(self.xbt_balance))
        print("num fills = " + str(self._count_per_symbol(self.fills_hist)))
        print("num orders = " + str(self._count_per_symbol(self.order_hist)))
        print("close price = " + str(self.candles.data.iloc[-1].close))
        total_pnl = 0.
        pnl = defaultdict(float)
        for symbol in self.closed_positions_hist:
            pnl[symbol.name] = sum([p.realized_pnl for p in self.closed_positions_hist[symbol]])
            total_pnl += pnl[symbol.name]
        print("PNL = " + str(dict(pnl)))
        print("PNL total = " + str(total_pnl))
        print("num order cancels = " + str(dict(self.n_cancels)))
        print("num liquidations = " + str(dict(self.n_liquidations)))

        assert abs(total_pnl - (self.xbt_balance - self.xbt_initial_balance)) < 1.e-8

    def print_output_files(self):
        output_dir = os.path.dirname(os.path.realpath(__file__))
        print("print results to " + output_dir)
        fills_file = open(os.path.join(output_dir, 'output.fills'), 'w')
        orders_file = open(os.path.join(output_dir, 'output.orders'), 'w')
        pnl_file = open(os.path.join(output_dir, 'output.pnl'), 'w')

        fills_file.write(Fill.get_header() + '\n')
        orders_file.write(Orders().to_csv() + '\n')

        for f in self.fills_hist:  # type: Fill
            fills_file.write(f.to_line() + '\n')
        orders_file.write(self.order_hist.to_csv(header=False))

        for s in self.SYMBOLS:
            for p in self.closed_positions_hist[s]:  # type: self.Position
                pnl_file.write(','.join([str(p.close_ts.strftime('%Y-%m-%dT%H:%M:%S')), str(p.realized_pnl)]) + '\n')

        pnl_file.close()
        fills_file.close()
        orders_file.close()


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
        exchange.print_output_files()

    return 0


if __name__ == "__main__":
    sys.exit(main())
