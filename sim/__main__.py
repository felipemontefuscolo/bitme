# NOTES:
# timestamps are of type pd.Timestamp
# side are of type str ('buy' or 'sell')
import argparse
import copy
import os
import shutil
import sys
from collections import defaultdict

import pandas as pd
# from autologging import logged
from enum import Enum
from numpy.core.umath import sign

from common import ExchangeInterface, Fill, FillType, OrderType, Orders, OrderCancelReason, \
    OrderStatus, OrderCommon, Position, to_ohlcv
from .sim_stats import SimSummary
from tactic import TacticInterface, TacticBitEwmWithStop
import tools

# import logging


def get_args(input_args=None):
    parser = argparse.ArgumentParser(description='Simulation')
    parser.add_argument('-f', '--file', type=str, help='csv filename with candles data', required=True)
    parser.add_argument('-l', '--log-dir', type=str, help='log directory')
    parser.add_argument('-b', '--begin', type=str, help='begin time')
    parser.add_argument('-e', '--end', type=str, help='end time')
    parser.add_argument('-x', '--pref', action='append', help='args for tactics, given in the format "key=value"')
    parser.add_argument('--no-summary', action="store_true", default=False)
    parser.add_argument('--no-output', action="store_true", default=False)

    args = parser.parse_args(args=input_args)

    if not os.path.isfile(args.file):
        raise ValueError("invalid file {}".format(args.file))

    if args.log_dir is not None:
        if os.path.isfile(args.log_dir):
            raise ValueError(args.log_dir + " is a file")
        args.log_dir = os.path.abspath(args.log_dir)
        if os.path.isdir(args.log_dir):
            shutil.rmtree(args.log_dir)
        os.makedirs(args.log_dir)

    if args.begin:
        args.begin = pd.Timestamp(args.begin)
    if args.end:
        args.end = pd.Timestamp(args.end)

    if not args.pref:
        args.pref = list()
    for i in range(len(args.pref)):
        args.pref[i] = args.pref[i].split("=")
    args.pref = dict(args.pref)

    return args


class Liquidator(TacticInterface):
    def __init__(self):
        pass

    def init(self, exchange, preferences):
        pass

    def handle_candles(self, exchange):
        pass

    def handle_submission_error(self, failed_order):
        pass

    def handle_fill(self, exchange, fill):
        pass

    def handle_cancel(self, exchange, order):
        pass

    def id(self):
        return self.__class__.__name__


#  only BTC is supported for now
# @logged
class SimExchangeBitMex(ExchangeInterface):
    FEE = {OrderType.limit: -0.00025, OrderType.market: 0.00075}

    class Symbol(Enum):
        XBTUSD = 'XBTUSD'
        XBTH18 = 'XBTH18'
        __iter__ = Enum.__iter__

    SYMBOLS = list(Symbol)

    # reference: https://www.bitmex.com/app/riskLimits#instrument-risk-limits
    RISK_LIMITS = {Symbol.XBTUSD: 0.0015, Symbol.XBTH18: 0.0015}

    def __init__(self, initial_balance, file_name, log_dir, tactics):
        ExchangeInterface.__init__(self)
        self.xbt_initial_balance = initial_balance
        self.xbt_balance = initial_balance

        self.positions = defaultdict(Position)  # type: defaultdict(Position)
        self.leverage = dict([(i, 100.) for i in self.SYMBOLS])  # 1 means 1%

        self.active_orders = Orders()

        # liq price for each position
        self.closed_positions_hist = defaultdict(list)  # type: defaultdict(Position)
        self.fills_hist = []
        self.order_hist = []

        self.file_name = os.path.abspath(file_name)
        self.log_dir = log_dir
        self.candles = to_ohlcv(filename=file_name)  # type: pd.DataFrame
        self.time_idx = 0

        self.tactics = tactics
        ss = [tac.get_symbol() for tac in tactics]
        zz = set(ss)
        if len(zz) != len(ss):
            raise ValueError("Tactics trading same symbol is not allowed.")

        self.n_cancels = defaultdict(int)
        self.n_liquidations = defaultdict(int)  # Symbol -> int

        self.can_call_handles = True  # type: bool

        # should be None until end of sim
        self.summary = None  # type: SimSummary

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
        return self.candles.iloc[self.time_idx].close

    def next_price(self):
        return self.candles.iloc[self.time_idx + 1].open

    def current_time(self):
        return self.candles.iloc[self.time_idx].name

    def current_candle(self):
        return self.candles.iloc[self.time_idx]

    def get_candles1m(self) -> pd.DataFrame:
        return self.candles.head(self.time_idx + 1)

    def is_last_candle(self):
        return self.time_idx == len(self.candles) - 1

    def get_position(self, symbol):
        # type: (self.Symbol) -> Position
        return self.positions[symbol]

    def get_closed_positions(self, symbol):
        # type: (Enum) -> list(Position)
        return self.closed_positions_hist[symbol]

    def get_xbt_balance(self):
        return self.xbt_balance

    # for sims only
    def advance_time(self, print_progress=True):
        if print_progress:
            sys.stdout.write(
                "progress: %d out of %d (%.4f%%)   \r" %
                (self.time_idx, len(self.candles), 100 * float(self.time_idx) / len(self.candles)))
            sys.stdout.flush()

        if self.is_last_candle():
            for symbol in self.SYMBOLS:
                self._execute_liquidation(symbol, order_cancel_reason=OrderCancelReason.end_of_sim)
            self.time_idx += 1
            sys.stdout.write("progress: %d out of %d (%.4f%%)\n" % (self.time_idx, len(self.candles), 100.))
            return
        else:
            if self.can_call_handles:
                for tactic in self.tactics:  # type: TacticInterface
                    tactic.handle_candles(self)
            self.time_idx += 1
            assert self.can_call_handles is True
            self._execute_all_orders()
            current_price = self._estimate_price()
            for symbol in self.SYMBOLS:
                position = self.positions[symbol]  # type: Position
                if position.is_open():
                    side = position.side
                    liq_price = self.positions[symbol].liq_price
                    if (side > 0 and current_price < liq_price) or (side < 0 and current_price > liq_price):
                        self._execute_liquidation(symbol)

    def is_open(self):
        if self.time_idx < len(self.candles):
            return True
        else:
            self.summary = self.get_summary()
            return False

    def cancel_orders(self,
                      orders: Orders,
                      drop_canceled=True,
                      status=OrderStatus.canceled,
                      reason=OrderCancelReason.requested_by_user):
        orders_list = [o for o in orders if o.id in self.active_orders.data]
        if len(orders_list) == 0:
            return
        for o in orders_list:  # type: OrderCommon
            if o.status != status:
                o.status = status
                o.status_msg = reason
                self.n_cancels[reason.name] += 1
        if self.can_call_handles and reason != OrderCancelReason.requested_by_user:
            for o in orders_list:
                if o.tactic and o.tactic != Liquidator:
                    o.tactic.handle_cancel(self, o)

        if orders.size() > 0 and not self.active_orders.drop_closed_orders() > 0:  # in case orders does not refers to self.opened_orders
            print("time at: " + str(self.current_time()))
            raise AttributeError("no orders were closed")
        if drop_canceled:
            if orders.size() > 0 and not orders.drop_closed_orders() > 0:
                print("time at: " + str(self.current_time()))
                raise AttributeError("no orders were closed")

    @staticmethod
    def _reject_order(order, time_posted, reason):
        # type: (OrderCommon, pd.Timestamp, OrderCancelReason) -> None
        assert order.status is OrderStatus.pending  # it can only reject pending orders
        order.time_posted = time_posted
        order.status = OrderStatus.canceled
        order.status_msg = reason

    def post_orders(self, orders):
        # type: (Orders) -> bool
        # may change order status
        # return true if any submission failed
        contain_errors = False
        current_time = self.current_time()
        self.order_hist += orders.values()

        for o in orders:
            assert o.status == OrderStatus.pending

        if self.time_idx == len(self.candles) - 1:
            #  this is the last candle, cancel all limit orders
            for o in orders:  # type: OrderCommon
                if o.type != OrderType.market:
                    self._reject_order(o, current_time, OrderCancelReason.end_of_sim)
            orders = orders.market_orders()

        # discard bad orders
        current_price = self.current_price()
        next_price = self.next_price() if not self.is_last_candle() else current_price
        for o in orders:  # type: OrderCommon
            if o.type is OrderType.limit:
                # in case open is different from close
                crossed = (o.is_sell() and o.price < current_price) or (o.is_buy() and o.price > current_price) or \
                          (o.is_sell() and o.price < next_price) or (o.is_buy() and o.price > next_price)
                if crossed or o.price < 0:
                    # only post_only are supported, so don't let it cross
                    self._reject_order(o, current_time, OrderCancelReason.invalid_price)
                    contain_errors = True
                    continue
            elif o.type is not OrderType.market:
                raise ValueError("invalid order type %s" % str(o.type))

        for o in orders:  # type: OrderCommon
            o.time_posted = current_time
            if o.status == OrderStatus.pending:
                o.status = OrderStatus.opened
                self.active_orders.add(o)
            else:
                assert o.status == OrderStatus.canceled

        # print " SIMMMMM " + Orders.to_csv(orders.data.values())
        # print " -------------- "

        orders_list = [o for o in self.active_orders]
        for o in orders_list:
            if o.status == OrderStatus.opened and o.type == OrderType.market:
                self._execute_order(o)
        self.active_orders.drop_closed_orders()
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
        if not self.time_idx < len(self.candles):
            print("self.time_idx = {}".format(self.time_idx))
            print("len(self.candles) = {}".format(len(self.candles)))
            assert self.time_idx < len(self.candles)
            
        for o in self.active_orders:
            assert o.status == OrderStatus.opened

        # print "_______________BEGIN_______________________"
        # orders status may change in the loop
        orders_list = [o for o in self.active_orders]
        for o in orders_list:
            if o.status == OrderStatus.opened:
                self._execute_order(o)  # type: Fill
        self.active_orders.drop_closed_orders()
        # print "_________________END_______________________"

    def _execute_order(self, order):
        # type: (OrderCommon) -> Fill
        assert self.time_idx < len(self.candles)

        if order.status != OrderStatus.opened:
            raise ValueError("expected order to be opened, but got " + str(order.status) + ". Order = \n"
                             + order.get_header() + "\n" + str(order))
        current_candle = self.current_candle()
        current_time = current_candle.name  # pd.Timestamp
        high = current_candle.high
        low = current_candle.low
        open = current_candle.open
        close = current_candle.close

        position = self.positions[order.symbol]  # type: Position
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
            if (open <= order.price <= close) or (close <= order.price <= open):
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
            return None

        if position.does_change_side(qty_fill):
            qty_to_close = sign(qty_fill) * min(abs(position_value), abs(qty_fill))
            outstanding_qty = qty_fill - qty_to_close

        if order.fill(qty_fill) or order.type == OrderType.market:
            order.status = OrderStatus.filled
            order.fill_price = price_fill

        if (open <= order.price <= close) or (close <= order.price <= open):
            assert order.is_fully_filled()

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
                    fill_time=current_time,
                    fill_type=FillType.complete if order.is_fully_filled() else FillType.partial)
        self.fills_hist += [fill]
        self.active_orders.drop_closed_orders()
        if self.can_call_handles:
            order.tactic.handle_fill(self, fill)
        return fill

    def _execute_liquidation(self, symbol, order_cancel_reason=OrderCancelReason.liquidation):
        self.can_call_handles = False
        self.cancel_orders(self.active_orders.of_symbol(symbol), reason=order_cancel_reason)
        self.can_call_handles = True
        assert self.active_orders.of_symbol(symbol).size() == 0
        position = self.positions[symbol]
        if not position.is_open():
            return
        order = OrderCommon(symbol=symbol, signed_qty=-position.position(), type=OrderType.market, tactic=Liquidator())
        order.status_msg = order_cancel_reason
        self.can_call_handles = False
        self.post_orders(Orders({order.id: order}))
        self.can_call_handles = True
        assert order.status == OrderStatus.filled
        if position.is_open():
            raise AttributeError("position was not close during liquidation. position = %f" % position.position())
        if not self.is_last_candle():
            self.n_liquidations[symbol.name] += 1
        if order_cancel_reason == OrderCancelReason.liquidation:
            closed = self.closed_positions_hist[symbol][-1]  # type: Position
            if closed.realized_pnl >= 0:
                raise AttributeError("Liquidation caused profit! position = \n" + str(position))
        assert self.active_orders.of_symbol(symbol).size() == 0

    def _close_position(self, symbol, force_close=False):
        position = self.positions[symbol]  # type: Position
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

    def get_summary(self):
        # type: () -> SimSummary
        total_pnl = 0.
        total_loss = 0.
        total_profit = 0.
        pnl = defaultdict(float)
        profit = defaultdict(float)
        loss = defaultdict(float)
        for symbol in self.closed_positions_hist:
            pnl[symbol.name] = sum([p.realized_pnl for p in self.closed_positions_hist[symbol]])
            profit[symbol.name] = sum(
                [p.realized_pnl for p in self.closed_positions_hist[symbol] if p.realized_pnl >= 0])
            loss[symbol.name] = sum([p.realized_pnl for p in self.closed_positions_hist[symbol] if p.realized_pnl < 0])
            total_pnl += pnl[symbol.name]
            total_profit += profit[symbol.name]
            total_loss += loss[symbol.name]

        assert abs(total_pnl - (self.xbt_balance - self.xbt_initial_balance)) < 1.e-8
        return SimSummary(
            initial_xbt=self.xbt_initial_balance,
            position_xbt=self.xbt_balance,
            num_fills=self._count_per_symbol(self.fills_hist),
            num_orders=self._count_per_symbol(self.order_hist),
            num_cancels=dict(self.n_cancels),
            num_liq=dict(self.n_liquidations),
            close_price=self.candles.iloc[-1].close,
            pnl=dict(pnl),
            pnl_total=total_pnl,
            profit_total=total_profit,
            loss_total=total_loss
        )

    def print_output_files(self, input_args):
        if self.log_dir is None:
            raise ValueError("asked to print results, but log_dir is None")
        print("printing results to " + self.log_dir)
        fills_file = open(os.path.join(self.log_dir, 'fills.csv'), 'w')
        orders_file = open(os.path.join(self.log_dir, 'orders.csv'), 'w')
        pnl_file = open(os.path.join(self.log_dir, 'pnl.csv'), 'w')
        pars_used_file = open(os.path.join(self.log_dir, 'parameters_used'), 'w')
        summary_file = open(os.path.join(self.log_dir, 'summary'), 'w')

        fills_file.write(Fill.get_header() + '\n')
        orders_file.write(OrderCommon.get_header() + '\n')

        for f in self.fills_hist:  # type: Fill
            fills_file.write(f.to_line() + '\n')
        for o in self.order_hist:  # type: OrderCommon
            orders_file.write(o.to_line() + '\n')

        pnl_file.write('time,symbol,pnl,cum_pnl\n')
        for s in self.SYMBOLS:
            sum = 0
            for p in self.closed_positions_hist[s]:  # type: Position
                sum += p.realized_pnl
                pnl_file.write(','.join([str(p.close_ts.strftime('%Y-%m-%dT%H:%M:%S')),
                                         s.name,
                                         str(p.realized_pnl),
                                         str(sum)])
                               + '\n')
        pars_used_file.write(str(input_args))
        pars_used_file.write("")

        summary_file.write(self.summary.to_str())

        pars_used_file.close()
        pnl_file.close()
        fills_file.close()
        orders_file.close()
        summary_file.close()


# @logged
def main(input_args=None):
    # logging.basicConfig(
    #     filename='messages',
    #     filemode='w',
    #     level=logging.INFO,
    #     format='%(levelname)s:%(name)s:%(funcName)s:%(message)s '
    # )
    # main._log.info("starting SIM")
    args = get_args(input_args)

    # defin here the tactic you want to activate
    tactics = [TacticBitEwmWithStop(SimExchangeBitMex.Symbol.XBTUSD)]

    with SimExchangeBitMex(0.2, args.file, args.log_dir, tactics) as exchange:

        for tac in exchange.tactics:
            tac.init(exchange, args.pref)

        while exchange.is_open():
            exchange.advance_time(print_progress=True)

        summary = exchange.summary  # type: SimSummary
        if not args.no_summary:
            print(summary.to_str())
        if args.log_dir is not None and not args.no_output:
            exchange.print_output_files(input_args if input_args else sys.argv)

    if __name__ == "__main__":
        return 0
    else:
        return summary


if __name__ == "__main__":
    sys.exit(main())
