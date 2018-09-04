# NOTES:
# timestamps are of type pd.Timestamp
# side are of type str ('buy' or 'sell')
import argparse
import os
import shutil
import sys
from collections import defaultdict
# from autologging import logged
from enum import Enum
from typing import Dict, List, Iterable

import pandas as pd
from numpy.core.umath import sign

from api.exchange_interface import ExchangeInterface
from api.symbol import Symbol
from common import Fill, FillType, OrderType, OrderCancelReason, to_ohlcv, OrderStatus, OrderCommon, drop_orders, \
    filter_symbol, OrderContainerType
from common.order import drop_closed_orders_dict
from tactic import TacticInterface, TacticBitEwmWithStop
from utils.utils import to_nearest
from .position_sim import PositionSim
from .sim_stats import SimSummary


# import logging


def get_args(input_args=None):
    parser = argparse.ArgumentParser(description='Simulation')
    parser.add_argument('--ohlcv_file', type=str, help='csv filename with candles data', required=True)
    parser.add_argument('--trades_file', type=str, help='csv filename with trades data', required=True)
    parser.add_argument('-l', '--log-dir', type=str, help='log directory')
    parser.add_argument('-b', '--begin', type=str, help='begin time')
    parser.add_argument('-e', '--end', type=str, help='end time')
    parser.add_argument('-x', '--pref', action='append', help='args for tactics, given in the format "key=value"')

    args = parser.parse_args(args=input_args)

    for f in [args.ohlcv_file, args.trades_file]:
        if not os.path.isfile(f):
            raise ValueError("invalid file {}".format(f))

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
        super().__init__()

    def get_symbol(self) -> Symbol:
        pass

    def initialize(self, exchange, preferences):
        pass

    def handle_1m_candles(self, candles: pd.DataFrame):
        pass

    def handle_submission_error(self, failed_order):
        pass

    def handle_fill(self, fill):
        pass

    def handle_cancel(self, order):
        pass

    def id(self):
        return self.__class__.__name__


#  only BTC is supported for now
# @logged
class SimExchangeBitMex(ExchangeInterface):
    FEE = {OrderType.Limit: -0.00025, OrderType.Market: 0.00075}

    SYMBOLS = list(Symbol)

    # reference: https://www.bitmex.com/app/riskLimits#instrument-risk-limits
    RISK_LIMITS = {Symbol.XBTUSD: 0.0015}

    def __init__(self,
                 initial_balance: float,
                 ohlcv: pd.DataFrame,
                 trades: pd.DataFrame,
                 log_dir: str,
                 tactics: list):
        ExchangeInterface.__init__(self)
        self.xbt_initial_balance = initial_balance
        self.xbt_balance = initial_balance

        self.positions = dict()  # type: Dict[Symbol, PositionSim]
        self.leverage = {i: 1. for i in self.SYMBOLS}

        self.active_orders = dict()  # type: Dict[str, OrderCommon]

        # liq price for each position
        self.closed_positions_hist = defaultdict(list)  # type: Dict[Symbol, List[PositionSim]]
        self.fills_hist = []
        self.order_hist = []

        self.log_dir = log_dir
        self.ohlcv = ohlcv  # type: pd.DataFrame
        self.md_trades = trades  # type: pd.DataFrame
        self.tick_num = 0

        self.tactics_map = {t.id(): t for t in tactics}
        ss = [tac.get_symbol() for tac in tactics]
        zz = set(ss)
        if len(zz) != len(ss):
            raise ValueError("Tactics trading same symbol is not allowed.")

        self.n_cancels = defaultdict(int)
        self.n_liquidations = defaultdict(int)  # Symbol -> int

        # should be None until end of sim
        self.summary = None  # type: SimSummary

        self.liquidator = Liquidator()

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

    def get_quote(self, symbol: Symbol = None) -> dict:
        # TODO: implement symbol
        if symbol is not None:
            raise NotImplementedError()
        else:
            symbol = Symbol.XBTUSD
        ticker = {'buy': self.ohlcv.iloc[self.tick_num].low,
                  'sell': self.ohlcv.iloc[self.tick_num].high,
                  'last': self.ohlcv.iloc[self.tick_num].close,
                  'mid': 0.5 * (self.ohlcv.iloc[self.tick_num].low + self.ohlcv.iloc[self.tick_num].high)
                  }
        return {k: to_nearest(float(v or 0), symbol.tick_size) for k, v in ticker.items()}

    def next_price(self):
        return self.ohlcv.iloc[self.tick_num + 1].open

    def current_time(self):
        return self.ohlcv.iloc[self.tick_num].name

    def current_candle(self):
        return self.ohlcv.iloc[self.tick_num]

    def get_candles1m(self) -> pd.DataFrame:
        return self.ohlcv.head(self.tick_num + 1)

    def is_last_tick(self):
        return self.tick_num == len(self.md_trades) - 1

    """ Interface : NOTE: it assumes that when there is no position, return emtpy position"""

    def get_position(self, symbol: Symbol) -> PositionSim:
        if symbol in self.positions:
            return self.positions[symbol]
        pos = PositionSim(symbol)
        self.positions[symbol] = pos
        return pos

    def get_pnl_history(self, symbol: Symbol = None):
        # type: (Enum) -> list(PositionSim)
        return self.closed_positions_hist[symbol]

    def get_xbt_balance(self):
        return self.xbt_balance

    def print_progress(self):
        sys.stdout.write(
            "progress: %d out of %d (%.4f%%)   \r" %
            (self.tick_num, len(self.md_trades), 100 * float(self.tick_num) / len(self.md_trades)))
        sys.stdout.flush()

    # for sims only
    def advance_time(self, print_progress=True):
        if print_progress:
            self.print_progress()

        if self.is_last_tick():
            for symbol in self.SYMBOLS:
                self._execute_liquidation(symbol, order_cancel_reason=OrderCancelReason.end_of_sim)
            self.tick_num += 1
            self.print_progress()
            return
        else:
            if self.can_call_handles:
                for tactic in self.tactics_map.values():  # type: TacticInterface
                    tactic.handle_1m_candles(self.get_candles1m())
            self.tick_num += 1
            assert self.can_call_handles is True
            self._execute_all_orders()
            current_price = self._estimate_price()
            for symbol in self.SYMBOLS:
                position = self.positions.get(symbol, None)  # type: PositionSim
                if position and position.is_open:
                    side = position.side
                    liq_price = self.positions[symbol].liquidation_price
                    if (side > 0 and current_price < liq_price) or (side < 0 and current_price > liq_price):
                        self._execute_liquidation(symbol)

    def is_open(self):
        if self.tick_num < len(self.ohlcv):
            return True
        else:
            self.summary = self.get_summary()
            return False

    def cancel_orders(self, orders: Dict) -> Dict:
        return self._cancel_orders_helper(orders, reason=OrderCancelReason.requested_by_user)

    def _cancel_orders_helper(self,
                              orders: Dict,
                              reason: OrderCancelReason) -> Dict:
        orders_list = [o for o in orders.values() if o.client_id in self.active_orders]
        if len(orders_list) == 0:
            return dict()
        for o in orders_list:  # type: OrderCommon
            if o.status != OrderStatus.Canceled:
                o.status = OrderStatus.Canceled
                o.status_msg = reason
                self.n_cancels[reason.name] += 1
        if self.can_call_handles and reason != OrderCancelReason.requested_by_user:
            for o in orders_list:
                if o.tactic and o.tactic != Liquidator:
                    o.tactic.handle_cancel(self, o)

        cancelled = {o.client_id: o for o in orders_list if o.status == OrderStatus.Canceled}
        if len(cancelled) < 1:
            print("time at: " + str(self.current_time()))
            raise AttributeError("no orders were closed")
        return cancelled

    @staticmethod
    def _reject_order(order, time_posted, reason):
        # type: (OrderCommon, pd.Timestamp, OrderCancelReason) -> None
        assert order.status is OrderStatus.Pending  # it can only reject pending orders
        order.time_posted = time_posted
        order.status = OrderStatus.Canceled
        order.status_msg = reason

    """ note: may restart self.position"""

    @staticmethod
    def verify_order(order) -> OrderCommon:
        # return OrderCommon(**order.__dict__)
        return order

    def send_orders(self, orders: List[OrderCommon]) -> List[OrderCommon]:
        current_time = self.current_time()
        for i in range(len(orders)):
            orders[i].client_id = '{}_{}'.format(orders[i].client_id, i + len(self.order_hist))

        assert all([o.client_id not in self.active_orders for o in orders])

        self.order_hist += orders

        for o in orders:
            assert o.status == OrderStatus.Pending

        if self.tick_num == len(self.ohlcv) - 1:
            #  this is the last candle, cancel all limit orders
            for o in orders:  # type: OrderCommon
                if o.type != OrderType.Market:
                    self._reject_order(o, current_time, OrderCancelReason.end_of_sim)
            orders = [o for o in orders if o.type == OrderType.Market]

        # discard bad orders
        current_price = self.get_quote()['last']
        next_price = self.next_price() if not self.is_last_tick() else current_price
        for o in orders:  # type: OrderCommon
            if o.type is OrderType.Limit:
                # in case open is different from close
                crossed = (o.is_sell() and o.price < current_price) or (o.is_buy() and o.price > current_price) or \
                          (o.is_sell() and o.price < next_price) or (o.is_buy() and o.price > next_price)
                if crossed or o.price < 0:
                    # only post_only are supported, so don't let it cross
                    self._reject_order(o, current_time, OrderCancelReason.invalid_price)
                    contain_errors = True
                    continue
            elif o.type is not OrderType.Market:
                raise ValueError("invalid order type %s" % str(o.type))

        for o in orders:  # type: OrderCommon
            o.time_posted = current_time
            if o.status == OrderStatus.Pending:
                o.status = OrderStatus.New
                self.active_orders[o.client_id] = o
            else:
                assert o.status == OrderStatus.Canceled

        # print " SIMMMMM " + Orders.to_csv(orders.data.values())
        # print " -------------- "

        orders_list = [o for o in self.active_orders.values()]
        for o in orders_list:
            if (o.status == OrderStatus.Pending or o.is_open()) and o.type == OrderType.Market:
                self._execute_order(o)
        self.active_orders = drop_closed_orders_dict(self.active_orders)
        return orders_list

    def _estimate_price(self, current_candle=None):
        if current_candle is None:
            current_candle = self.current_candle()
        high = current_candle.high
        low = current_candle.low
        close_p = current_candle.close
        open_p = current_candle.open
        return (3 * open_p + 2. * (low + high) + close_p) / 8.

    """ note: may restart self.position """

    def _execute_all_orders(self):
        if not self.tick_num < len(self.ohlcv):
            print("self.time_idx = {}".format(self.tick_num))
            print("len(self.candles) = {}".format(len(self.ohlcv)))
            assert self.tick_num < len(self.ohlcv)

        for o in self.active_orders.values():
            assert o.status == OrderStatus.Pending or o.is_open()

        # self.active_orders may change during loop, so creating a copy
        orders_list = [o for o in self.active_orders.values()]
        for o in orders_list:
            self._execute_order(o)  # type: Fill
        self.active_orders = drop_closed_orders_dict(self.active_orders)

    def _update_position(self, symbol: Symbol, *args, **kwargs) -> PositionSim:
        if symbol in self.positions:
            position = self.positions[symbol].update(*args, **kwargs)
        else:
            position = PositionSim(symbol)
            self.positions[symbol] = position.update(*args, **kwargs)
            assert self.positions[symbol] == position

        if not position.is_open:
            self.closed_positions_hist[symbol] += [position]
            self.xbt_balance += position.realized_pnl  # TODO: position realized_pnl is deprecated
            del self.positions[symbol]

        return position

    """
    Note: may restart self.position
    """

    def get_opened_orders(self, symbol=None) -> OrderContainerType:
        return self.active_orders

    def _execute_order(self, order: OrderCommon) -> Fill:
        assert self.tick_num < len(self.ohlcv)

        if order.status != OrderStatus.Pending and not order.is_open():
            raise ValueError("expected order to be opened, but got " + str(order.status) + ". Order = \n"
                             + order.get_header() + "\n" + str(order))
        current_candle = self.current_candle()
        current_time = current_candle.name  # pd.Timestamp
        high = current_candle.high
        low = current_candle.low
        open = current_candle.open
        close = current_candle.close

        position = self.positions[order.symbol]  # type: PositionSim
        current_qty = position.current_qty
        qty_fill = qty_to_close = outstanding_qty = None
        crossed = False

        if order.type is OrderType.Market:
            crossed = True
            price_fill = self._estimate_price()
            qty_fill = order.signed_qty
        elif order.type is OrderType.Limit:
            price_fill = order.price
            max_qty_fill = order.leaves_qty * order.side()
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
            return None  # type: Fill

        if position.is_open and position.would_change_side(qty_fill):
            qty_to_close = float(sign(qty_fill)) * min(abs(current_qty), abs(qty_fill))
            outstanding_qty = qty_fill - qty_to_close

        if order.fill(qty_fill) or order.type == OrderType.Market:
            order.status = OrderStatus.Filled
            order.fill_price = price_fill

        if order.price is not None and ((open <= order.price <= close) or (close <= order.price <= open)):
            assert order.status == OrderStatus.Filled

        fee = self.FEE[order.type]

        if outstanding_qty:
            position = self._update_position(order.symbol,
                                             qty=qty_to_close,
                                             price=price_fill,
                                             leverage=self.leverage[order.symbol],
                                             current_timestamp=current_time,
                                             fee=fee)
            assert not position.is_open

            position = self._update_position(order.symbol,
                                             qty=outstanding_qty,
                                             price=price_fill,
                                             leverage=self.leverage[order.symbol],
                                             current_timestamp=current_time,
                                             fee=fee)
            assert position.is_open
        else:
            self._update_position(order.symbol,
                                  qty=qty_fill,
                                  price=price_fill,
                                  leverage=self.leverage[order.symbol],
                                  current_timestamp=current_time,
                                  fee=fee)

        fill = Fill(order=order,
                    qty_filled=qty_fill,
                    price_fill=price_fill,
                    fill_time=current_time,
                    fill_type=FillType.complete if (order.status == OrderStatus.Filled) else FillType.partial)
        self.fills_hist += [fill]
        self.active_orders = drop_closed_orders_dict(self.active_orders)
        if self.can_call_handles:
            order.tactic.handle_fill(fill)
        return fill

    def _execute_liquidation(self, symbol, order_cancel_reason=OrderCancelReason.liquidation):
        self.can_call_handles = False
        orders = filter_symbol(self.active_orders, symbol)
        cancelled = self._cancel_orders_helper(orders, reason=order_cancel_reason)
        self.active_orders = drop_orders(self.active_orders, cancelled)
        self.can_call_handles = True
        try:
            assert len(filter_symbol(self.active_orders, symbol)) == 0
        except:
            for i in filter_symbol(self.active_orders, symbol):
                print("-" + str(i[1].status))
            print("-" + str(len(cancelled)))
            raise AttributeError()
        position = self.positions.get(symbol, None)
        if not position or not position.is_open:
            return
        assert position.is_open
        order = OrderCommon(symbol=symbol,
                            signed_qty=-position.current_qty,
                            type=OrderType.Market,
                            tactic=self.liquidator)
        order.status_msg = order_cancel_reason
        self.can_call_handles = False
        self.send_orders([order])
        self.can_call_handles = True
        assert order.status == OrderStatus.Filled
        if position.is_open:
            raise AttributeError("position was not close during liquidation. position = %f" % position.current_qty)
        if not self.is_last_tick():
            self.n_liquidations[symbol.name] += 1
        if order_cancel_reason == OrderCancelReason.liquidation:
            closed = self.closed_positions_hist[symbol][-1]  # type: PositionSim
            if closed.realized_pnl >= 0:
                raise AttributeError("Liquidation caused profit! position = {},\n current price = {}"
                                     .format(str(position), self._estimate_price()))
        assert len(filter_symbol(self.active_orders, symbol)) == 0

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
            close_price=self.ohlcv.iloc[-1].close,
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
            for p in self.closed_positions_hist[s]:  # type: PositionSim
                sum += p.realized_pnl
                pnl_file.write(','.join([str(p.current_timestamp.strftime('%Y-%m-%dT%H:%M:%S')),
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


def read_ohlcv_and_trades(ohlcv_file, trades_file) -> tuple:
    ohlcv = pd.read_csv(ohlcv_file)
    ohlcv.set_index('timestamp', inplace=True)
    ohlcv.index = pd.DatetimeIndex(ohlcv.index)
    assert list(ohlcv.columns) == ['symbol', 'open', 'high', 'low', 'close', 'size']

    trades = pd.read_csv(trades_file)
    trades.set_index('timestamp', inplace=True)
    trades.index = pd.DatetimeIndex(trades.index)
    assert list(trades.columns) == ['symbol', 'side', 'price', 'size', 'tickDirection']

    return ohlcv, trades


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
    tactics = [TacticBitEwmWithStop(Symbol.XBTUSD)]

    ohlcv, trades = read_ohlcv_and_trades(args.ohlcv_file, args.trades_file)

    with SimExchangeBitMex(0.2, ohlcv, trades, args.log_dir, tactics) as exchange:

        for tac in exchange.tactics_map.values():
            tac.initialize(exchange, args.pref)

        while exchange.is_open():
            exchange.advance_time(print_progress=True)

        summary = exchange.summary  # type: SimSummary
        print(summary.to_str())
        exchange.print_output_files(input_args if input_args else sys.argv)

    if __name__ == "__main__":
        return 0
    else:
        return summary


if __name__ == "__main__":
    sys.exit(main())
