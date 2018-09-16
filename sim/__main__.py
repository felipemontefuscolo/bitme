# NOTES:
# timestamps are of type pd.Timestamp
# side are of type str ('buy' or 'sell')
import argparse
import heapq
import os
import shutil
import sys
import time
from collections import defaultdict
from typing import List, Union, Dict

import numpy as np
import pandas as pd

from api import PositionInterface
from api.exchange_interface import ExchangeInterface
from api.symbol import Symbol
from common import OrderType, OrderContainerType, OrderCommon, OHLCV_COLUMNS, Fill, FillType, get_orders_id, \
    OrderCancelReason, OrderStatus
from common.quote import Quote
from common.trade import Trade, TICK_DIRECTION
from sim.liquidator import Liquidator
from sim.position_sim import PositionSim
from tactic import TacticInterface
from tactic.tactic_tests.SimTacticLimitTest import SimTacticLimitTest
from tactic.tactic_tests.SimTacticMarketTest import SimTacticMarketTest

# logger = log.setup_custom_logger('root')

REQUEST_DELAY = pd.Timedelta('10ms')
WS_DELAY = pd.Timedelta('1ms')
ORDER_TO_FILL_DELAY = REQUEST_DELAY
CANCEL_DELAY = REQUEST_DELAY
CANCEL_NOTIF_DELAY = REQUEST_DELAY


# from autologging import logged

# TODO: prevent send orders where you would be liquidated immediately
# TODO; prevent send orders if you have no funds (xbt_balance is not in use)


# import logging


ALL_TACTICS = {
    SimTacticMarketTest.id(): SimTacticMarketTest,
    SimTacticLimitTest.id(): SimTacticLimitTest
}

def get_args(input_args=None):
    files_required = '--files' not in (input_args if input_args else sys.argv)

    parser = argparse.ArgumentParser(description='Simulation')
    parser.add_argument('--ohlcv-file', type=str, help='csv filename with candles data', required=files_required)
    parser.add_argument('--trades-file', type=str, help='csv filename with trades data', required=files_required)
    parser.add_argument('--quotes-file', type=str, help='csv filename with quotes data', required=files_required)
    parser.add_argument('--files', type=str, help='template path to the ohlcv, trades and quotes files (use %TYPE%)')
    parser.add_argument('--tactics', type=str, help='comma-delimited list of tactics (id) to run', required=True)
    parser.add_argument('-l', '--log-dir', type=str, help='log directory', required=True)
    parser.add_argument('-b', '--begin', type=str, help='begin time')
    parser.add_argument('-e', '--end', type=str, help='end time')
    parser.add_argument('-x', '--pref', action='append', help='args for tactics, given in the format "key=value"')

    args = parser.parse_args(args=input_args)

    if not files_required:
        args.ohlcv_file = args.files.replace('%TYPE%', 'ohlcv')
        args.trades_file = args.files.replace('%TYPE%', 'trades')
        args.quotes_file = args.files.replace('%TYPE%', 'quotes')

    for f in [args.ohlcv_file, args.trades_file, args.quotes_file]:
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

    if args.begin and args.end:
        if args.begin >= args.end:
            raise ValueError("begin time must be before end time")

    if not args.pref:
        args.pref = list()
    for i in range(len(args.pref)):
        args.pref[i] = args.pref[i].split("=")
    args.pref = dict(args.pref)

    tactics = args.tactics.split(',')
    for tactic in tactics:
        if tactic not in ALL_TACTICS.keys():
            ValueError("Unknown tactic {}".format(tactic))
    args.tactics = [ALL_TACTICS[t] for t in tactics]

    return args


#  only BTC is supported for now
# @logged
class SimExchangeBitMex(ExchangeInterface):
    FEE = {OrderType.Limit: -0.00025, OrderType.Market: 0.00075}

    SYMBOLS = list(Symbol)

    # reference: https://www.bitmex.com/app/riskLimits#instrument-risk-limits
    RISK_LIMITS = {Symbol.XBTUSD: 0.0015}

    def __init__(self,
                 begin_timestamp: pd.Timestamp,
                 end_timestamp: pd.Timestamp,
                 initial_balance: float,
                 ohlcv: pd.DataFrame,
                 trades: pd.DataFrame,
                 quotes: pd.DataFrame,
                 log_dir: str,
                 tactics: List[TacticInterface],
                 tactic_prefs: dict):
        ExchangeInterface.__init__(self)

        self.finished = False
        self.sim_start_time = False
        self.xbt_initial_balance = initial_balance
        self.xbt_balance = initial_balance
        self.log_dir = log_dir
        self.tactic_prefs = tactic_prefs

        assert ohlcv is not None and len(ohlcv) > 0
        assert trades is not None and len(trades) > 0
        assert quotes is not None and len(quotes) > 0
        self.ohlcv = ohlcv  # type: pd.DataFrame
        self.trades = trades  # type: pd.DataFrame
        self.quotes = quotes  # type: pd.DataFrame

        self.begin_timestamp = begin_timestamp
        self.end_timestamp = end_timestamp

        self.ohlcv_idx = 0
        self.trade_idx = 0
        self.quote_idx = 0

        if self.end_timestamp <= self.begin_timestamp:
            raise ValueError(
                "end_timestamp ({}) less or equal begin_timestamp ({}). Please check arguments and make sure "
                "the market data first event occurs before the end time".format(
                    self.end_timestamp, self.begin_timestamp
                ))

        self.current_timestamp = self.begin_timestamp  # type: pd.Timestamp
        # market data tick: quotes, trades, ohlcv
        self.next_tick_ts = self.begin_timestamp  # type: pd.Timestamp

        self.liquidator_tactic = Liquidator()
        self.tactics_map = {t.id(): t for t in [self.liquidator_tactic] + tactics}  # type: Dict[str, TacticInterface]
        ss = [tac.get_symbol() for tac in tactics]
        zz = set(ss)
        if len(zz) != len(ss):
            raise ValueError("Tactics trading same symbol is not allowed.")

        self.volume = {s: 0. for s in self.SYMBOLS}  # type: Dict[Symbol, float]
        self.n_fills = {s: 0 for s in self.SYMBOLS}  # type: Dict[Symbol, int]
        self.n_orders = {s: 0 for s in self.SYMBOLS}  # type: Dict[Symbol, int]
        self.n_unsolicited_cancels = {s: 0 for s in self.SYMBOLS}  # type: Dict[Symbol, int]
        self.n_liquidations = {s: 0 for s in self.SYMBOLS}  # type: Dict[Symbol, int]
        self.cum_pnl = {s: 0. for s in self.SYMBOLS}  # type: Dict[Symbol, float]
        self.pnl_history = defaultdict(list)  # type: Dict[Symbol, List[float]]
        self.positions = {s: PositionSim(s, self._log_and_update_pnl) for s in
                          self.SYMBOLS}  # type: Dict[Symbol, PositionSim]
        self.leverage = {s: 1.0 for s in self.SYMBOLS}  # type: Dict[Symbol, float]

        self.queue = []  # queue of events. See self._queue_append
        self.event_num = 0

        self.active_orders = dict()  # type: OrderContainerType

        self.current_quote = None  # type: Quote

    def _init_files(self, log_dir):
        self.fills_file = open(os.path.join(log_dir, 'fills.csv'), 'w')
        self.orders_file = open(os.path.join(log_dir, 'orders.csv'), 'w')
        self.pnl_file = open(os.path.join(log_dir, 'pnl.csv'), 'w')

        self.fills_file.write(Fill.get_header() + '\n')
        self.orders_file.write(OrderCommon.get_header() + '\n')
        self.pnl_file.write('time,symbol,pnl,cum_pnl\n')

    def _init_tactics(self):
        for tac in self.tactics_map.values():
            tac.initialize(self, self.tactic_prefs)
            tac.is_live = False

    def _close_files(self):
        assert self.finished
        self.fills_file.close()
        self.orders_file.close()
        self.pnl_file.close()

    def _log_fill(self, fill: Fill):
        self.fills_file.write(fill.to_line() + '\n')

    def _log_order(self, order: OrderCommon):
        self.orders_file.write(order.to_line() + '\n')

    def _log_and_update_pnl(self, position: PositionSim):
        timestamp = self.current_timestamp
        pnl = position.realized_pnl
        assert not np.isnan(pnl)
        symbol = position.symbol
        self.pnl_history[symbol].append(pnl)
        self.cum_pnl[symbol] += pnl
        self.pnl_file.write(','.join([str(timestamp.strftime('%Y-%m-%dT%H:%M:%S')),
                                      symbol.name,
                                      str(pnl),
                                      str(self.cum_pnl[symbol])])
                            + '\n')

    def __enter__(self):
        self.start_main_loop()
        return self

    def __exit__(self, type, value, traceback):
        self.end_main_loop()
        pass

    def run_sim(self):
        self.start_main_loop()
        self.end_main_loop()

    def start_main_loop(self):
        if self.sim_start_time:
            raise AttributeError('Live already started')
        self.sim_start_time = time.time()

        self._init_files(self.log_dir)
        self._init_tactics()

        tactics_names = [i.__class__.__name__ for i in self.tactics_map.values()]
        tactics_names = [i for i in tactics_names if i != Liquidator.__name__]
        print("Tactics running: {}".format(tactics_names))
        print("Market data tick range: {} - {}".format(self.begin_timestamp, self.end_timestamp))
        while not self.finished:
            self._advance_tick_timestamp()
            self._process_queue_until(self.next_tick_ts)
        self._process_queue_until(execute_all=True)

    def end_main_loop(self):
        for symbol in list(Symbol):
            self._exec_liquidation(symbol, reason=OrderCancelReason.end_of_sim)
        for tac in self.tactics_map.values():
            tac.finalize()
        self._close_files()

        profit_part = defaultdict(float)
        loss_part = defaultdict(float)

        for s in self.pnl_history.keys():
            for p in self.pnl_history[s]:
                if p >= 0:
                    profit_part[s] += p
                else:
                    loss_part[s] += p

        summary = {'initial_xbt': self.xbt_initial_balance,
                   'position_xbt': 'Not implemented',
                   'num_fills': self.n_fills,
                   'volume': self.volume,
                   'num_orders': self.n_orders,
                   'n_unsolicited_cancels': self.n_unsolicited_cancels,
                   'num_liq': self.n_liquidations,
                   'close_price': self.current_quote.w_mid(),
                   'pnl (XBT)': self.cum_pnl,
                   'pnl_total': sum([p for p in self.cum_pnl.values()]),
                   'profit_part': profit_part,
                   'loss_part': loss_part,
                   'sim time': time.time() - self.sim_start_time}

        for k, v in summary.items():
            print('{}: {}'.format(k, self._transf(v, len(k))))

    @staticmethod
    def _transf(d, l):
        if isinstance(d, dict) or isinstance(d, defaultdict):
            s = []
            l += 2
            for k, v in d.items():
                s += ["{}: {}".format(str(k), v)]
            return ('\n' + ' ' * l).join(s)
        else:
            return d

    def _process_queue_until(self, end_inclusive: pd.Timestamp = None, execute_all=False):
        assert (end_inclusive is not None) ^ execute_all  # XOR
        while len(self.queue) > 0 and (execute_all or self.queue[0][0] <= end_inclusive):
            task = heapq.heappop(self.queue)
            if self.current_timestamp > task[0]:
                raise AttributeError("Going back in time")
            self.current_timestamp = task[0]
            method = task[2]
            args = task[3]  # if we ever need to change the number of arguments to more than one, you can use *args
            # print(self.current_timestamp, method)
            method(args)

        if end_inclusive:
            self.print_progress(self.current_timestamp)
        else:
            self.print_progress(self.end_timestamp)

    def _queue_append(self, timestamp, method, method_argument):
        tuple_ = (timestamp, self.event_num, method, method_argument)
        self.event_num += 1
        heapq.heappush(self.queue, tuple_)

    def _advance_tick_timestamp(self):
        # DEV NOTE: advance time stamp and put market data in the queue

        def get_table_ts_at_idx(table, idx):
            if idx >= len(table):
                return self.end_timestamp
            return table.index[idx]

        next_trade_ts = get_table_ts_at_idx(self.trades, self.trade_idx)
        next_quote_ts = get_table_ts_at_idx(self.quotes, self.quote_idx)
        next_ohlcv_ts = get_table_ts_at_idx(self.ohlcv, self.ohlcv_idx)

        next_tick_ts = min(next_trade_ts, next_quote_ts, next_ohlcv_ts)

        if next_tick_ts >= self.end_timestamp:
            self.finished = True
            return

        quote_row = self.quotes.iloc[self.quote_idx]
        self.current_quote = Quote(symbol=Symbol[quote_row['symbol']],
                                   timestamp=next_quote_ts,
                                   bid_size=quote_row['bidSize'],
                                   bid_price=quote_row['bidPrice'],
                                   ask_size=quote_row['askSize'],
                                   ask_price=quote_row['askPrice'])

        def add_data_to_queue_until(end_ts_inclusive, method, table, idx):
            while idx < len(table) and table.index[idx] <= end_ts_inclusive:
                self._queue_append(table.index[idx], method, table.iloc[idx])
                idx += 1
            return idx

        self.trade_idx = add_data_to_queue_until(next_tick_ts, self._process_trade, self.trades, self.trade_idx)
        self.quote_idx = add_data_to_queue_until(next_tick_ts, self._process_quote, self.quotes, self.quote_idx)
        self.ohlcv_idx = add_data_to_queue_until(next_tick_ts, self._process_ohlcv, self.ohlcv, self.ohlcv_idx)

        self.next_tick_ts = next_tick_ts
        return

    def _exec_liquidation(self, symbol: Symbol, reason=OrderCancelReason.liquidation):
        orders_to_cancel = [o for o in self.active_orders.values() if o.symbol == symbol]
        for order in orders_to_cancel:
            order.status_msg = reason
        self._exec_order_cancels(orders_to_cancel)

        if reason == OrderCancelReason.liquidation:
            self.n_liquidations[symbol] += 1

        if self.positions[symbol].is_open:
            order = OrderCommon(symbol=symbol,
                                type=OrderType.Market,
                                client_id=self.liquidator_tactic.gen_order_id(),
                                signed_qty=-self.positions[symbol].signed_qty
                                )

            self._exec_market_order(order)

        return

    def _exec_order_cancels(self, orders: List[OrderCommon]):
        """
        DEV NOTE: important! the cancel reasons should be set directly in the order!
        DEV NOTE: important! this method should support cancel orders that are not in self.active_orders yet
        :param orders:
        :return:
        """
        for o in orders:
            if o.status_msg is None:
                raise AttributeError("Order cancel must have a reason")

        ids_to_delete = {o.client_id for o in orders}
        self.active_orders = {oid: self.active_orders[oid] for oid in self.active_orders.keys()
                              if oid not in ids_to_delete}

        for order in orders:
            if order.status_msg == OrderCancelReason.invalid_price:
                raise AttributeError('We should always have valid prices')
            should_notify_tactic = order.status_msg != OrderCancelReason.requested_by_user and \
                                   order.status_msg != OrderCancelReason.end_of_sim
            if should_notify_tactic:
                method = self._get_tactic(order.client_id).handle_cancel
                self._queue_append(self.current_timestamp + CANCEL_NOTIF_DELAY, method, order)
                self.n_unsolicited_cancels[order.symbol] += 1

    def _get_tactic(self, order_client_id: str) -> TacticInterface:
        tactic_id = order_client_id.split('_')[0]
        return self.tactics_map[tactic_id]

    def _exec_market_order(self, order: OrderCommon):
        assert order.type == OrderType.Market
        assert order.status == OrderStatus.Pending
        order.status = OrderStatus.New
        tick_size = order.symbol.tick_size

        # TODO: watch for self-cross

        # We only fill at two price levels: best price and second best price
        if order.side() > 0:
            best_price = self.current_quote.ask_price
            second_best = best_price + tick_size
            size = self.current_quote.ask_size
        else:
            best_price = self.current_quote.bid_price
            second_best = best_price - tick_size
            size = self.current_quote.bid_size

        qty_fill1 = min(size, order.leaves_qty)
        qty_fill2 = order.leaves_qty - qty_fill1

        fully_filled = self._fill_order(order=order, qty_to_fill=qty_fill1, price_fill=best_price)
        if not fully_filled:
            assert qty_fill2 > 0
            self._fill_order(order=order, qty_to_fill=qty_fill2, price_fill=second_best)

        return

    def _exec_limit_order(self, order: OrderCommon):
        assert order.type == OrderType.Limit

        bid = self.current_quote.bid_price
        ask = self.current_quote.ask_price
        order_side = order.side()

        self._check_price_sanity(order)

        violated_post_only = order_side > 0 and order.price >= ask or order_side < 0 and order.price <= bid
        if violated_post_only:
            order.status = OrderStatus.Canceled
            order.status_msg = OrderCancelReason.cross_during_post_only
            self._exec_order_cancels([order])
            return

        order.status = OrderStatus.New
        self.active_orders[order.client_id] = order
        order._made_spread = order_side > 0 and order.price > bid or order_side < 0 and order.price < ask

    def _check_price_sanity(self, order: OrderCommon):
        if not self.is_price_sane(order.price, order.symbol):
            raise ValueError('Order with invalid price. {}'.format(order))

    @staticmethod
    def is_price_sane(price: float, symbol: Symbol = Symbol.XBTUSD):
        if symbol == Symbol.XBTUSD:
            two = price + price
            return abs(int(two) - two) < 2e-10
        # TODO: add sanity check for other symbols
        return True

    def _process_trade(self, trade: pd.Series):
        # this method fills against limit orders
        # trade expected fields: ['timestamp', 'symbol', 'side', 'price', 'size', 'tickDirection']

        side_str = trade['side']
        trade_side = +1 if side_str[0] == 'B' else -1
        trade_price = trade['price']
        trade_qty = trade['size']  # unsigned
        trade_sym = Symbol[trade['symbol']]

        for tac in self.tactics_map.values():
            if tac.get_symbol() != trade_sym:
                continue
            trade = Trade(symbol=trade_sym,
                          timestamp=self.current_timestamp,
                          side=+1 if trade['side'][0] == 'B' else -1,
                          price=trade_price,
                          size=trade_qty,
                          tick_direction=TICK_DIRECTION[trade['tickDirection']])

            self._queue_append(self.current_timestamp + WS_DELAY, tac.handle_trade, trade)

        for order in self.active_orders.values():
            if order.type != OrderType.Limit:
                continue
            assert order.is_open()

            order_side = order.side()
            cross = order_side != trade_side and trade_price * trade_side >= order.price * trade_side
            if cross:
                if order._made_spread:
                    # _made_spread means that the order were posted at a better price than bid/ask
                    # liquidity from order has higher priority than the liquidity in the book
                    fill_qty = min(trade_qty, order.leaves_qty)
                else:
                    if order_side > 0:
                        book_price = self.current_quote.bid_price
                        book_size = self.current_quote.bid_size
                    else:
                        book_price = self.current_quote.ask_price
                        book_size = self.current_quote.ask_size

                    is_quote_better_price = book_price * order_side >= order.price * order_side
                    if is_quote_better_price:
                        fill_qty = min(max(trade_qty - book_size, 0), order.leaves_qty)
                    else:
                        fill_qty = min(trade_qty, order.leaves_qty)

                if fill_qty > 0:
                    self._fill_order(order=order,
                                     qty_to_fill=fill_qty,
                                     price_fill=order.price)

        self.active_orders = {o.client_id: o for o in self.active_orders.values() if o.is_open()}
        return

    def _process_quote(self, quote: pd.Series):
        # quote expected columns: ['timestamp', 'symbol', 'bidSize', 'bidPrice', 'askPrice', 'askSize']

        bid = quote['bidPrice']
        ask = quote['askPrice']
        symbol = Symbol[quote['symbol']]
        if len(self.active_orders) > 0:
            for order in self.active_orders.values():  # type: OrderCommon
                if order.type == OrderType.Limit:
                    order_side = order.side()
                    if (order_side > 0 and ask <= order.price) or (order_side < 0 and bid >= order.price):
                        self._fill_order(order=order,
                                         qty_to_fill=order.leaves_qty,
                                         price_fill=order.price)

            self.active_orders = {o.client_id: o for o in self.active_orders.values() if o.is_open()}

        for tac in self.tactics_map.values():
            if tac.get_symbol() != symbol:
                continue
            q = Quote(symbol=symbol,
                      timestamp=self.current_time(),
                      bid_size=quote['bidSize'],
                      bid_price=quote['bidPrice'],
                      ask_size=quote['askSize'],
                      ask_price=quote['askPrice'])

            self._queue_append(self.current_timestamp + WS_DELAY, tac.handle_quote, q)
        return

    def _process_ohlcv(self, ohlcv: pd.Series):
        # ohlcv expect columns: timestamp, symbol, open, high, low, close, size

        ohlcv_view = self.get_candles1m()
        for tactic in self.tactics_map.values():
            method = tactic.handle_1m_candles
            self._queue_append(self.current_timestamp + WS_DELAY, method, ohlcv_view)

    def _fill_order(self, order: OrderCommon, qty_to_fill, price_fill, fee=0.):
        fully_filled = order.fill(qty_to_fill)
        side = order.side()
        fill = Fill(symbol=order.symbol,
                    side=order.side_str(),
                    qty_filled=qty_to_fill,
                    price_fill=price_fill,
                    fill_time=self.current_timestamp,
                    fill_type=FillType.complete if fully_filled else FillType.partial,
                    order_id=order.client_id)
        self._log_fill(fill)
        self.volume[order.symbol] += abs(qty_to_fill) * price_fill
        self.n_fills[order.symbol] += 1

        position = self.positions[order.symbol]
        # we have to fill in 2 steps, in case the size to be filled invert our position
        cur_pos = position.signed_qty
        signed_qty_to_fill = qty_to_fill * side
        qty1 = max(-cur_pos, signed_qty_to_fill) if cur_pos > 0 else min(-cur_pos, signed_qty_to_fill)
        qty2 = signed_qty_to_fill - qty1
        position.update(signed_qty=qty1,
                        price=price_fill,
                        leverage=self.leverage[order.symbol],
                        current_timestamp=self.current_timestamp,
                        fee=fee)
        if abs(qty2) > 0:
            assert not position.is_open
            position.update(signed_qty=qty2,
                            price=price_fill,
                            leverage=self.leverage[order.symbol],
                            current_timestamp=self.current_timestamp,
                            fee=fee)

        method = self._get_tactic(order.client_id).handle_fill
        self._queue_append(self.current_timestamp + WS_DELAY, method, fill)

        return fully_filled

    def get_candles1m(self) -> pd.DataFrame:
        return self.ohlcv.iloc[:max(self.ohlcv_idx, 1)]

    def get_opened_orders(self, symbol: Symbol, client_id_prefix: str) -> OrderContainerType:
        return {o.client_id: o for o in self.active_orders.values()
                if o.symbol == symbol and client_id_prefix == o.client_id[0:len(client_id_prefix)]}

    def send_orders(self, orders: List[OrderCommon]):
        for o in orders:
            if o.status != OrderStatus.Pending:
                raise ValueError("Sending order with status different from Pending: {}".format(o))
            self.n_orders[o.symbol] += 1

        self._queue_append(self.current_timestamp + ORDER_TO_FILL_DELAY, self._send_orders_impl, orders)

    def _send_orders_impl(self, orders: List[OrderCommon]):
        for o in orders:
            o.time_posted = self.current_timestamp
            if o.type == OrderType.Market:
                self._exec_market_order(o)
            elif o.type == OrderType.Limit:
                self._exec_limit_order(o)
            else:
                raise NotImplementedError()

            self._log_order(o)

        pass

    def cancel_orders(self, orders: Union[OrderContainerType, List[OrderCommon], List[str]]):
        self._queue_append(self.current_timestamp + CANCEL_DELAY, self._cancel_orders_impl, orders)

    def _cancel_orders_impl(self, orders: Union[OrderContainerType, List[OrderCommon], List[str]]):
        # The reason to split this function from cancel_orders is to simulate the delay in the cancels
        ids = get_orders_id(orders)
        for i in ids:
            order = self.active_orders.get(i)
            if order:
                order.status = OrderStatus.Canceled
                order.status_msg = OrderCancelReason.requested_by_user
                self._exec_order_cancels([order])
            else:
                raise AttributeError("Invalid order id {}. Opened orders are: {}".format(i, [k for k in self.active_orders]))

    def current_time(self) -> pd.Timestamp:
        return self.current_timestamp

    def get_quote(self, symbol: Symbol) -> Quote:
        return self.current_quote

    def get_position(self, symbol: Symbol) -> PositionInterface:
        return self.positions[symbol]

    def get_pnl_history(self, symbol: Symbol) -> List[float]:
        return self.pnl_history[symbol]

    def set_leverage(self, symbol: Symbol, leverage: float):
        if not (0.01 <= leverage <= 100.):
            raise ValueError("Invalid leverage {}, allowed range is 0.01 <= leverage <= 100".format(leverage))
        self.leverage[symbol] = leverage

    def get_balance_xbt(self) -> float:
        return self.xbt_balance

    def is_open(self):
        return not self.finished

    def print_progress(self, current_ts: pd.Timestamp):
        progress = min(1.0, (current_ts - self.begin_timestamp) / (self.end_timestamp - self.begin_timestamp))
        if progress < 1.:
            sys.stdout.write("progress: %.4f%%   \r" % (100 * progress))
        else:
            print("progress: 100%                 ")
        sys.stdout.flush()


def read_timeseries(filename: str, cols: list, begin: pd.Timestamp, end: pd.Timestamp, end_inclusive: bool = False):
    table = pd.read_csv(filename)
    table.set_index('timestamp', inplace=True)
    table = table[cols]
    table.index = pd.DatetimeIndex(table.index)
    if end is None:
        if end_inclusive is True:
            raise ValueError("If 'end_inclusive' is set, 'end' must be specified (not None)")
        else:
            end_inclusive = True
    md_range = (table.index[0], table.index[-1])
    begin = begin if begin else table.index[0]
    end = end if end else table.index[-1]
    if end_inclusive:
        table = table[(table.index >= begin) & (table.index <= end)]
    else:
        table = table[(table.index >= begin) & (table.index < end)]

    if len(table) == 0:
        raise ValueError("Specified begin {} and end time {} produces empty market data (range {})".format(
            begin, end, md_range
        ))

    if abs(table.index[0] - begin) > pd.Timedelta('1 min'):
        raise ValueError('Provided begin_timestamp is too early for the market data (more the 1 min of inactivity)')

    if abs(table.index[-1] - end) > pd.Timedelta('1 min'):
        raise ValueError('Provided end_timestamp is too early for the market data (more the 1 min of inactivity)')

    return table


def read_market_data(args):
    start = time.time()
    ohlcv = read_timeseries(filename=args.ohlcv_file,
                            cols=['symbol', 'open', 'high', 'low', 'close', 'size'],
                            begin=args.begin,
                            end=args.end)
    print('ohlcv : n_rows={}, time reading: {}s'.format(len(ohlcv), '%.2f' % (time.time() - start)))

    start = time.time()
    trades = read_timeseries(filename=args.trades_file,
                             cols=['symbol', 'side', 'price', 'size', 'tickDirection'],
                             begin=args.begin,
                             end=args.end)
    print('trades: n_rows={}, time reading: {}s'.format(len(trades), '%.2f' % (time.time() - start)))

    start = time.time()
    quotes = read_timeseries(filename=args.quotes_file,
                             cols=['symbol', 'bidSize', 'bidPrice', 'askPrice', 'askSize'],
                             begin=args.begin,
                             end=args.end)
    print('quotes: n_rows={}, time reading: {}s'.format(len(quotes), '%.2f' % (time.time() - start)))

    return ohlcv, trades, quotes


def test_all():
    pass


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

    ohlcv, trades, quotes = read_market_data(args)
    tactics = [t() for t in args.tactics]

    sim = SimExchangeBitMex(args.begin,
                            args.end,
                            initial_balance=1,
                            ohlcv=ohlcv,
                            trades=trades,
                            quotes=quotes,
                            log_dir=args.log_dir,
                            tactics=tactics,
                            tactic_prefs=args.pref)

    sim.run_sim()

    return 0


if __name__ == "__main__":
    sys.exit(main())
