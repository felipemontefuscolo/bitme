# NOTES:
# timestamps are of type pd.Timestamp
# side are of type str ('buy' or 'sell')
import argparse
import os
import shutil
import sys
import time
from collections import defaultdict, deque
from typing import List, Union, Dict

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
from tactic.TacticMarketOrderTest import TacticMarketOrderTest
from .sim_stats import SimSummary


# from autologging import logged

# TODO: prevent send orders where you would be liquidated immediately
# TODO; prevent send orders if you have no funds


# import logging


def get_args(input_args=None):
    files_required = '--files' not in (input_args if input_args else sys.argv)

    parser = argparse.ArgumentParser(description='Simulation')
    parser.add_argument('--ohlcv-file', type=str, help='csv filename with candles data', required=files_required)
    parser.add_argument('--trades-file', type=str, help='csv filename with trades data', required=files_required)
    parser.add_argument('--quotes-file', type=str, help='csv filename with quotes data', required=files_required)
    parser.add_argument('--files', type=str, help='template path to the ohlcv, trades and quotes files (use %TYPE%)')
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

    return args


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

        assert ohlcv is not None
        assert trades is not None
        assert quotes is not None
        self.ohlcv = ohlcv  # type: pd.DataFrame
        self.trades = trades  # type: pd.DataFrame
        self.quotes = quotes  # type: pd.DataFrame
        self.ohlcv_idx = 0
        self.trade_idx = 0
        self.quote_idx = 0

        self.begin_timestamp = min(trades.index[0], quotes.index[0], ohlcv.index[0])  # type: pd.Timestamp
        self.end_timestamp = min(trades.index[-1], quotes.index[-1], ohlcv.index[-1])  # type: pd.Timestamp

        if self.end_timestamp <= self.begin_timestamp:
            raise ValueError("end_timestamp less or equal begin_timestamp. Please check arguments and make sure "
                             "the market data first event occurs before end time")

        self.current_timestamp = self.begin_timestamp  # type: pd.Timestamp

        self.liquidator_tactic = Liquidator()
        self.tactics_map = {t.id(): t for t in [self.liquidator_tactic] + tactics}  # type: Dict[str, TacticInterface]
        ss = [tac.get_symbol() for tac in tactics]
        zz = set(ss)
        if len(zz) != len(ss):
            raise ValueError("Tactics trading same symbol is not allowed.")

        self.n_cancels = defaultdict(int)
        self.n_liquidations = defaultdict(int)  # Symbol -> int
        self.cum_pnl = defaultdict(float)  # type: Dict[Symbol, float]
        self.pnl_history = defaultdict(list)  # type: Dict[Symbol, List[float]]
        self.positions = {s: PositionSim(s, self._log_and_update_pnl) for s in
                          self.SYMBOLS}  # type: Dict[Symbol, PositionSim]
        self.leverage = {s: 1.0 for s in self.SYMBOLS}  # type: Dict[Symbol, float]

        # should be None until end of sim
        self.summary = None  # type: SimSummary

        self.queue = deque()  # queue of tuples (timestamp, method, method_argument) to be called

        self.active_orders = dict()  # type: OrderContainerType

        self.current_quote = None  # type: Quote

    def _init_files(self, log_dir):
        self.fills_file = open(os.path.join(log_dir, 'fills.csv'), 'w')
        self.orders_file = open(os.path.join(log_dir, 'orders.csv'), 'w')
        self.pnl_file = open(os.path.join(log_dir, 'pnl.csv'), 'w')
        self.candles_file = open(os.path.join(log_dir, 'candles.csv'), 'w')

        self.fills_file.write(Fill.get_header() + '\n')
        self.orders_file.write(OrderCommon.get_header() + '\n')
        self.pnl_file.write('time,symbol,pnl,cum_pnl\n')
        self.candles_file.write(','.join(['timestamp'] + OHLCV_COLUMNS) + '\n')

    def _init_tactics(self):
        for tac in self.tactics_map.values():
            tac.initialize(self, self.tactic_prefs)

    def _close_files(self):
        assert self.finished
        self.fills_file.close()
        self.orders_file.close()
        self.pnl_file.close()
        self.candles_file.close()

    def _log_fill(self, fill: Fill):
        self.fills_file.write(fill.to_line() + '\n')

    def _log_order(self, order: OrderCommon):
        self.orders_file.write(order.to_line() + '\n')

    def _log_candle(self, vals: list):
        self.candles_file.write(','.join([str(i) for i in vals]) + '\n')

    def _log_and_update_pnl(self, position: PositionSim):
        timestamp = self.current_timestamp
        pnl = position.realized_pnl
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

        print("Running sims from {} to {}".format(self.begin_timestamp, self.end_timestamp))
        while not self.finished:
            self._process_queue_until(self.current_timestamp)
            self._advance_timestamp(print_progress=True)
            pass
        self._process_queue_until(execute_all=True)

    def end_main_loop(self):
        for symbol in list(Symbol):
            self._exec_liquidation(symbol, reason=OrderCancelReason.end_of_sim)
        self._close_files()

        print("Total pnl (in XBT): ")
        for k, v in self.cum_pnl.items():
            print("{}: {}".format(k, v))
        print("sim time: {}s".format(time.time() - self.sim_start_time))

    def _process_queue_until(self, end_inclusive: pd.Timestamp = None, execute_all = False):
        while len(self.queue) > 0 and (execute_all or self.queue[0][0] <= end_inclusive):
            task = self.queue.popleft()
            method = task[1]
            args = task[2]  # if we ever need to change the number of arguments to more than one, you can use *args
            method(args)
        pass

    def _advance_timestamp(self, print_progress: bool = True):
        # DEV NOTE: advance time stamp and put market data in the queue
        if print_progress:
            self.print_progress()

        def get_table_ts_at_idx(table, idx):
            return table.index[min(idx, len(table) - 1)]

        next_trade_ts = get_table_ts_at_idx(self.trades, self.trade_idx)
        next_quote_ts = get_table_ts_at_idx(self.quotes, self.quote_idx)
        next_ohlcv_ts = get_table_ts_at_idx(self.ohlcv, self.ohlcv_idx)

        next_ts = min(next_trade_ts, next_quote_ts, next_ohlcv_ts)

        if next_ts >= self.end_timestamp:
            self.finished = True
            return

        def add_data_to_queue_until(end_ts_inclusive, method, table, idx):
            while idx < len(table) and table.index[idx] <= end_ts_inclusive:
                self.queue.append((end_ts_inclusive, method, table.iloc[idx]))
                idx += 1
            return idx

        quote_row = self.quotes.iloc[self.quote_idx]
        self.current_quote = Quote(symbol=Symbol[quote_row['symbol']],
                                   timestamp=next_quote_ts,
                                   bid_size=quote_row['bidSize'],
                                   bid_price=quote_row['bidPrice'],
                                   ask_size=quote_row['askSize'],
                                   ask_price=quote_row['askPrice'])

        self.trade_idx = add_data_to_queue_until(next_ts, self._process_trade, self.trades, self.trade_idx)
        self.quote_idx = add_data_to_queue_until(next_ts, self._process_quote, self.quotes, self.quote_idx)
        self.ohlcv_idx = add_data_to_queue_until(next_ts, self._process_ohlcv, self.ohlcv, self.ohlcv_idx)

        self.current_timestamp = next_ts
        return

    def _exec_liquidation(self, symbol: Symbol, reason = OrderCancelReason.liquidation):
        orders_to_cancel = [o for o in self.active_orders.values() if o.symbol == symbol]
        for order in orders_to_cancel:
            order.status_msg = reason
        self._exec_order_cancels(orders_to_cancel)

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
                self.queue.append((self.current_timestamp, method, order))

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

            self.queue.append((self.current_timestamp, tac.handle_trade, trade))

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
        for order in self.active_orders.values():  # type: OrderCommon
            if order.type == OrderType.Limit:
                order_side = order.side()
                if (order_side > 0 and ask <= order.price) or (order_side < 0 and bid >= order.price):
                    self._fill_order(order=order,
                                     qty_to_fill=order.leaves_qty,
                                     price_fill=order.price)

        self.active_orders = {o.client_id: o for o in self.active_orders.values() if o.is_open()}
        return

    def _process_ohlcv(self, ohlcv: pd.Series):
        # ohlcv expect columns: timestamp, symbol, open, high, low, close, size

        ohlcv_view = self.get_candles1m()
        for tactic in self.tactics_map.values():
            method = tactic.handle_1m_candles
            self.queue.append((self.current_timestamp, method, ohlcv_view))

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
        self.positions[order.symbol].update(signed_qty=qty_to_fill * side,
                                            price=order.price,
                                            leverage=self.leverage[order.symbol],
                                            current_timestamp=self.current_timestamp,
                                            fee=fee)

        method = self._get_tactic(order.client_id).handle_fill
        self.queue.append((self.current_timestamp, method, fill))

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
        self.queue.append((self.current_timestamp, self._send_orders_impl, orders))

    def _send_orders_impl(self, orders: List[OrderCommon]):
        for o in orders:
            if o.type == OrderType.Market:
                self._exec_market_order(o)
            elif o.type == OrderType.Limit:
                self._exec_limit_order(o)
            else:
                raise NotImplementedError()
        pass

    def cancel_orders(self, orders: Union[OrderContainerType, List[OrderCommon], List[str]]):
        self.queue.append((self.current_timestamp, self._cancel_orders_impl, orders))

    def _cancel_orders_impl(self, orders: Union[OrderContainerType, List[OrderCommon], List[str]]):
        # The reason to split this function from cancel_orders is to simulate the delay in the cancels
        ids = get_orders_id(orders)
        for i in ids:
            order = self.active_orders.get(i)
            if order:
                order.status = OrderStatus.Canceled
                order.status_msg = OrderCancelReason.requested_by_user
        self._exec_order_cancels(orders)

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

    def print_progress(self):
        current_ts = self.current_timestamp if self.current_timestamp else self.begin_timestamp
        progress = min(1.0, (current_ts - self.begin_timestamp) / (self.end_timestamp - self.begin_timestamp))
        sys.stdout.write("progress: %.4f%%   \r" % (100 * progress))
        sys.stdout.flush()

    def _is_last_tick(self):
        return self.trade_idx == len(self.trades) - 1


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
    begin = begin if begin else table.index[0]
    end = end if end else table.index[-1]
    if end_inclusive:
        table = table[(table.index >= begin) & (table.index <= end)]
    else:
        table = table[(table.index >= begin) & (table.index < end)]
    return table


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

    # defin here the tactic you want to activate
    tactics = [TacticMarketOrderTest(2, 2)]

    start = time.time()
    ohlcv = read_timeseries(filename=args.ohlcv_file,
                            cols=['symbol', 'open', 'high', 'low', 'close', 'size'],
                            begin=args.begin,
                            end=args.end)
    print('ohlcv : n_rows={}, range=[{},{}], time reading: {}s'.format(len(ohlcv), ohlcv.index[0], ohlcv.index[-1],
                                                                       time.time() - start))

    start = time.time()
    trades = read_timeseries(filename=args.trades_file,
                             cols=['symbol', 'side', 'price', 'size', 'tickDirection'],
                             begin=args.begin,
                             end=args.end)
    print('trades: n_rows={}, range=[{},{}], time reading: {}s'.format(len(trades), trades.index[0], trades.index[-1],
                                                                       time.time() - start))

    start = time.time()
    quotes = read_timeseries(filename=args.quotes_file,
                             cols=['symbol', 'bidSize', 'bidPrice', 'askPrice', 'askSize'],
                             begin=args.begin,
                             end=args.end)
    print('quotes: n_rows={}, range=[{},{}], time reading: {}s'.format(len(quotes), quotes.index[0], quotes.index[-1],
                                                                       time.time() - start))

    sim = SimExchangeBitMex(initial_balance=1,
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
