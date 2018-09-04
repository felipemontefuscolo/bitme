# NOTES:
# timestamps are of type pd.Timestamp
# side are of type str ('buy' or 'sell')
import argparse
import os
import shutil
import sys
from collections import defaultdict, deque
from typing import List, Union

import pandas as pd

from api import PositionInterface
from api.exchange_interface import ExchangeInterface
from api.symbol import Symbol
from common import OrderType, OrderContainerType, OrderCommon, OrderCancelReason
from common.quote import Quote
from sim.liquidator import Liquidator
from tactic.TacticMarketOrderTest import TacticMarketOrderTest
from .sim_stats import SimSummary


# from autologging import logged


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
        self.log_dir = log_dir

        self.ohlcv = ohlcv
        self.trades = trades
        self.candle_num = 0  # 0 to len(self.ohlcv)-1
        self.tick_num = 0  # 0 to len(self.trades)-1

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

        self.queue = deque()  # essentially queue of send_orders and cancel_orders

        self.active_orders = dict()  # type: OrderContainerType

        self.current_quote = Quote()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

    def get_candles1m(self) -> pd.DataFrame:
        return self.ohlcv[self.candle_num]

    def get_opened_orders(self, symbol: Symbol, client_id_prefix: str) -> OrderContainerType:
        return {o.client_id: o for o in self.active_orders.values()
                if o.symbol == symbol and client_id_prefix == o.client_id[0:len(client_id_prefix)]}

    def send_orders(self, orders: List[OrderCommon]):
        self.queue.append([self._send_orders_impl, orders])

    def _send_orders_impl(self, orders: List[OrderCommon]):
        pass

    def cancel_orders(self, orders: Union[OrderContainerType, List[OrderCommon], List[str]]):
        self.queue.append([self._cancel_orders_impl, orders])

    def _cancel_orders_impl(self, orders: Union[OrderContainerType, List[OrderCommon], List[str]]):
        pass

    def current_time(self) -> pd.Timestamp:
        return self.trades.index[self.tick_num]

    def get_quote(self, symbol: Symbol) -> Quote:
        pass

    def get_position(self, symbol: Symbol) -> PositionInterface:
        pass

    def get_pnl_history(self, symbol: Symbol) -> List[float]:
        pass

    def set_leverage(self, symbol: Symbol, leverage: float):
        pass

    def get_balance_xbt(self) -> float:
        pass

    def is_open(self):
        return self.tick_num < len(self.trades)

    def print_progress(self):
        sys.stdout.write(
            "progress: %d out of %d (%.4f%%)   \r" %
            (self.tick_num, len(self.md_trades), 100 * float(self.tick_num) / len(self.md_trades)))
        sys.stdout.flush()

    def advance_time(self, print_progress: bool = True):
        if print_progress:
            self.print_progress()

        if self._is_last_tick():
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

    def _is_last_tick(self):
        return self.tick_num == len(self.trades) - 1

    def _compute_quote(self, symbol: Symbol, tick: int, trades: pd.DataFrame) -> Quote:
        if tick >= len(trades):
            raise ValueError('Invalid tick: {}, trades len: {}'.format(tick, len(trades)))

        max_ahead_lookup = 50

        ts = trades.index[tick]
        trade = trades.iloc[tick]
        tick_size = symbol.value['tick_size']

        if trade['side'][0] == 'S':
            bid_price = trade['price']
            bid_size = trade['size']
            ask_price = bid_price + tick_size

            j = tick + 1
            while j < tick + max_ahead_lookup:
                next_trade = trades.iloc[j]
                if next_trade['price'] == bid_price and trade['side'][0] == 'S':
                    bid_size += next_trade['size']
                    j += 1
                else:
                    break

            while j < tick + max_ahead_lookup:
                next_trade = trades.iloc[j]
                if next_trade['side'][0] == 'B':
                    ask_price = max(ask_price, next_trade['price'])
                    break
                elif :
                    bid_size += next_trade['size']


        elif trade['side'][0] == 'B':
            ask_price = trade['price']
            bid_price = ask_price - tick_size
            for j in range(tick + 1, len(trades)):
                if trades.iloc[j]['side'][0] == 'S':
                    bid_price = min(bid_price, trades.iloc[j]['price'])
                    break
        else:
            raise AttributeError('Unknown trade size {}'.format(trade['side']))

        assert bid_price < ask_price


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

    ohlcv = read_timeseries(filename=args.ohlcv_file,
                            cols=['symbol', 'open', 'high', 'low', 'close', 'size'],
                            begin=args.begin,
                            end=args.end)

    trades = read_timeseries(filename=args.trades_file,
                             cols=['symbol', 'side', 'price', 'size', 'tickDirection'],
                             begin=args.begin,
                             end=args.end)

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
