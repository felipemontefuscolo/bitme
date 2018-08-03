import math

import pandas as pd
from sympy import sign
from typing import List, Dict

from sim.position_sim import PositionSim
from api.exchange_interface import ExchangeInterface
from api.position_interface import PositionInterface
from api.symbol import Symbol
from common.fill import FillType, Fill
from common.order import OrderCancelReason, OrderCommon, OrderType, OrderStatus, \
    drop_orders, drop_closed_orders_dict
from tactic.tactic_tools import does_reduce_position
from tactic.tactic_interface import TacticInterface


# Same as TacticBitEwm, but this variation temporarily stop trading if it's losing too much
# @logged
class TacticBitEwmWithStop(TacticInterface):

    def __init__(self, product_id):
        TacticInterface.__init__(self)
        self.product_id = product_id  # type: Symbol

        self.opened_orders = dict()  # type: Dict[str, OrderCommon]
        self.last_activity_time = None  # type: pd.Timestamp
        self.multiplier = 100.
        self.span = 20
        self.greediness = 0.1  # 0. -> post order at EMA, 1. -> post order at EMA + std
        self.qty_to_trade = 0.3  # 1. -> qty=price, 0. -> qty=0
        self.no_activity_tol = 180  # num of minutes of no activity tolerance.
        # After that, non filled orders are cancelled
        self.loss_limit = 3  # number of time we tolerate loss, after that trading is suspended by self.span times

        self.last_ema_std = (float('nan'), float('nan'))
        self.last_fill = None

        self.exchange = None

    def id(self) -> str:
        return 'TBEWS'

    def init(self, exchange: ExchangeInterface, preferences: dict):
        self.exchange = exchange
        self.exchange.set_leverage(self.product_id, self.multiplier)
        self.last_activity_time = self.exchange.current_time()

        if 'span' in preferences:
            self.span = int(preferences['span'])
        if 'greediness' in preferences:
            self.greediness = float(preferences['greediness'])
        if 'qty_to_trade' in preferences:
            self.qty_to_trade = float(preferences['qty_to_trade'])
        if 'loss_limit' in preferences:
            self.loss_limit = int(preferences['loss_limit'])

    def handle_submission_error(self, failed_order: OrderCommon) -> None:
        raise AttributeError("Not implemented yet")

    def get_symbol(self) -> Symbol:
        return self.product_id

    def send_order(self, exchange: ExchangeInterface, order: OrderCommon, n_try=1) -> bool:
        # return True if failed
        for i in range(n_try):
            orders_sent = exchange.post_orders([order])
            if len(orders_sent) != 0:
                order_sent = next(iter(orders_sent))  # type: OrderCommon
                if order_sent.is_open():
                    self.opened_orders[order_sent.id] = order_sent
                self.last_activity_time = order.time_posted
                return False
        return True

    def handle_cancel(self, order: OrderCommon):
        position = self.exchange.get_position(self.product_id)  # type: PositionInterface
        if not position.is_open:
            return

        self.opened_orders = drop_closed_orders_dict(self.opened_orders)

    def handle_fill(self, fill: Fill):
        qty_filled = fill.qty
        order = fill.order
        try:
            assert order.id in self.opened_orders
        except KeyError:
            return
        position = self.exchange.get_position(self.product_id)  # type: PositionSim
        self.last_activity_time = fill.fill_time
        # self.__log.info("handling fill")

        if fill.fill_type == FillType.complete or order.status == OrderStatus.Filled:
            self.opened_orders = drop_closed_orders_dict(self.opened_orders)
            if not (fill.fill_type == FillType.complete and order.status == OrderStatus.Filled):
                raise AttributeError("fill status is {} and order.is_fully_filled is {}"
                                     .format(fill.fill_type == FillType.complete, order.status == OrderStatus.Filled))

        if not position.is_open:
            assert order.status == OrderStatus.Filled
            self.opened_orders = drop_orders(self.opened_orders, self.exchange.cancel_orders(self.opened_orders))
            self.handle_1m_candles(self.exchange.get_candles1m())
            return

        reduced_position = not does_reduce_position(-qty_filled, position)

        if not reduced_position:
            # create a profit order to reduce position
            ema, std = self.last_ema_std
            price = (ema + sign(qty_filled) * std) * self.greediness + ema * (1. - self.greediness)

            order_to_send = OrderCommon(symbol=self.product_id,
                                        signed_qty=-qty_filled,
                                        price=price,
                                        type=OrderType.Limit,
                                        tactic=self)
            failed = self.send_order(self.exchange, order_to_send)
            if failed:
                order_to_send = OrderCommon(symbol=self.product_id,
                                            signed_qty=-qty_filled,
                                            type=OrderType.Market,
                                            tactic=self)
                self.send_order(self.exchange, order_to_send, 10)

    def is_losing_too_much(self, exchange):
        closed_position = exchange.get_closed_positions(self.product_id)
        if not closed_position:
            return False
        last_losses = [bool(i.realized_pnl <= 0) for i in closed_position[-self.loss_limit:]]
        if len(last_losses) == sum(last_losses) and \
                exchange.current_time() - self.last_activity_time < pd.Timedelta(minutes=self.span):
            return True
        return False

    def handle_1m_candles(self, candles1m: pd.DataFrame):
        # self.__log.info("handling candles")

        price = self.exchange.get_tick_info()['last']

        position = self.exchange.get_position(self.product_id)  # type: PositionInterface

        # warming up
        if len(candles1m) < self.span:
            return

        self.opened_orders = drop_closed_orders_dict(self.opened_orders)

        if len(self.opened_orders) == 0 and position and position.is_open:
            raise AttributeError("Invalid state. We have a position of {} but there is not opened order to reduce this"
                                 " position. Probably a tactic logic error.".format(position.current_qty))

        if len(self.opened_orders) > 0:
            if self.exchange.current_time() - self.last_activity_time > pd.Timedelta(minutes=self.no_activity_tol):
                if self.close_position_if_no_loss(self.exchange, price, position):
                    assert len(self.opened_orders) == 0
            return

        reverse_logic = self.is_losing_too_much(self.exchange)
        if reverse_logic:
            return
            # print("STOPPPPPPPPPPPP LOSING IT!!!")
            # return

        df = candles1m['close']  # type: pd.Series
        ema = df.ewm(span=self.span).mean()[-1]
        std = df.tail(self.span).std()

        if price - ema > std:
            should_trade = -1  # sell
        elif ema - price > std:
            should_trade = +1  # buy
        else:
            should_trade = 0

        should_trade = should_trade * (1 - 2 * reverse_logic)
        if not should_trade:
            # print("NOT GOOD EMA ... " + str((str(abs(price - ema)), str(std))) + "                    ")
            return

        self.last_ema_std = (ema, std)

        order_to_send = OrderCommon(symbol=self.product_id,
                                    signed_qty=should_trade * math.floor(price * self.qty_to_trade),
                                    price=price,
                                    type=OrderType.Limit,
                                    tactic=self)

        if self.send_order(self.exchange, order_to_send) and not position.is_open:
            self.exchange.cancel_orders([order_to_send])
        else:
            # self.__log.info(" -- sending order " + str(order_to_send))
            1

    # this method doesn't drop closed orders
    def close_position_if_no_loss(self, exchange, current_price, position):
        if not position.is_open:
            return False
        if sign(current_price - position.break_even_price) == position.side:
            self.opened_orders = drop_orders(self.opened_orders, self.exchange.cancel_orders(self.opened_orders))
            if position.is_open:
                order_to_send = OrderCommon(symbol=self.product_id,
                                            signed_qty=-position.current_qty,
                                            type=OrderType.Market,
                                            tactic=self)
                if not self.send_order(self.exchange, order_to_send):
                    self.last_activity_time = order_to_send.time_posted
            # self.__log.info("closing position due to inactivity " + str([str(o) + '\n' for o in self.opened_orders]))
            return True
        return False
