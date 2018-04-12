import logging
import math

import numpy
from enum import Enum
from pandas import Series, Timedelta, Timestamp
from sympy import sign

from exchange_interface import ExchangeCommon
from fill import FillType, Fill
from orders import Orders, OrderCommon, OrderType, OrderCancelReason
from position import Position
from simcandles import SimCandles
from autologging import logged


# @logged
class TacticInterface:
    def __init__(self):
        pass

    def init(self, exchange, preferences):
        # type: (ExchangeCommon, dict) -> None
        raise AttributeError("interface class")

    def get_symbol(self):
        # type: () -> Enum
        raise AttributeError("interface class")

    def handle_candles(self, exchange):
        # type: (ExchangeCommon) -> None
        raise AttributeError("interface class")

    def handle_submission_error(self, failed_order):
        # type: (OrderCommon) -> None
        raise AttributeError("interface class")

    def handle_fill(self, exchange, fill):
        # type: (ExchangeCommon, Fill) -> None
        raise AttributeError("interface class")

    def handle_cancel(self, exchange, order):
        # type: (ExchangeCommon, OrderCommon) -> None
        raise AttributeError("interface class")

    def id(self):
        # type: () -> str
        return self.__class__.__name__


# @logged
class TacticBitEwm(TacticInterface):
    def __init__(self, product_id):
        TacticInterface.__init__(self)
        self.product_id = product_id  # type: Enum

        self.opened_orders = Orders()
        self.position = None  # type: Position
        self.last_activity_time = None  # type: Timestamp
        self.multiplier = 100.
        self.span = 20
        self.greediness = 0.1  # 0. -> post order at EMA, 1. -> post order at EMA + std
        self.qty_to_trade = 0.3  # 1. -> qty=price, 0. -> qty=0
        self.no_activity_tol = 180  # num of minutes of no activity tolerance.
        # After that, non filled orders are cancelled

        self.last_ema_std = (float('nan'), float('nan'))
        self.last_fill = None

    def init(self, exchange, preferences):
        # type: (ExchangeCommon, dict) -> None
        exchange.set_leverage(self.product_id, self.multiplier)
        self.last_activity_time = exchange.current_time()

        if 'span' in preferences:
            self.span = int(preferences['span'])
        if 'greediness' in preferences:
            self.greediness = float(preferences['greediness'])
        if 'qty_to_trade' in preferences:
            self.qty_to_trade = float(preferences['qty_to_trade'])

    def get_symbol(self):
        # type: () -> Enum
        return self.product_id

    def has_position(self):
        return not self.position.is_closeable()

    def send_order(self, exchange, order, n_try=1):
        # type: (ExchangeCommon, OrderCommon) -> bool
        # return True if failed
        for i in range(n_try):
            orders_to_send = Orders()
            orders_to_send.add(order)
            if not exchange.post_orders(orders_to_send):
                if order.is_open():
                    self.opened_orders.merge(orders_to_send)
                self.last_activity_time = order.time_posted
                return False
        return True

    def handle_cancel(self, exchange, order):
        # type: (ExchangeCommon, OrderCommon) -> None
        self.position = exchange.get_position(self.product_id)  # type: Position
        if self.position.is_closeable() or \
                order.status_msg == OrderCancelReason.liquidation or \
                order.status_msg == OrderCancelReason.end_of_sim or \
                order.status_msg == OrderCancelReason.requested_by_user:
            return
        if not self.send_order(exchange, OrderCommon(symbol=order.symbol,
                                                     signed_qty=order.signed_qty,
                                                     price=order.price,
                                                     type=order.type,
                                                     tactic=self)):
            # self.__log.info("canceling order")
            1
        self.opened_orders.drop_closed_orders()
        raise ValueError()  # test

    def handle_fill(self, exchange, fill):
        # type: (ExchangeCommon, Fill) -> None
        qty_filled = fill.qty
        order = fill.order
        try:
            assert order == self.opened_orders[order.id]
        except KeyError:
            return
        self.position = exchange.get_position(self.product_id)  # type: Position
        self.last_activity_time = fill.fill_time
        # self.__log.info("handling fill")

        if fill.fill_type == FillType.complete or order.is_fully_filled():
            self.opened_orders.clean_filled(order)
            if not (fill.fill_type == FillType.complete and order.is_fully_filled()):
                raise AttributeError("fill status is {} and order.is_fully_filled is {}"
                                     .format(fill.fill_type == FillType.complete, order.is_fully_filled()))

        if not self.has_position():
            assert order.is_fully_filled()
            exchange.cancel_orders(self.opened_orders)
            self.handle_candles(exchange)
            return

        reduced_position = not self.position.does_reduce_position(-qty_filled)

        if not reduced_position:
            # create a profit order to reduce position
            ema, std = self.last_ema_std
            price = (ema + sign(qty_filled) * std) * self.greediness + ema * (1. - self.greediness)

            order_to_send = OrderCommon(symbol=self.product_id,
                                        signed_qty=-qty_filled,
                                        price=price,
                                        type=OrderType.limit,
                                        tactic=self)
            failed = self.send_order(exchange, order_to_send)
            if failed:
                order_to_send = OrderCommon(symbol=self.product_id,
                                            signed_qty=-qty_filled,
                                            type=OrderType.market,
                                            tactic=self)
                self.send_order(exchange, order_to_send, 10)

    def handle_candles(self, exchange):
        #self.__log.info("handling candles")

        # type: (ExchangeCommon, float, float) -> None
        candles1m = exchange.get_candles1m()  # type: SimCandles
        price = exchange.current_price()
        assert price == candles1m.at(-1)['close']

        self.position = exchange.get_position(self.product_id)  # type: Position

        # warming up
        if candles1m.size() < self.span:
            return

        self.opened_orders.drop_closed_orders()

        if self.opened_orders.size() == 0 and not self.position.is_closeable():
            raise AttributeError("Invalid state. We have a position of {} but there is not opened order to reduce this"
                                 " position. Probably a tactic logic error.".format(self.position.position()))

        if self.opened_orders.size() > 0:
            if exchange.current_time() - self.last_activity_time > Timedelta(minutes=self.no_activity_tol):
                if self.close_position_if_no_loss(exchange, price):
                    assert self.opened_orders.size() == 0
            return

        df = candles1m.data['close']  # type: Series
        ema = df.ewm(span=self.span).mean()[-1]
        std = df.tail(self.span).std()

        if price - ema > std:
            should_trade = -1  # sell
        elif ema - price > std:
            should_trade = +1  # buy
        else:
            should_trade = 0

        if not should_trade:
            # print("NOT GOOD EMA ... " + str((str(abs(price - ema)), str(std))) + "                    ")
            return

        self.last_ema_std = (ema, std)

        order_to_send = OrderCommon(symbol=self.product_id,
                                    signed_qty=should_trade * math.floor(price * self.qty_to_trade),
                                    price=price,
                                    type=OrderType.limit,
                                    tactic=self)

        if self.send_order(exchange, order_to_send) and not self.has_position():
            exchange.cancel_orders(Orders({order_to_send.id: order_to_send}))
        else:
            # self.__log.info(" -- sending order " + str(order_to_send))
            1

    # this method doesn't drop closed orders
    def close_position_if_no_loss(self, exchange, current_price):
        if not self.position.is_open():
            return False
        pnl = self.position.hypothetical_close_at_price(current_price, self.multiplier)
        if pnl >= 0:
            exchange.cancel_orders(self.opened_orders, drop_canceled=True)
            if not self.position.is_closeable():
                order_to_send = OrderCommon(symbol=self.product_id,
                                            signed_qty=-self.position.net_qty(),
                                            type=OrderType.market,
                                            tactic=self)
                if not self.send_order(exchange, order_to_send):
                    self.last_activity_time = order_to_send.time_posted
            # self.__log.info("closing position due to inactivity " + str([str(o) + '\n' for o in self.opened_orders]))
            return True
        return False


# Same as TacticBitEwm, but this variation temporarily stop trading if it's losing too much
# @logged
class TacticBitEwmWithStop(TacticInterface):
    def __init__(self, product_id):
        TacticInterface.__init__(self)
        self.product_id = product_id  # type: Enum

        self.opened_orders = Orders()
        self.position = None  # type: Position
        self.last_activity_time = None  # type: Timestamp
        self.multiplier = 100.
        self.span = 20
        self.greediness = 0.1  # 0. -> post order at EMA, 1. -> post order at EMA + std
        self.qty_to_trade = 0.3  # 1. -> qty=price, 0. -> qty=0
        self.no_activity_tol = 180  # num of minutes of no activity tolerance.
        # After that, non filled orders are cancelled
        self.loss_limit = 3  # number of time we tolerate loss, after that trading is suspended by self.span times

        self.last_ema_std = (float('nan'), float('nan'))
        self.last_fill = None

    def init(self, exchange, preferences):
        # type: (ExchangeCommon, dict) -> None
        exchange.set_leverage(self.product_id, self.multiplier)
        self.last_activity_time = exchange.current_time()

        if 'span' in preferences:
            self.span = int(preferences['span'])
        if 'greediness' in preferences:
            self.greediness = float(preferences['greediness'])
        if 'qty_to_trade' in preferences:
            self.qty_to_trade = float(preferences['qty_to_trade'])
        if 'loss_limit' in preferences:
            self.loss_limit = int(preferences['loss_limit'])

    def get_symbol(self):
        # type: () -> Enum
        return self.product_id

    def has_position(self):
        return not self.position.is_closeable()

    def send_order(self, exchange, order, n_try=1):
        # type: (ExchangeCommon, OrderCommon) -> bool
        # return True if failed
        for i in range(n_try):
            orders_to_send = Orders()
            orders_to_send.add(order)
            if not exchange.post_orders(orders_to_send):
                if order.is_open():
                    self.opened_orders.merge(orders_to_send)
                self.last_activity_time = order.time_posted
                return False
        return True

    def handle_cancel(self, exchange, order):
        # type: (ExchangeCommon, OrderCommon) -> None
        self.position = exchange.get_position(self.product_id)  # type: Position
        if self.position.is_closeable() or \
                order.status_msg == OrderCancelReason.liquidation or \
                order.status_msg == OrderCancelReason.end_of_sim or \
                order.status_msg == OrderCancelReason.requested_by_user:
            return
        if not self.send_order(exchange, OrderCommon(symbol=order.symbol,
                                                     signed_qty=order.signed_qty,
                                                     price=order.price,
                                                     type=order.type,
                                                     tactic=self)):
            # self.__log.info("canceling order")
            1
        self.opened_orders.drop_closed_orders()
        raise ValueError()  # test

    def handle_fill(self, exchange, fill):
        # type: (ExchangeCommon, Fill) -> None
        qty_filled = fill.qty
        order = fill.order
        try:
            assert order == self.opened_orders[order.id]
        except KeyError:
            return
        self.position = exchange.get_position(self.product_id)  # type: Position
        self.last_activity_time = fill.fill_time
        # self.__log.info("handling fill")

        if fill.fill_type == FillType.complete or order.is_fully_filled():
            self.opened_orders.clean_filled(order)
            if not (fill.fill_type == FillType.complete and order.is_fully_filled()):
                raise AttributeError("fill status is {} and order.is_fully_filled is {}"
                                     .format(fill.fill_type == FillType.complete, order.is_fully_filled()))

        if not self.has_position():
            assert order.is_fully_filled()
            exchange.cancel_orders(self.opened_orders)
            self.handle_candles(exchange)
            return

        reduced_position = not self.position.does_reduce_position(-qty_filled)

        if not reduced_position:
            # create a profit order to reduce position
            ema, std = self.last_ema_std
            price = (ema + sign(qty_filled) * std) * self.greediness + ema * (1. - self.greediness)

            order_to_send = OrderCommon(symbol=self.product_id,
                                        signed_qty=-qty_filled,
                                        price=price,
                                        type=OrderType.limit,
                                        tactic=self)
            failed = self.send_order(exchange, order_to_send)
            if failed:
                order_to_send = OrderCommon(symbol=self.product_id,
                                            signed_qty=-qty_filled,
                                            type=OrderType.market,
                                            tactic=self)
                self.send_order(exchange, order_to_send, 10)

    def is_losing_too_much(self, exchange):
        last_losses = [i.realized_pnl <= 0 for i in exchange.get_closed_positions(self.product_id)[-self.loss_limit:]]
        if len(last_losses) == sum(last_losses) and \
                exchange.current_time() - self.last_activity_time < Timedelta(minutes=self.span):
            return True
        return False

    def handle_candles(self, exchange):
        # self.__log.info("handling candles")

        # type: (ExchangeCommon, float, float) -> None
        candles1m = exchange.get_candles1m()  # type: SimCandles
        price = exchange.current_price()
        assert price == candles1m.at(-1)['close']

        self.position = exchange.get_position(self.product_id)  # type: Position

        # warming up
        if candles1m.size() < self.span:
            return

        self.opened_orders.drop_closed_orders()

        if self.opened_orders.size() == 0 and not self.position.is_closeable():
            raise AttributeError("Invalid state. We have a position of {} but there is not opened order to reduce this"
                                 " position. Probably a tactic logic error.".format(self.position.position()))

        if self.opened_orders.size() > 0:
            if exchange.current_time() - self.last_activity_time > Timedelta(minutes=self.no_activity_tol):
                if self.close_position_if_no_loss(exchange, price):
                    assert self.opened_orders.size() == 0
            return

        reverse_logic = self.is_losing_too_much(exchange)
        if reverse_logic:
            return
            #print("STOPPPPPPPPPPPP LOSING IT!!!")
            #return

        df = candles1m.data['close']  # type: Series
        ema = df.ewm(span=self.span).mean()[-1]
        std = df.tail(self.span).std()

        if price - ema > std:
            should_trade = -1  # sell
        elif ema - price > std:
            should_trade = +1  # buy
        else:
            should_trade = 0

        should_trade = should_trade * (1-2*reverse_logic)
        if not should_trade:
            # print("NOT GOOD EMA ... " + str((str(abs(price - ema)), str(std))) + "                    ")
            return

        self.last_ema_std = (ema, std)

        order_to_send = OrderCommon(symbol=self.product_id,
                                    signed_qty=should_trade * math.floor(price * self.qty_to_trade),
                                    price=price,
                                    type=OrderType.limit,
                                    tactic=self)

        if self.send_order(exchange, order_to_send) and not self.has_position():
            exchange.cancel_orders(Orders({order_to_send.id: order_to_send}))
        else:
            # self.__log.info(" -- sending order " + str(order_to_send))
            1

    # this method doesn't drop closed orders
    def close_position_if_no_loss(self, exchange, current_price):
        if not self.position.is_open():
            return False
        pnl = self.position.hypothetical_close_at_price(current_price, self.multiplier)
        if pnl >= 0:
            exchange.cancel_orders(self.opened_orders, drop_canceled=True)
            if not self.position.is_closeable():
                order_to_send = OrderCommon(symbol=self.product_id,
                                            signed_qty=-self.position.net_qty(),
                                            type=OrderType.market,
                                            tactic=self)
                if not self.send_order(exchange, order_to_send):
                    self.last_activity_time = order_to_send.time_posted
            # self.__log.info("closing position due to inactivity " + str([str(o) + '\n' for o in self.opened_orders]))
            return True
        return False
