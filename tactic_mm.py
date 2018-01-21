import math

from pandas import Series, Timedelta
from sympy import sign

from exchange_interface import ExchangeCommon
from fill import FillType, Fill
from orders import Orders, OrderCommon, OrderType, OrderCancelReason
from position import Position
from simcandles import SimCandles


class TacticInterface:
    def __init__(self):
        pass

    def init(self, exchange):
        # type: (ExchangeCommon) -> None
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


class TacticBitEwm(TacticInterface):
    def __init__(self, product_id):
        TacticInterface.__init__(self)
        self.product_id = product_id

        self.opened_orders = Orders()
        self.position = None  # type: Position
        self.span = 20
        self.greediness = 0.1  # 0. -> post order at EMA, 1. -> post order at EMA + std
        self.no_activity_tol = 60  # num of minutes of no activity tolerance.
        # After that, non filled orders are cancelled

        self.last_ema_std = (float('nan'), float('nan'))
        self.last_fill = None

    def init(self, exchange):
        # type: (ExchangeCommon) -> None
        exchange.set_leverage(self.product_id, 100.)

    def has_position(self):
        return not self.position.is_closeable()

    def send_order(self, exchange, order, n_try=1):
        # type: (ExchangeCommon, OrderCommon) -> bool
        # return True if failed
        for i in range(n_try):
            orders_to_send = Orders()
            orders_to_send.add(order)
            if not exchange.post_orders(orders_to_send):
                self.opened_orders.merge(orders_to_send)
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
        self.send_order(exchange, OrderCommon(symbol=order.symbol,
                                              signed_qty=order.signed_qty,
                                              price=order.price,
                                              type=order.type,
                                              tactic=self))
        self.opened_orders.drop_closed_orders()

    def handle_fill(self, exchange, fill):
        # type: (ExchangeCommon, Fill) -> None
        qty_filled = fill.qty
        order = self.opened_orders[fill.order_id]  # type: OrderCommon
        self.position = exchange.get_position(self.product_id)  # type: Position

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
            if self.position.is_closeable():
                assert self.opened_orders.size() == 1
                last_order = self.opened_orders.data.values()[0]  # type: OrderCommon
                if exchange.current_time() - last_order.time_posted > Timedelta(minutes=self.no_activity_tol):
                    exchange.cancel_orders(Orders({last_order.id: last_order}))
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
                                    signed_qty=should_trade * math.floor(price),
                                    price=price,
                                    type=OrderType.limit,
                                    tactic=self)

        if self.send_order(exchange, order_to_send) and not self.has_position():
            exchange.cancel_orders(Orders({order_to_send.id: order_to_send}))
