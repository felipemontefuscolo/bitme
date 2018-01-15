import math

from pandas import Series
from sympy import sign

from orders import Orders, OrderCommon, OrderType, OrderCancelReason
from sim import ExchangeCommon, SimExchangeBitMex, Position
from fill import FillType, Fill
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


# This tactic trades at nc candles trend only, when the price change is pc%.
# It trade on pairs
class TacticForBitMex2(TacticInterface):
    def __init__(self, product_id):
        TacticInterface.__init__(self)
        self.product_id = product_id

        self.opened_orders = Orders()

    def init(self, exchange):
        # type: (SimExchangeBitMex) -> None
        exchange.set_leverage(self.product_id, 100.)

    @staticmethod
    def get_last_trend(candles1m):
        """ :return direction and duration """
        direction = sign(candles1m.at(-1).close - candles1m.at(-1).open)

        candles_len = candles1m.size()

        trend_size = 0  # = number of candles sticks with same trend
        while abs(trend_size + 1) < candles_len and \
                sign(candles1m.at(-trend_size - 1).close - candles1m.at(-trend_size - 1).open) == direction:
            trend_size += 1

        price_change = abs((candles1m.last_price() - candles1m.at(-trend_size).open)) / \
                       candles1m.at(-trend_size).open

        return direction, trend_size, price_change

    def handle_fill(self, exchange, fill):
        raise AttributeError("Please implement me!!!!!")

    def handle_candles(self, exchange):
        # type: (SimExchangeBitMex, float, float) -> None
        candles1m = exchange.get_candles1m()
        price = exchange.current_price()
        assert price == candles1m.at(-1)['close']

        if self.opened_orders.size() > 2:
            raise ValueError("should not have more than 2 orders placed")
        self.opened_orders.drop_closed_orders()

        if self.opened_orders.size() > 0:
            return

        num_buys = sum([order.is_buy() for order in self.opened_orders])
        num_sells = sum([order.is_sell() for order in self.opened_orders])

        direction, trend_size, price_change = self.get_last_trend(candles1m)

        should_trade = False
        if direction > 0:
            if trend_size > 2 and price_change > 0.003499:
                should_trade = True
        else:
            if trend_size > 1 and price_change > 0.002499:
                should_trade = True

        if not should_trade:
            return

        orders_to_send = Orders()

        if direction < 0:  # go long
            recover_price = 0.4  # the lower, the more conservative

            buy = OrderCommon(symbol=self.product_id,
                              signed_qty=math.floor(price),
                              price=price,
                              type=OrderType.limit,
                              tactic=self)
            orders_to_send.add(buy)

            sell = OrderCommon(symbol=self.product_id,
                               signed_qty=-math.floor(price),
                               price=price * (1. - recover_price) + candles1m.at(-trend_size).open * recover_price,
                               type=OrderType.limit,
                               tactic=self)
            orders_to_send.add(sell)
        else:  # go short
            recover_price = 0.5
            sell = OrderCommon(symbol=self.product_id,
                               signed_qty=-math.floor(price),
                               price=price,
                               type=OrderType.limit,
                               tactic=self)
            orders_to_send.add(sell)

            buy = OrderCommon(symbol=self.product_id,
                              signed_qty=math.floor(price),
                              price=price * (1. - recover_price) + candles1m.at(-trend_size).open * recover_price,
                              type=OrderType.limit,
                              tactic=self)
            orders_to_send.add(buy)

        if exchange.post_orders(orders_to_send):
            exchange.cancel_orders(orders_to_send, drop_canceled=True)
        else:
            self.opened_orders.merge(orders_to_send)


class TacticBitEwm(TacticInterface):
    def __init__(self, product_id):
        TacticInterface.__init__(self)
        self.product_id = product_id

        self.opened_orders = Orders()
        self.position = None  # type: Position
        self.span = 20
        self.greediness = 0.1  # 0. -> post order at EMA, 1. -> post order at EMA + std

        self.last_ema_std = (float('nan'), float('nan'))
        self.last_fill = None

    def init(self, exchange):
        # type: (SimExchangeBitMex) -> None
        exchange.set_leverage(self.product_id, 100.)

    def has_position(self):
        return not self.position.is_closeable()

    def send_order(self, exchange, order):
        # type: (SimExchangeBitMex, OrderCommon) -> bool
        # return True if failed
        orders_to_send = Orders()
        orders_to_send.add(order)
        if not exchange.post_orders(orders_to_send):
            self.opened_orders.merge(orders_to_send)
            return False
        else:
            return True

    def handle_cancel(self, exchange, order):
        # type: (SimExchangeBitMex, OrderCommon) -> None
        self.position = exchange.get_position(self.product_id)  # type: Position
        if self.position.is_closeable() or \
                order.status_msg == OrderCancelReason.liquidation or \
                order.status_msg == OrderCancelReason.end_of_sim or \
                order.status_msg == OrderCancelReason.cancel_requested:
            return
        self.send_order(exchange, OrderCommon(symbol=order.symbol,
                                              signed_qty=order.signed_qty,
                                              price=order.price,
                                              type=order.type,
                                              tactic=self))

    def handle_fill(self, exchange, fill):
        # type: (SimExchangeBitMex, Fill) -> None
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
            failed = self.send_order(exchange, OrderCommon(symbol=self.product_id,
                                                           signed_qty=-qty_filled,
                                                           price=price,
                                                           type=OrderType.limit,
                                                           tactic=self))
            #if not failed:
            #    print ("POSTED ORDER THRU HANDLE FILL")

    def handle_candles(self, exchange):
        # type: (SimExchangeBitMex, float, float) -> None
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
            return

        df = candles1m.data['close']  # type: Series
        ema = df.ewm(span=self.span).mean()[-1]
        std = df.std()

        if price - ema > std:
            should_trade = -1  # sell
        elif ema - price > std:
            should_trade = +1  # buy
        else:
            should_trade = 0

        if not should_trade:
            return

        self.last_ema_std = (ema, std)

        order_to_send = OrderCommon(symbol=self.product_id,
                                    signed_qty=should_trade * math.floor(price),
                                    price=price,
                                    type=OrderType.limit,
                                    tactic=self)

        if self.send_order(exchange, order_to_send) and not self.has_position():
            exchange.cancel_orders(Orders({order_to_send.id: order_to_send}))
        #else:
        #    print("POSTED ORDER THRU HANDLE CANDLE")
