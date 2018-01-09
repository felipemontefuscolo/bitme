import math

import pandas as pd
from pandas import Series
from sympy import sign

from candles import Candles
from orders import Orders, OrderCommon, OrderType
from sim import ExchangeCommon, SimExchangeBitMex
from simcandles import SimCandles


class TacticInterface:
    def __init__(self):
        pass

    def init(self, exchange):
        # type: (ExchangeCommon) -> None
        raise AttributeError("interface class")

    def handle_candles(self, exchange):
        # type: (ExchangeCommon) -> Orders
        raise AttributeError("interface class")

    def handle_submission_error(self, failed_order):
        # type: (OrderCommon) -> Orders
        raise AttributeError("interface class")

    def handle_fill(self, candles, order_id, size_filled):
        # type: (Candles, str, float) -> Orders
        raise AttributeError("interface class")

    def id(self):
        # type: () -> str
        return self.__class__.__name__


class TacticForBitMex1(TacticInterface):
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

    def handle_candles(self, exchange):
        # type: (SimExchangeBitMex, float, float) -> Orders
        candles1m = exchange.get_candles1m()
        price = exchange.current_price()
        assert price == candles1m.at(-1)['close']

        if self.opened_orders.size() > 2:
            raise ValueError("should not have more than 2 orders placed")
        self.opened_orders.drop_closed_orders()

        if self.opened_orders.size() > 0:
            return Orders()

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
            return Orders()

        orders_to_send = Orders()

        if direction < 0:  # go long
            recover_price = 0.4  # the lower, the more conservative

            buy = OrderCommon(symbol=self.product_id,
                              signed_qty=math.floor(price),
                              price=price,
                              type=OrderType.limit)
            orders_to_send.add(buy)

            sell = OrderCommon(symbol=self.product_id,
                               signed_qty=-math.floor(price),
                               price=price * (1. - recover_price) + candles1m.at(-trend_size).open * recover_price,
                               type=OrderType.limit)
            orders_to_send.add(sell)
        else:  # go short
            recover_price = 0.5
            sell = OrderCommon(symbol=self.product_id,
                               signed_qty=-math.floor(price),
                               price=price,
                               type=OrderType.limit)
            orders_to_send.add(sell)

            buy = OrderCommon(symbol=self.product_id,
                              signed_qty=math.floor(price),
                              price=price * (1. - recover_price) + candles1m.at(-trend_size).open * recover_price,
                              type=OrderType.limit)
            orders_to_send.add(buy)

        if exchange.post_orders(orders_to_send):
            exchange.cancel_orders(orders_to_send, drop_canceled=True)
        else:
            self.opened_orders.merge(orders_to_send)


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

    def handle_candles(self, exchange):
        # type: (SimExchangeBitMex, float, float) -> Orders
        candles1m = exchange.get_candles1m()
        price = exchange.current_price()
        assert price == candles1m.at(-1)['close']

        if self.opened_orders.size() > 2:
            raise ValueError("should not have more than 2 orders placed")
        self.opened_orders.drop_closed_orders()

        if self.opened_orders.size() > 0:
            return Orders()

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
            return Orders()

        orders_to_send = Orders()

        if direction < 0:  # go long
            recover_price = 0.4  # the lower, the more conservative

            buy = OrderCommon(symbol=self.product_id,
                              signed_qty=math.floor(price),
                              price=price,
                              type=OrderType.limit)
            orders_to_send.add(buy)

            sell = OrderCommon(symbol=self.product_id,
                               signed_qty=-math.floor(price),
                               price=price * (1. - recover_price) + candles1m.at(-trend_size).open * recover_price,
                               type=OrderType.limit)
            orders_to_send.add(sell)
        else:  # go short
            recover_price = 0.5
            sell = OrderCommon(symbol=self.product_id,
                               signed_qty=-math.floor(price),
                               price=price,
                               type=OrderType.limit)
            orders_to_send.add(sell)

            buy = OrderCommon(symbol=self.product_id,
                              signed_qty=math.floor(price),
                              price=price * (1. - recover_price) + candles1m.at(-trend_size).open * recover_price,
                              type=OrderType.limit)
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
        self.position = 0
        self.span = 20
        self.greediness = 0.1  # 0. -> post order at EMA, 1. -> post order at EMA + std

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

    def handle_candles(self, exchange):
        # type: (SimExchangeBitMex, float, float) -> Orders
        candles1m = exchange.get_candles1m()  # type: SimCandles
        price = exchange.current_price()
        assert price == candles1m.at(-1)['close']

        # warming up
        if candles1m.size() < self.span:
            return Orders()

        if self.opened_orders.size() > 2:
            raise ValueError("should not have more than 2 orders placed")
        self.opened_orders.drop_closed_orders()

        if self.opened_orders.size() > 0:
            return Orders()

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
            return Orders()

        orders_to_send = Orders()

        if direction < 0:  # go long
            recover_price = 0.4  # the lower, the more conservative

            buy = OrderCommon(symbol=self.product_id,
                              signed_qty=math.floor(price),
                              price=price,
                              type=OrderType.limit)
            orders_to_send.add(buy)

            sell = OrderCommon(symbol=self.product_id,
                               signed_qty=-math.floor(price),
                               price=price * (1. - recover_price) + candles1m.at(-trend_size).open * recover_price,
                               type=OrderType.limit)
            orders_to_send.add(sell)
        else:  # go short
            recover_price = 0.5
            sell = OrderCommon(symbol=self.product_id,
                               signed_qty=-math.floor(price),
                               price=price,
                               type=OrderType.limit)
            orders_to_send.add(sell)

            buy = OrderCommon(symbol=self.product_id,
                              signed_qty=math.floor(price),
                              price=price * (1. - recover_price) + candles1m.at(-trend_size).open * recover_price,
                              type=OrderType.limit)
            orders_to_send.add(buy)

        if exchange.post_orders(orders_to_send):
            exchange.cancel_orders(orders_to_send, drop_canceled=True)
        else:
            self.opened_orders.merge(orders_to_send)