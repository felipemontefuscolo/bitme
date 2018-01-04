import math

import pandas as pd
from sympy import sign

from candles import Candles
from orders import Orders, OrderCommon, OrderType
from sim import ExchangeCommon, SimExchangeBitMex


class TaticInterface:
    def __init__(self):
        pass

    def handle_candles(self, exchange, positions, available_balance):
        # type: (ExchangeCommon, dict, float) -> Orders
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


# This tactic only places one order at a time, alternating buy and sell
# always buy & sell 1 bitcoin
# This tactic only works if
#   initial_position_btc = 0
#   initial_position_usd >= current price of bitcoin
class TacticForBitMex1:
    def __init__(self, product_id):
        self.product_id = product_id
        self.round = 0
        self.last_filled_buy_price = 0

        self.active_orders = Orders()

        #  parameters to be adjusted:
        self.fixed_size = 1000  # usd contracts
        pass

    def handle_candles(self, exchange, positions, available_balance):
        # type: (ExchangeCommon, dict, float) -> Orders

        position_instrument = positions[self.product_id]
        candles1m = exchange.get_candles1m()
        t = candles1m.last_timestamp()
        price = candles1m.last_price()

        c5m = candles1m.sample_candles(pd.Timedelta(minutes=5), t - pd.Timedelta(minutes=15), t)

        # warm up period
        if c5m.size() < 4:
            return Orders()

        orders_to_send = Orders()

        # going_up = sum([open_p >= close_p for t, low, high, open_p, close_p, volume in c6h])
        # if going_up < 2:
        #    # market not favorable
        #    return Orders()

        if self.active_orders.size() > 2:
            raise ValueError("should have more than 2 orders placed")

        num_buys = sum([order.side[0] == 'b' for order in self.active_orders])
        num_sells = sum([order.side[0] == 's' for order in self.active_orders])

        if num_sells > 0:
            # just wait sell be filled
            return Orders()

        if num_buys == 0 and available_balance >= self.fixed_size * price and position_instrument < self.fixed_size:  # send a buy
            dec = (c5m.data.open > c5m.data.close).values
            if all(dec):
                orders_to_send.post_limit_order(side='buy',
                                                price=price - 0.5,
                                                size=self.fixed_size,
                                                product_id=self.product_id,
                                                time_posted=t)
                self.last_filled_buy_price = price - 0.5
                return orders_to_send

        if position_instrument >= self.fixed_size:
            orders_to_send.post_limit_order(side='sell',
                                            price=max(self.last_filled_buy_price, price) + 1,
                                            size=self.fixed_size,
                                            product_id=self.product_id,
                                            time_posted=t)
            return orders_to_send

        return Orders()


# This tactic trades at nc candles trend only, when the price change is pc%.
# It trade on pairs
class TacticForBitMex2(TaticInterface):
    def __init__(self, product_id):
        TaticInterface.__init__(self)
        self.product_id = product_id

        self.opened_orders = Orders()

    def init(self, exchange):
        # type: (SimExchangeBitMex) -> None
        exchange.set_leverage(self.product_id, 50.)

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

    def handle_candles(self, exchange, position_instrument, available_balance_btc):
        # type: (SimExchangeBitMex, float, float) -> Orders
        candles1m = exchange.get_candles1m()
        price = exchange.current_price()

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
