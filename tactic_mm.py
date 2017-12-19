from candles import Candles
from orders import Orders, OrderCommon
from utils import Min, Hour
import pandas as pd


class TaticInterface:
    def __init__(self):
        pass

    def handle_candles(self, candles, active_orders, position_coin, position_usd):
        # type: (Candles, Orders, float, float) -> Orders
        raise AttributeError("interface class")

    def handle_submission_error(self, failed_order):
        # type: (OrderCommon) -> Orders
        raise AttributeError("interface class")

    def handle_fill(self, candles, order_id, size_filled):
        # type: (Candles, Orders, float, float) -> Orders
        raise AttributeError("interface class")


class TacticTest:
    def __init__(self, product_id):
        self.product_id = product_id
        self.round = 0
        pass

    def handle_candles(self, candles, active_orders, position_coin, position_usd):
        # type: (Candles, Orders, float, float) -> Orders
        orders_to_send = Orders()
        return orders_to_send


# This tactic only places one order at a time, alternating buy and sell
# always buy & sell 1 bitcoin
# This tactic only works if
#   initial_position_btc = 0
#   initial_position_usd >= current price of bitcoin
class Tactic1:
    def __init__(self, product_id):
        self.product_id = product_id
        self.round = 0
        self.last_filled_buy_price = 0

        #  parameters to be adjusted:
        self.fixed_size = 1  # buy or sell

        pass

    def handle_candles(self, candles1s, active_orders, position_coin, position_usd):
        # type: (Candles, Orders, float, float) -> Orders

        t = candles1s.last_timestamp()
        price = candles1s.last_price()

        c5m = candles1s.sample_candles(pd.Timedelta(minutes=5), t - pd.Timedelta(minutes=15), t)

        # warm up period
        if c5m.size() < 4:
            return Orders()

        orders_to_send = Orders()

        # going_up = sum([open_p >= close_p for t, low, high, open_p, close_p, volume in c6h])
        # if going_up < 2:
        #    # market not favorable
        #    return Orders()

        if active_orders.size() > 2:
            raise ValueError("should have more than 2 orders placed")

        num_buys = sum([order.side[0] == 'b' for order in active_orders])
        num_sells = sum([order.side[0] == 's' for order in active_orders])

        if num_sells > 0:
            # just wait sell be filled
            return Orders()

        if num_buys == 0 and position_usd >= self.fixed_size * price and position_coin < self.fixed_size:  # send a buy
            dec = (c5m.data.open > c5m.data.close).values
            if all(dec):
                orders_to_send.post_limit_order(side='buy',
                                                price=price - 0.5,
                                                size=self.fixed_size,
                                                product_id=self.product_id,
                                                time_posted=t)
                self.last_filled_buy_price = price - 0.5
                return orders_to_send

        if position_coin >= self.fixed_size:
            orders_to_send.post_limit_order(side='sell',
                                            price=max(self.last_filled_buy_price, price) + 1,
                                            size=self.fixed_size,
                                            product_id=self.product_id,
                                            time_posted=t)
            return orders_to_send

        return Orders()


# This tactic trades at nc candles trend only, when the price change is pc%.
# It trade on pairs
class Tactic2(TaticInterface):
    def __init__(self, product_id):
        TaticInterface.__init__(self)
        self.product_id = product_id
        self.last_filled_buy_price = 0

        self.opened_orders = Orders()

        #  parameters to be adjusted:
        self.fixed_size = 1  # buy or sell
        self.n_trend_candles = 3
        self.price_change = 0.013  # (0, 1), trade when price change is this much
        self.recover_price = 0.5  # (0, 1)

    def handle_candles(self, candles1s, active_orders, position_coin, position_usd):
        # type: (Candles, Orders, float, float) -> Orders

        t = candles1s.last_timestamp()
        price = candles1s.last_price()

        if active_orders.size() > 2:
            raise ValueError("should have more than 2 orders placed")

        num_buys = sum([order.side[0] == 'b' for order in active_orders])
        num_sells = sum([order.side[0] == 's' for order in active_orders])

        if num_sells > 0:
            # just wait sell be filled
            return Orders()

        trend_size = 0
        while candles1s.at(-trend_size - 1).open > candles1s.at(-trend_size - 1).close:
            trend_size += 1

        price_change = abs((candles1s.last_price() - candles1s.at(-trend_size).open)) / \
                       candles1s.at(-trend_size).open

        should_trade = num_buys == 0 and trend_size >= self.n_trend_candles and price_change >= self.price_change and \
                     position_usd >= self.fixed_size * price

        if not should_trade:
            return Orders()

        orders_to_send = Orders()

        orders_to_send.post_limit_order(side='buy',
                                        price=price - 0.5,
                                        size=self.fixed_size,
                                        product_id=self.product_id,
                                        time_posted=t)
        self.last_filled_buy_price = price - 0.5

        orders_to_send.post_limit_order(side='sell',
                                        price=candles1s.last_price() * (1. - self.recover_price) +
                                              candles1s.at(-trend_size).open * self.recover_price,
                                        size=self.fixed_size,
                                        product_id=self.product_id,
                                        time_posted=t)

        return orders_to_send
