from candles import Candles
from orders import Orders
from utils import Min, Hour
import pandas as pd


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
#   initial_position_usd = 100
class Tactic1:
    def __init__(self, product_id):
        self.product_id = product_id
        self.round = 0
        self.last_filled_buy_price = 0
        pass

    def handle_candles(self, candles1s, active_orders, position_coin, position_usd):
        # type: (Candles, Orders, float, float) -> Orders

        t = candles1s.last_timestamp()
        price = candles1s.last_price()

        c5m = candles1s.sample_candles(pd.Timedelta(minutes=5), t - pd.Timedelta(minutes=15), t)

        # warm up period
        aa = c5m.size()
        if c5m.size() < 4:
            return Orders()

        orders_to_send = Orders()

        # going_up = sum([open_p >= close_p for t, low, high, open_p, close_p, volume in c6h])
        #if going_up < 2:
        #    # market not favorable
        #    return Orders()

        if active_orders.size() > 2:
            raise ValueError("should have more than 2 orders placed")

        num_buys = sum([order.side[0] == 'b' for order in active_orders])
        num_sells = sum([order.side[0] == 's' for order in active_orders])

        if num_sells > 0:
            # just wait sell be filled
            return Orders()

        if num_buys == 0 and position_usd >= price and position_coin < 1:  # send a buy
            dec = (c5m.data.open > c5m.data.close).values
            if all(dec):
                orders_to_send.post_limit_order('buy', price - 0.1, 1, self.product_id, t)
                self.last_filled_buy_price = price - 0.1
                return orders_to_send

        if position_coin >= 1:
            orders_to_send.post_limit_order('sell', max(self.last_filled_buy_price, price) + 1, 1, self.product_id, t)
            return orders_to_send

        return Orders()


# to test market orders
class TacticLearner():
    def __init__(self, product_id):
        self.product_id = product_id
        self.round = 0
        pass