from candles import last_non_none
from orders import Orders
from utils import Min, Hour


class TacticTest:
    def __init__(self, product_id):
        self.product_id = product_id
        self.round = 0
        pass

    def handle_candles(self, candles1s, active_orders, position_coin, position_usd):
        # should return orders to send

        # if self.round > 0:
        #     return Orders()
        # self.round = 1

        t = candles1s.ts_l[-1]
        # c1h = candles1s.get_candles(Hour().to_sec(1), t - Hour().to_sec(24), t)
        # c15m = candles1s.get_candles(Min().to_sec(15), t - Hour().to_sec(6), t)
        # c1m = candles1s.get_candles(Min().to_sec(1), t - Min().to_sec(24), t)

        c1h = candles1s.get_candles(Hour(1).to_sec(), t - Hour(24).to_sec(), t)
        c15m = candles1s.get_candles(Min(15).to_sec(), t - Hour(6).to_sec(), t)
        c1m = candles1s.get_candles(Min(1).to_sec(), t - Min(24).to_sec(), t)

        # c1h.printf()
        # c15m.printf()
        # c1m.printf()

        orders_to_send = Orders()
        orders_to_send.post_limit_order('buy', 178.02, position_usd / 178.02 - 0.00001, self.product_id, t)
        orders_to_send.post_limit_order('sell', 179.02, 5, self.product_id, t)

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
        # should return orders to send

        t = last_non_none(candles1s.ts_l)
        price = last_non_none(candles1s.close_l)

        c6h = candles1s.get_candles(Hour(6).to_sec(), t - Hour(24).to_sec()-1, t)

        orders_to_send = Orders()

        if len(c6h.ts_l) < 4:
            # warm up period
            return Orders()

        # going_up = sum([open_p >= close_p for t, low, high, open_p, close_p, volume in c6h])
        #if going_up < 2:
        #    # market not favorable
        #    return Orders()

        if active_orders.size() > 2:
            raise ValueError("should have more than 2 orders placed")

        num_buys = sum([order.size[0] == 'b' for order in active_orders])
        num_sells = sum([order.size[0] == 's' for order in active_orders])

        if num_sells > 0:
            # just wait sell be filled
            return Orders()

        if num_buys == 0 and position_usd >= price and position_coin < 1:  # send a buy
            c5m = candles1s.get_candles(Min(5).to_sec(), t - Min(15).to_sec() - 1, t)
            if all([open_p > close_p for t, low, high, open_p, close_p, volume in c5m]):
                orders_to_send.post_limit_order('buy', price - 0.1, 1, self.product_id, t)
                self.last_filled_buy_price = price - 0.1
                return orders_to_send

        if position_coin >= 1:
            orders_to_send.post_limit_order('sell', max(self.last_filled_buy_price, price) + 1, 1, self.product_id, t)
            return orders_to_send

        return Orders()