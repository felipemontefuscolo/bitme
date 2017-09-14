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
