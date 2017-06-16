from orders import Orders
from utils import Min, Hour


class TacticMM:
    def __init__(self, product_id):
        self.product_id = product_id
        pass

    def handle_candles(self, candles1s, current_time, orders):
        # should return orders to send

        t = current_time
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
        orders_to_send.post_limit_order('buy', 123.44444, 2, self.product_id, current_time)
        orders_to_send.post_limit_order('sell', 123.666666, 1.1111, self.product_id, current_time)

        return orders_to_send
