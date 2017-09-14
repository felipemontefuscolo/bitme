import sys
from candles import Candles
from tactic_mm import TacticTest
from utils import Hour, to_iso_utc
from orders import LimitOrder, Orders, to_str, TWOPLACES, EIGHPLACES
import math


class Fill:
    def __init__(self, side, size, price, order_type, fill_time):
        self.side = side
        self.size = size
        self.price = price
        self.order_type = order_type
        self.fill_time = fill_time

    def __repr__(self):
        return str(self.to_json())

    def __str__(self):
        return str(self.to_json())

    def to_json(self):
        params = {
            'time': to_iso_utc(self.fill_time),
            'side': self.side,
            'price': to_str(self.price, TWOPLACES),  # USD
            'size': to_str(self.size, EIGHPLACES),  # BTC
            'type': self.order_type
        }
        return params


def main():
    print("starting sim")
    candles = Candles.fromfilename('/Users/felipe/bitme/data/test')
    # candles = Candles.fromfilename('/Users/felipe/bitme/data/data1s.csv')

    tac = TacticTest('BTC-USD')
    active_orders = Orders()

    fills = []
    position_btc = 0
    position_usd = 100

    for t, low, high, open_p, close_p, volume in candles:
        if t is None:
            # no activity
            continue

        orders_to_send = tac.handle_candles(candles, active_orders, position_btc, position_usd)
        orders_to_send.remove_no_fund_orders(position_btc, position_usd)
        active_orders.merge(orders_to_send)

        # fill sim
        for order in active_orders.data:
            # special case

            vol_fill = 0
            is_sell = order.side[0] == 's'
            is_buy = not is_sell
            if high == low:
                if (is_sell and order.price < high) or (is_buy and order.price > low):
                    vol_fill = 0.5 * volume
            else:
                if is_sell and order.price < high:
                    vol_fill = ((high - order.price) / (high - low)) * volume
                elif is_buy and order.price > low:
                    vol_fill = ((low - order.price) / (low - high)) * volume

            if vol_fill > 0:
                filled = order.fill(vol_fill)
                fills += [Fill(order.side, filled, order.price, order.order_type, t + 1)]

                filled = filled if is_buy else -filled
                position_btc += filled
                if position_btc < 0:
                    raise RuntimeError(
                        "Position should never be negative (it is %s). Last fill was %s" % (position_btc, filled))
                position_usd -= filled * order.price


                    # print("vol_fill = " + str(vol_fill))
                    # active_orders.printf()

        active_orders.clean_filled()

    # active_orders.printf()
    print_fills = True
    if print_fills:
        print("Fills:")
        for fill in fills:
            print(str(fill.to_json()))

    print("position btc = " + str(position_btc))
    print("position usd = " + str(position_usd))
    print("optimist realized profit = " + str(position_usd + position_btc * close_p))

    return 0


if __name__ == "__main__":
    sys.exit(main())
