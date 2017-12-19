# NOTES:
# timestamps are of type pd.Timestamp
# side are of type str ('buy' or 'sell')


import sys

from orders import LimitOrder, to_str, TWOPLACES, EIGHPLACES, MarketOrder
from simcandles import SimCandles
from tactic_mm import *


class Fill:
    def __init__(self, order, size_filled, price_fill, fill_time):
        # type: (OrderCommon, int, float, pd.Timestamp) -> None
        self.order_id = order.id
        self.side = order.side
        self.size = size_filled
        self.price = price_fill
        self.order_type = order.type
        self.fill_time = fill_time

    def __repr__(self):
        return str(self.to_json())

    def __str__(self):
        return str(self.to_line())

    def to_json(self):
        params = {
            'time': str(self.fill_time),
            'order_id': self.order_id,
            'side': self.side,
            'price': to_str(self.price, TWOPLACES),  # USD
            'size': to_str(self.size, EIGHPLACES),  # BTC
            'type': self.order_type
        }
        return params

    def to_line(self):
        return ','.join([
            str(self.fill_time),
            str(self.order_id),
            str(self.side),
            str(to_str(self.price, TWOPLACES)),  # USD
            str(to_str(self.size, EIGHPLACES)),  # BTC
            str(self.order_type)
        ])

    @staticmethod
    def get_header():
        return "time,order_id,side,price,size,type"


# try to simulate cancels -- for example, cancel invalid order
def simulate_cancel_orders(orders_to_send, position_btc, position_usd):
    # type: (Orders, float, float) -> Orders
    orders_to_send.remove_no_fund_orders(position_btc, position_usd)
    return orders_to_send


def main():
    print("starting sim")
    # candles = Candles.fromfilename('/Users/felipe/bitme/data/test')

    #file_name = '/Users/felipe/bitme/data/data1s.csv'
    file_name = '/Users/felipe/bitme/data/bitmex_1day.csv'
    #file_name = '/Users/felipe/bitme/data/test'
    product_id = 'BTC-USD'
    candles = SimCandles(file_name)
    opened_orders = Orders()
    #tac = TacticTest(product_id)
    tac = Tactic2(product_id)

    initial_position_btc = 0
    initial_position_usd = 20000
    close_p = -999999999

    fills = []
    fills_file = open('/Users/felipe/bitme/output.fills', 'w')
    orders_file = open('/Users/felipe/bitme/output.orders', 'w')
    fills_file.write(Fill.get_header() + '\n')
    orders_file.write(Orders().to_csv() + '\n')
    position_btc = initial_position_btc
    position_usd = initial_position_usd

    k = 0

    for candles_view in candles.views():
        # candles_view = all candles from 0 to current

        if True:
            sys.stdout.write("progress: %d out of %d (%.4f%%)   \r" % (k, candles.size(), 100*float(k)/candles.size()))
            sys.stdout.flush()
            k = k + 1

        orders_to_send = tac.handle_candles(SimCandles(data=candles_view), opened_orders, position_btc, position_usd)

        orders_to_send = simulate_cancel_orders(orders_to_send, position_btc, position_usd)

        if orders_to_send.size() > 0:
            print("")
            print("ORDERS")
            print(orders_to_send.to_csv())
            orders_file.write(orders_to_send.to_csv(False) + '\n')
            print("--------")

        opened_orders.merge(orders_to_send)

        last_candle = candles_view.iloc[-1]
        current_time = last_candle.name  # pd.Timestamp
        high = last_candle.high
        low = last_candle.low
        volume = last_candle.volume
        close_p = last_candle.close
        open_p = last_candle.open

        # fill sim
        for order in opened_orders.data.values():
            is_sell = order.side[0] == 's'
            is_buy = not is_sell

            # handle market order
            if isinstance(order, MarketOrder):
                size_filled = volume
                price_fill = (open_p + 2.*(low + high) + 3.*close_p) / 8.
                cost = 0.00075 * volume * price_fill  # BITMEX cost
                order.set_filled()

                size_filled = size_filled if is_buy else -size_filled
                position_btc += size_filled
                position_usd -= size_filled * price_fill + cost

                fill = Fill(order, size_filled, price_fill, current_time)
                fills += [fill]

                fills_file.write(fills[-1].to_line() + "\n")
                fills_file.flush()
                print("FILL: " + str(fill))
                sys.stdout.flush()

            elif isinstance(order, LimitOrder):
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
                    size_filled = order.fill(vol_fill)
                    fill = Fill(order, size_filled, order.price, current_time)
                    fills += [fill]

                    size_filled = size_filled if is_buy else -size_filled
                    position_btc += size_filled
                    if position_btc < 0:
                        raise RuntimeError(
                            "Position should never be negative (it is %s). Last fill was %s" % (position_btc, size_filled))
                    position_usd -= size_filled * order.price

                    fills_file.write(fills[-1].to_line() + "\n")
                    fills_file.flush()
                    print("FILL: " + str(fill))
                    sys.stdout.flush()

                    # print("vol_fill = " + str(vol_fill))
                    # active_orders.printf()
            else:
                raise ValueError("order type not supported")
        opened_orders.clean_filled()

    print_fills = False
    if print_fills:
        print("Fills:")
        for fill in fills:
            print(str(fill.to_json()))

    print("")
    print("position btc = " + str(position_btc))
    print("position usd = " + str(position_usd))
    print("close price = " + str(close_p))
    print("optimist realized profit = " + str(position_usd + position_btc * close_p - initial_position_usd))

    fills_file.close()
    orders_file.close()

    return 0


if __name__ == "__main__":
    sys.exit(main())
