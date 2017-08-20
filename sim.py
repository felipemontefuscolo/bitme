import sys
from candles import Candles
from tactic_mm import TacticMM
from utils import Hour
from orders import LimitOrder, Orders
import math

def main():
    print("starting sim")
    candles = Candles.fromfilename('/Users/felipe/bitme/data/test')
    #candles = Candles.fromfilename('/Users/felipe/bitme/data/data1s.csv')

    tac = TacticMM('ETH-USD')
    orders = Orders()

    beg = candles.ts_l[0]
    end = candles.ts_l[-1]

    beg = beg + Hour(24).to_sec()

    #sent_orders = tac.handle_candles(candles, beg, orders)

    # for i in sent_orders.sells + sent_orders.buys:
    #     print i

    candles.get_candles(3, candles.ts_l[0], candles.ts_l[-1]).printf()
    #candles.printf()


    return 0


if __name__ == "__main__":
    sys.exit(main())
