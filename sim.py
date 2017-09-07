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
    active_orders = Orders()

    active_orders.merge(tac.handle_candles(candles, active_orders))
    active_orders.merge(tac.handle_candles(candles, active_orders))
    active_orders.printf()

    # for i in sent_orders.sells + sent_orders.buys:
    #     print i

    #candles.get_candles(1, candles.ts_l[0], candles.ts_l[-1]).printf()
    #candles.printf()


    return 0


if __name__ == "__main__":
    sys.exit(main())
