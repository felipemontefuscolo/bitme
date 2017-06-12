import sys
from candles import Candles
from tactic_mm import TacticMM
from utils import Hour


def main():
    print("starting sim")
    #candles = Candles.fromfilename('/Users/felipe/bitme/data/test')
    candles = Candles.fromfilename('/Users/felipe/bitme/data/data1s.csv')

    tac = TacticMM()

    beg = candles.ts_l[0]
    end = candles.ts_l[-1]

    beg = beg + Hour.to_sec(24)

    tac.handle_candles(candles, beg)

    return 0


if __name__ == "__main__":
    sys.exit(main())
