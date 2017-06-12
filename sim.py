import sys
from candles import Candles


def main():
    # candles1s = read_candles('/Users/felipe/bitme/data/data1s.csv')
    # candles1s = read_candles('/Users/felipe/bitme/data/test')
    print("starting")
    candles = Candles.fromfilename('/Users/felipe/bitme/data/test')
    candles.get_candles(2, candles.ts_l[-1] - 5, candles.ts_l[-1]).printf()
    print "------"
    candles.get_candles(58, candles.ts_l[-1] - 25, candles.ts_l[-1] + 29).printf()
    print "------"
    #candles.get_candles(2, candles.ts_l[-1] - 5, candles.ts_l[-1]).printf()

    return 0


if __name__ == "__main__":
    sys.exit(main())
