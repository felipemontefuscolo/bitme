import csv
import datetime
import sys
import time

import GDAX
import dateutil.parser
from utils import Day, Hour, Min, to_ts, to_iso_utc, get_current_ts

NUM_CANDLES_GDAX_LIMIT = 199


class Data:
    def __init__(self, filename):
        """
        :param filename: path to 1 seconds granular candles
        """
        self._beg_ts = 0

        pass


def convert_row(row):
    return [int(to_ts(row[0])), float(row[1]), float(row[2]), float(row[3]), float(row[4]), float(row[5])]


def read_candles(filename):
    table = []
    print("reading table ... ")
    with open(filename, 'r') as csvfile:
        csvfile.readline()
        table.append(convert_row(csvfile.readline().strip().split(',')))
        ref = table[0][0]
        reader = csv.reader(csvfile, delimiter=',')
        for row in reader:
            row = convert_row(row)
            for i in range(0, row[0] - ref - 1):
                table.append([])
            table.append(row)
            ref = row[0]
    print("table read")
    return table


def get_last_candles(candles1s, gran, begin_time, end_time):
    candlesNs = []


def main():
    #candles1s = read_candles('/Users/felipe/bitme/data/data1s.csv')
    candles1s = read_candles('/Users/felipe/bitme/data/test')

    for r in candles1s:
        print r

    return 0


if __name__ == "__main__":
    sys.exit(main())

