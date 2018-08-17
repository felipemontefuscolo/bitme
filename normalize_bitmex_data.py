import sys
import argparse
import pandas as pd

parser = argparse.ArgumentParser(description="Get bitmex data")

parser.add_argument('-b', '--begin-time', type=pd.Timestamp, required=True, help="Example: '2018-04-01T00:00:01'")

parser.add_argument('-e', '--end-time', type=pd.Timestamp, required=True, help="Example: '2018-04-01T00:00:10'")

parser.add_argument('-s', '--symbol', type=str, default='XBTUSD',
                    help='Instrument symbol. Send a bare series (e.g. XBU) to get data for the nearest expiring'
                         'contract in that series. You can also send a timeframe, e.g. `XBU:monthly`. '
                         'Timeframes are `daily`, `weekly`, `monthly`, `quarterly`, and `biquarterly`. (optional)')

parser.add_argument('-o', '--file-or-stdout', type=str, required=True, help='Output filename or "-" for stdout')
parser.add_argument('-i', '--raw-data', type=str, required=True, help='Output filename or "-" for stdout')

args = parser.parse_args(args, namespace)

args.begin_time = args.begin_time.to_pydatetime().astimezone(tzutc())
args.end_time = args.end_time.to_pydatetime().astimezone(tzutc())

return args


def main():
    if len(sys.argv) < 3:
        raise ValueError('Should provide two arguments: input-name and output-name')
    print(sys.argv)

    return 0


if __name__ == '__main__':
    sys.exit(main())
