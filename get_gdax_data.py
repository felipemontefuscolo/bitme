import math
import sys
import time

import GDAX

from utils import Hour, get_current_ts, to_iso_utc, to_iso_local

NUM_CANDLES_GDAX_LIMIT = 199


# granularity in sec
# start and end are timestamps
def get_candles(publicClient, granularity, begin_ts, end_ts):
    n_candles = math.ceil((end_ts - begin_ts) / granularity)
    n_calls = int(math.ceil(n_candles / NUM_CANDLES_GDAX_LIMIT))
    candles = []
    for i in range(n_calls - 1, -1, -1):
        ss = begin_ts + i * NUM_CANDLES_GDAX_LIMIT * granularity
        ee = min(granularity * NUM_CANDLES_GDAX_LIMIT + ss, end_ts)
        candles_tmp = publicClient.getProductHistoricRates(granularity=granularity, start=to_iso_utc(ss), end=to_iso_utc(ee))
        if 'message' in candles_tmp:
            raise Exception('invalid gdax message: ' + str(candles_tmp))
        candles += candles_tmp
        time.sleep(1.005)
        # print str((n_calls - i)/n_calls)*100 + '% done'
    return candles


def main():
    # key = os.environ['GDAX_KEY']
    # b64secret = os.environ['GDAX_SECRET']
    # passphrase = os.environ['GDAX_PASS']

    publicClient = GDAX.PublicClient(product_id="ETH-USD")
    # authClient = GDAX.AuthenticatedClient(key, b64secret, passphrase, product_id="BTC-USD")

    current_time = get_current_ts()
    print("current time is:\n"
          "ts: " + str(current_time) + "\n" 
          "local: " + to_iso_local(current_time) + "\n"
          "utc: " + to_iso_utc(current_time))

    gran = Hour(1).to_sec()
    e = int(get_current_ts())
    e = e - e % gran
    b = e - Hour(2).to_sec()

    print("getting candles for (utc) " + to_iso_utc(b) + " -- " + to_iso_utc(e))
    candles = get_candles(publicClient, granularity=gran, begin_ts=b, end_ts=e)

    # time, low, high, open, close, volume
    # table = np.array(candles)
    # np.savetxt("data.txt", table);

    last_ts = sys.maxsize
    for candle in candles:
        if last_ts > int(candle[0]):
            last_ts = int(candle[0])
            candle[0] = to_iso_utc(candle[0])
        else:
            raise Exception(
                'invalid time stamp order: last = ' + str(to_iso_utc(last_ts)) + ', current = ' + to_iso_utc(candle[0]))

    print "time,low,high,open,close,volume"
    for candle in reversed(candles):
        print ','.join(str(e) for e in candle)

    return 0


if __name__ == "__main__":
    sys.exit(main())
