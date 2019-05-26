import pandas as pd
import sys

FOR_DATE = sys.argv[1] if len(sys.argv) > 1 else '20190515'

INTERVAL = '10s'
HORIZON = pd.Timedelta(INTERVAL) * 30  # forecast horizon
DATA_PATH = '/Users/felipe/bitcoin/{type}/{date}.csv.gz'
OUTPUT = '/Users/felipe/bitcoin/data/{date}-training.csv'
SPANS = [2, 3, 6, 15, 30, 60]


def read_trades(date):
    date = str(date)
    path = DATA_PATH.format(type='trades', date=date)
    t = pd.read_csv(path,
                    index_col='timestamp',
                    parse_dates=True,
                    date_parser=lambda x: pd.Timestamp(x.replace('D', ' ')),
                    infer_datetime_format=True,
                    usecols=['timestamp', 'symbol', 'price', 'side', 'size'])
    t.index.name = 'time'
    t = t[(t.symbol == 'XBTUSD') & (t.price > 1)]
    t = t.dropna()
    t.drop(columns={'symbol'}, inplace=True)
    t = t[['price', 'side', 'size']]
    return t


def read_quotes(date):
    date = str(date)
    path = DATA_PATH.format(type='quotes', date=date)
    t = pd.read_csv(path,
                    index_col='timestamp',
                    parse_dates=True,
                    date_parser=lambda x: pd.Timestamp(x.replace('D', ' ')),
                    infer_datetime_format=True,
                    usecols=['timestamp', 'symbol', 'bidPrice', 'askPrice', 'bidSize', 'askSize'])
    t.index.name = 'time'
    t = t[(t.symbol == 'XBTUSD') & (t.bidPrice > 1) & (t.askPrice > 1) & (t.bidPrice < t.askPrice)]
    t = t.dropna()
    t.drop(columns={'symbol'}, inplace=True)
    t = t[['bidPrice', 'askPrice', 'bidSize', 'askSize']]
    return t


print('reading trades ...')
t = read_trades(FOR_DATE)

# resample trades
t.loc[t.side == 'Sell', 'size'] *= -1
t.rename(columns={'size': 'boughtSum'}, inplace=True)
t['soldSum'] = t['boughtSum']
t = t[['boughtSum', 'soldSum']]
t['boughtSum'].clip_lower(0, inplace=True)
t['soldSum'].clip_upper(0, inplace=True)
t.loc[:, 'soldSum'] *= -1
t = t.resample(INTERVAL).agg('sum').fillna(method='ffill')
t = t[['boughtSum', 'soldSum']]

print('reading quotes ...')
q = read_quotes(FOR_DATE)


def rolling_future(s, h, method):
    df2 = s[::-1]
    df2.index = pd.datetime(2050, 1, 1) - df2.index
    r = df2.rolling(h)
    df2 = getattr(r, method)()
    df3 = df2[::-1]
    df3.index = s.index
    return df3


def add_fcst(q, horizon=HORIZON):
    print('  rolling big ...')
    bidMax = rolling_future(q['bidPrice'], horizon, 'max')
    print('  rolling ask ...')
    askMin = rolling_future(q['askPrice'], horizon, 'min')
    print('  long ...')
    q['longPnl'] = bidMax - q['askPrice']
    print('  short ...')
    q['shortPnl'] = q['bidPrice'] - askMin


print('adding forecasts ...')
add_fcst(q)
q.dropna(inplace=True)

print('spread')
q['spread'] = 2. * (q['askPrice'] - q['bidPrice'])  # in Tick unit
print(q.head(2))
print('resampling quotes')
q = q.resample(INTERVAL).agg('mean').fillna(method='ffill')
q.rename(columns={c: c + 'Avg' for c in q.columns}, inplace=True)

print('t head = {}'.format(t.head(2)))
print('concating trades ...')
df = pd.concat([t, q], axis=1, join='inner')
Y_cols = ['longPnlAvg', 'shortPnlAvg']
X_cols = ['boughtSum', 'soldSum', 'bidPriceAvg', 'askPriceAvg', 'bidSizeAvg', 'askSizeAvg', 'spreadAvg']
df = df[Y_cols + X_cols]

df.dropna(inplace=True)


def ema(df, spans, columns):
    """ computes ewm for each column, for each span in spans"""
    dfs = [df]
    for span in spans:
        cols = {i: 'E{}{}'.format(span, i) for i in columns}
        dfs.append(df[columns].ewm(span=span).mean().rename(columns=cols))
    return pd.concat(dfs, axis=1)


df = ema(df, SPANS, X_cols)

assert len(df) > 10

print('printing to {}'.format(OUTPUT.format(date=FOR_DATE)))
df.to_csv(OUTPUT.format(date=FOR_DATE))
