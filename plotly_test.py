import pandas as pd
import datetime

def read_data(file: str, begin: pd.Timestamp=None, end: pd.Timestamp=None) -> pd.DataFrame:
    timeparser = lambda s: pd.datetime.strptime(str(s), '%Y-%m-%dT%H:%M:%S')
    data = pd.DataFrame(pd.read_csv(file, parse_dates=True, index_col='time', date_parser=timeparser))

    if begin and end:
        data = data.loc[begin:end]
    elif begin:
        data = data.loc[begin:]
    elif end:
        data = data.loc[:end]
    return data

df = read_data('/Users/felipe/bitme/data/bitmex_1day.csv')
df.reset_index('time', inplace=True)
df.head()

import plotly.plotly as py
import plotly.graph_objs as go
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
import plotly.offline as offline

#offline.init_notebook_mode(connected=True)
from datetime import datetime

trace = go.Candlestick(x=df.time,
                open=df.open,
                high=df.high,
                low=df.low,
                close=df.close)

layout = go.Layout(
    xaxis = dict(
        rangeslider = dict(
            visible = False
        )
    )
)

data = [trace]
#offline.iplot(data, filename='/Users/felipe/bitme/simple_ohlc')
#offline.iplot(data)
fig = go.Figure(data=data,layout=layout)
offline.plot(fig, filename='/Users/felipe/bitme/simple_ohlc.html')

