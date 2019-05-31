import pandas as pd
import numpy as np


def add_max_return(q: pd.DataFrame,
                   horizon: pd.Timedelta,
                   bidPriceCol: str,
                   askPriceCol: str):
    r = max_return(q, horizon, bidPriceCol, askPriceCol)
    q[r.columns] = r


def max_return(q: pd.DataFrame,
               horizon: pd.Timedelta,
               bidPriceCol: str,
               askPriceCol: str):
    
    horizon = pd.Timedelta(horizon)
    assert bidPriceCol in q.columns
    assert askPriceCol in q.columns
    
    bidMax = forward_rolling(q[bidPriceCol], horizon, 'max')
    askMin = forward_rolling(q[askPriceCol], horizon, 'min')
    
    maxLongReturn = (bidMax - q[askPriceCol])/q[askPriceCol]
    maxShortReturn = (q[bidPriceCol] - askMin)/q[bidPriceCol]
    
    h_name = f"{horizon.seconds//60}min"
    
    returns = pd.DataFrame({f'maxLongReturn{h_name}': maxLongReturn,
                            f'maxShortReturn{h_name}': maxShortReturn})
    return returns


def ema(df, spans, columns=None):
    """ computes ewm for each column, for each span in spans"""
    assert isinstance(df.index, pd.DatetimeIndex)
    assert df.index.freq is not None, 'Times must have constant intervals'

    columns = columns or df.columns
    df_ = df.copy()
    for span in spans:
        cols = [f'E{span}{i}' for i in columns]
        df_[cols] = df[columns].ewm(span=span).mean()  # 2/(span + 1)
    return df_

    
def forward_rolling(df, window, operation, **kwargs):
    """ Same as pandas.DataFrame.rolling, but looking forward.
        e.g.: forward_rolling(df, pd.Timedelta('2D'), 'max') """
    df2 = df[::-1]
    df2.index = df.index[-1] - df2.index
    df2 = getattr(df2.rolling(window, closed='both'), operation)(**kwargs)
    df3 = df2[::-1]
    df3.index = df.index
    if isinstance(df3, pd.DataFrame):
        df3.columns = (i + operation.capitalize() for i in df3.columns)
    return df3

    
if __name__ == '__main__':
    import unittest
    from pandas.util.testing import assert_frame_equal

    class MyToolsTest(unittest.TestCase):
        
        def setUp(self):
            self.index = pd.date_range('20160606', '20160613', name='time')
            self.horizon = pd.Timedelta('2D')
            self.df = pd.DataFrame({'bidPrice': [8.0, 7.0, 5.0, 1.0, 9.0, 25.0, 57.0, 121.0],
                                    'askPrice': [16.0, 14.0, 10.0, 2.0, 18.0, 50.0, 114.0, 242.0]},
                                    index=self.index)
        
        def test_forward_rolling(self):
            actual = forward_rolling(self.df, self.horizon, 'max')
            expected = pd.DataFrame({'bidPriceMax': [8.0, 7.0, 9.0, 25.0, 57.0, 121.0, 121.0, 121.0],
                                     'askPriceMax': [16.0, 14.0, 18.0, 50.0, 114.0, 242.0, 242.0, 242.0]},
                                     index=self.index)
            assert_frame_equal(expected, actual)
    
        def test_future_returns(self):
            actual = future_returns(self.df, self.horizon)
            expected = pd.DataFrame({'longPnl': [-8.0, -7.0, -1.0, 23.0, 39.0, 71.0, 7.0, -121.0],
                                     'shortPnl': [-2.0, 5.0, 3.0, -1.0, -9.0, -25.0, -57.0, -121.0]},
                                     index=self.index)
            assert_frame_equal(expected, actual)
            
            # test when inplace=True
            expected = self.df.merge(expected, on='time')
            actual = self.df.copy()
            future_returns(actual, self.horizon, inplace=True)
            assert_frame_equal(expected, actual)

        def test_ema(self):
            span = 2
            df = ema(self.df, [span])
            alpha = 2./(span + 1)
            self.assertEqual(df[f'E{span}bidPrice'][1],
                             (df['bidPrice'][0] * (1-alpha) + df['bidPrice'][1])/(2-alpha))
    
    unittest.main()
