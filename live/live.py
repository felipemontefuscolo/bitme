from collections import OrderedDict

import sys
from enum import Enum
# import swagger_client

from common import ExchangeCommon, Position, Orders, Candles
import pandas as pd

import swagger_client  # bitmex lib
from pandas import Timestamp, Timedelta

MAX_NUM_CANDLES_BITMEX = 500


def time_parser(s):
    pd.datetime.strptime(str(s), '%Y-%m-%dT%H:%M:%S')


class LiveBitMex(ExchangeCommon):
    def __init__(self):
        ExchangeCommon.__init__(self)
        self.span = 10  # minutes; bitmex REST API limit = 500
        assert self.span <= MAX_NUM_CANDLES_BITMEX

        configuration = swagger_client.Configuration()
        configuration.host = 'https://www.bitmex.com/api/v1'
        self.api_client = swagger_client.ApiClient(configuration)
        self.trade_api = swagger_client.TradeApi(self.api_client)
        self.quote_api = swagger_client.QuoteApi(self.api_client)
        self.position_api = swagger_client.PositionApi(self.api_client)

        pass

    def get_candles1m(self):
        # type: (None) -> Candles

        t1 = Timestamp.now()
        t0 = t1 - Timedelta(minutes=self.span)

        page = self.trade_api.trade_get_bucketed(
            bin_size='1m',
            partial=True,
            symbol='XBTUSD',  # TODO: should be configurable
            count=self.span,
            start=0.0,
            reverse=False,
            start_time=t0,
            end_time=t1
        )

        # data = pd.DataFrame(  # is this conversion inefficient?

        #   pd.read_csv(filename, parse_dates=True, index_col='time', date_parser=time_parser))
        for i in range(len(page)):
            p = page[i]  # type: swagger_client.TradeBin
            page[i] = OrderedDict([('time', Timestamp(p.timestamp.strftime('%Y-%m-%dT%H:%M:%S'))),
                                   ('open', p.open),
                                   ('high', max(p.high, p.open)),
                                   ('low', min(p.low, p.open)),
                                   ('close', p.close),
                                   ('volume', p.volume)])

        data = pd.DataFrame.from_records(page[::-1], index='time')
        print(data)
        return Candles(data=data)

    def post_orders(self, orders):
        # type: (Orders) -> bool
        """
        :param orders:
        :return: True if any order was rejected
        """
        raise AttributeError("interface class")

    def current_time(self):
        # type: () -> Timestamp
        return Timestamp.now()

    def current_price(self):
        # type: () -> float
        q = self.quote_api.quote_get(count=1)[0]  # type: swagger_client.Quote
        return q.ask_price

    def get_position(self, symbol='XBTUSD'):
        # type: (Enum) -> Position
        self.position_api.position_get(count=3)

    def get_closed_positions(self, symbol='XBTUSD'):
        # type: (Enum) -> list(Position)
        raise AttributeError("interface class")

    def set_leverage(self, symbol, value):
        # type: (Enum, float) -> bool
        """
        :param symbol
        :param value Leverage value. Send a number between 0.01 and 100 to enable isolated margin with a fixed leverage.
               Send 0 to enable cross margin.
        :return: True if succeeded
        """
        raise AttributeError("interface class")

    def cancel_orders(self, orders, drop_canceled=True):
        # type: (Orders, bool) -> None
        raise AttributeError("interface class")


if __name__ == "__main__":
    live = LiveBitMex()
    #print(live.get_candles1m())
    for i in live.get_position():
        print(i)

