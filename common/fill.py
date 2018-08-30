from enum import Enum

from api.symbol import Symbol
from utils.utils import to_str
import pandas as pd
from pandas import Timestamp


class FillType(Enum):
    complete = 'COMPLETE'  # means that this fill close an order
    partial = 'PARTIAL'


class Fill:
    def __init__(self,
                 symbol: Symbol,
                 side: str,
                 qty_filled: float,
                 price_fill: float,
                 fill_time: pd.Timestamp,
                 fill_type: FillType,
                 order_id: str):
        self.symbol = symbol  # type: Symbol
        self.side = side
        self.qty = qty_filled  # type: float
        self.price = price_fill  # type: float
        self.fill_time = fill_time  # type: Timestamp
        self.fill_type = fill_type  # type: FillType
        self.order_id = order_id  # type: str

    @staticmethod
    def create_from_raw(raw: dict):
        return Fill(symbol=Symbol[raw['symbol']],
                    side=raw['side'],
                    qty_filled=raw['lastQty'],
                    price_fill=raw['lastPx'],
                    fill_time=pd.Timestamp(raw['transactTime']),
                    fill_type=FillType.complete if raw['leavesQty'] == 0 else FillType.partial,
                    order_id=raw['clOrdID'] if 'clOrdID' in raw else 'Bitmex')

    def __repr__(self):
        return str(self.to_json())

    def __str__(self):
        return str(self.to_line())

    def to_json(self):
        params = {
            'time': str(self.fill_time),
            'symbol': str(self.symbol.name),
            'order_id': self.order_id,
            'side': self.side,
            'qty': str(int(self.qty)),  # USD
            'price': to_str(self.price, 2),  # USD
            'type': self.fill_type.name
        }
        return params

    def to_line(self):
        return ','.join([
            str(self.fill_time),  # type: pd.Timestamp
            str(self.symbol.name),
            str(self.order_id),
            str(self.side),
            str(int(self.qty)),  # USD
            str(to_str(self.price, 2)),  # USD
            str(self.fill_type.name)
        ])

    @staticmethod
    def get_header():
        return "time,symbol,order_id,side,qty,price,type"
