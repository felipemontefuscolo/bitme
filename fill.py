from enum import Enum

from orders import to_str, TWOPLACES, OrderCommon, OrderType
import pandas as pd
from pandas import Timestamp


class FillType(Enum):
    complete = 'COMPLETE'  # means that this fill close an order
    partial = 'PARTIAL'


class Fill:
    def __init__(self, order, qty_filled, price_fill, fill_time, fill_type):
        # type: (OrderCommon, float, float, pd.Timestamp) -> None
        self.symbol = order.symbol  # type: Enum
        self.order_id = order.id  # type: str
        self.side = 'buy' if order.signed_qty > 0 else 'sell'  # type: str
        self.qty = qty_filled  # type: float
        self.price = price_fill  # type: float
        self.order_type = order.type  # type: OrderType
        self.fill_time = fill_time  # type: Timestamp
        self.fill_type = fill_type  # type: FillType

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
            'price': to_str(self.price, TWOPLACES),  # USD
            'order_type': self.order_type.name,
            'type': self.fill_type.name
        }
        return params

    def to_line(self):
        return ','.join([
            str(self.fill_time.strftime('%Y-%m-%dT%H:%M:%S')),  # type: pd.Timestamp
            str(self.symbol.name),
            str(self.order_id),
            str(self.side),
            str(int(self.qty)),  # USD
            str(to_str(self.price, TWOPLACES)),  # USD
            str(self.order_type.name),
            str(self.fill_type.name)
        ])

    @staticmethod
    def get_header():
        return "time,symbol,order_id,side,qty,price,order_type,type"
