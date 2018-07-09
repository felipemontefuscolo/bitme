import base64
import uuid

import math
import pandas as pd
from enum import Enum

# Order container util
from typing import Union, Iterable, Dict

from api.symbol import Symbol


class Orders:
    def __init__(self, orders=None):
        # type: (Iterable) -> None
        if orders is None:
            self.data = dict()  # id -> OrderCommon
        elif isinstance(orders, Dict):
            self.data = orders
        else:
            self.data = dict([(o.id, o) for o in orders])

        pass

    def __getitem__(self, key):
        # type: (key) -> OrderCommon
        return self.data[key]

    def __iter__(self):
        return iter(self.data.values())

    def of_symbol(self, symbol):
        # type: (Enum) -> Orders
        return Orders(dict([(o_id, o) for o_id, o in self.data.items() if o.symbol == symbol]))

    def size(self):
        return len(self.data)

    def values(self):
        return self.data.values()

    def keys(self):
        return self.data.keys()

    def merge(self, orders):
        # type: (Orders) -> None
        for k in orders.data:
            self.data[k] = orders.data[k]

    def drop_closed_orders(self):
        # return num of dropped orders
        l = len(self.data)
        self.data = dict([(o.id, o) for o in self.data.values()
                          if o.status == OrderStatus.opened or o.status == OrderStatus.pending])
        return l - len(self.data)

    def market_orders(self):
        return Orders(dict([(o.id, o) for o in self.data.values()
                            if o.type == OrderType.market]))

    # replace old order with same id
    def add(self, order):
        self.data[order.id] = order

    def clean_filled(self, specific_order=None):
        # type: (OrderCommon) -> None
        if not specific_order:
            self.data = dict([(oid, order) for oid, order in self.data.items() if order.status != OrderStatus.filled])
        else:
            del self.data[specific_order.id]

    @staticmethod
    def to_csv(orders_list):
        s = OrderCommon.get_header() + '\n'
        for o in orders_list:  # type: OrderCommon
            s += o.to_line() + '\n'
        return s


class OrderCommon:
    _count = 0

    def __init__(self, **kargs):
        # self.id = str('bitme_' + base64.b64encode(uuid.uuid4().bytes).decode('utf8').rstrip('=\n'))  # type: str
        self.id = str('zaloe_' + str(OrderCommon._count))  # type: str
        OrderCommon._count += 1
        self.symbol = kargs['symbol']  # type: Symbol
        self.signed_qty = math.floor(_get(kargs, 'signed_qty', float('nan')))  # type: float
        self.price = round(_get(kargs, 'price', float('nan')), 1)  # type: float
        self.stop_price = _get(kargs, 'stop_price', float('nan'))  # type: float
        self.linked_order_id = _get(kargs, 'linked_order_id', None)  # type: str
        self.type = kargs['type']  # type: OrderType
        self.time_in_force = _get(kargs, 'time_in_foce', None)  # type: TimeInForce
        self.contingency_type = _get(kargs, 'contingency_type', None)  # type: ContingencyType
        self.tactic = kargs['tactic']

        # data change by the exchange
        self.filled = 0.  # type: float
        self.fill_price = float('nan') if self.type == OrderType.market else self.price
        self.time_posted = None  # type: pd.Timestamp
        self.status = OrderStatus.pending  # type: OrderStatus
        self.status_msg = None  # type: OrderCancelReason

        # sanity check
        q = abs(self.signed_qty) * 2 + 1.e-10
        assert abs(q - math.floor(q)) < 1.e-8

    def qty_sign(self):
        self._sign(self.signed_qty)

    def is_sell(self):
        return self.signed_qty < 0

    def is_buy(self):
        return self.signed_qty > 0

    def is_open(self):
        return self.status == OrderStatus.opened

    def is_pending(self):
        return self.status == OrderStatus.pending

    def is_fully_filled(self):
        return abs(self.filled - self.signed_qty) < 1.e-10

    @staticmethod
    def _sign(x):
        return -1 if x < 0 else +1

    def fill(self, size):
        # type: (float) -> bool
        """ :return: True if fully filled, False otherwise  """
        assert self._sign(size) == self._sign(self.signed_qty)
        assert self.status == OrderStatus.opened
        remaining = self.signed_qty - self.filled
        if abs(size) >= abs(remaining):
            size = remaining
            self.status = OrderStatus.filled
            self.filled += size
            return True
        self.filled += size
        return False

    def __str__(self):
        return str(self.to_line())

    def to_line(self):
        side = 'buy' if self.signed_qty > 0 else 'sell'
        return ','.join([
            str(self.time_posted.strftime('%Y-%m-%dT%H:%M:%S')),
            str(self.symbol),
            str(self.id),
            str(side),
            str(self.signed_qty),
            str(self.filled),
            str(self.price),
            str(self.type.name),
            str(self.status.name),
            str(self.e_str(self.status_msg))
        ])

    @staticmethod
    def get_header():
        return 'time,symbol,id,side,qty,filled,price,type,status,status_msg'

    @staticmethod
    def e_str(x):
        # type: (Enum) -> str
        try:
            return str(x.value)
        except AttributeError:
            return ""


class OrderStatus(Enum):
    pending = 'PENDING'
    opened = 'OPENED'
    filled = 'FILLED'
    canceled = 'CANCELED'


class OrderCancelReason(Enum):
    insufficient_funds = "insufficient funds"
    invalid_price = "invalid price"
    end_of_sim = "end of sim"
    requested_by_user = "requested by user"
    liquidation = "liquidation"
    unknown = "unknown"


class OrderType(Enum):
    market = 'MARKET'
    limit = 'LIMIT'
    stop = 'STOP'


class TimeInForce(Enum):
    day = 'day'
    good_til_cancel = 'good_til_cancel'
    immediate_or_cancel = 'immediate_or_cancel'
    fill_or_kill = 'fill_or_kill'


class ContingencyType(Enum):
    one_cancels_the_other = 'one_cancels_the_other'
    one_triggers_the_other = 'one_triggers_the_other'
    one_updates_the_other_absolute = 'one_updates_the_other_absolute'
    one_updates_the_other_proportional = 'one_updates_the_other_proportional'


def _get(dicti, key, default):
    try:
        return dicti[key]
    except KeyError:
        return default


def to_str(number, precision=2):
    return "%.{}f".format(precision) % round(float(number), precision)
