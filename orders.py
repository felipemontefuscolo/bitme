import base64
import uuid
from cdecimal import Decimal

import math
import pandas as pd
from enum import Enum

ONEPLACE = Decimal(10) ** -1
TWOPLACES = Decimal(10) ** -2
EIGHPLACES = Decimal(10) ** -8


# Order container util
class Orders:
    def __init__(self, id_to_order_map=None):
        # type: (dict) -> None
        if id_to_order_map is None:
            self.data = dict()  # id -> OrderCommon
        else:
            self.data = id_to_order_map
        pass

    def __getitem__(self, index):
        # type: (index) -> OrderCommon
        return self.data[index]

    def __iter__(self):
        return iter(self.data.values())

    @staticmethod
    def from_orders_list(orders_list):
        # type: (list) -> Orders
        return Orders(dict([(o.id, o) for o in orders_list]))

    def of_symbol(self, symbol):
        # type: (Enum) -> Orders
        return Orders(dict([(o.id, o) for o in self.data if o.symbol == symbol]))

    def _gen_order_id(self):
        return str('bitme_' + base64.b64encode(uuid.uuid4().bytes).decode('utf8').rstrip('=\n'))

    def size(self):
        return len(self.data)

    def merge(self, orders):
        # type: (Orders) -> None
        self.data.update(orders.data)

    def drop_closed_orders(self):
        self.data = dict([(o.id, o) for o in self.data.values() if o.status == OrderStatus.opened])

    # replace old order with same id
    def add(self, order):
        self.data[order.id] = order

    def clean_filled(self):
        self.data = dict([(oid, order) for oid, order in self.data.iteritems() if order.size > 0])

    def to_csv(self, header=True):
        # type: () -> str
        r = ['time,symbol,side,qty,price,type,status'] if header else []
        for o in self.data.values(): # type: OrderCommon
            side = 'buy' if o.signed_qty > 0 else 'sell'
            r += [','.join([o.time_posted.strftime('%Y-%m-%dT%H:%M:%S'), str(o.symbol), side, str(o.signed_qty),
                            str(o.price), str(o.type.name), str(o.status.name)])]
        return '\n'.join(r)


class OrderCommon:
    def __init__(self, **kargs):
        self.id = str('bitme_' + base64.b64encode(uuid.uuid4().bytes).decode('utf8').rstrip('=\n'))  # type: str
        self.symbol = kargs['symbol']  # type: Enum
        self.signed_qty = get_or_none(kargs, 'signed_qty')  # type: float
        # self.signed_simple_qty = get_or_none(kargs, 'signed_simple_qty') # type: float
        self.price = get_or_none(kargs, 'price')  # type: float
        self.stop_price = get_or_none(kargs, 'stop_price')  # type: float
        self.linked_order_id = get_or_none(kargs, 'linked_order_id')  # type: str
        self.type = kargs['type']  # type: OrderType
        self.time_in_force = get_or_none(kargs, 'time_in_foce')  # type: TimeInForce
        self.contingency_type = get_or_none(kargs, 'contingency_type')  # type: ContingencyType

        # data change by the exchange
        self.filled = 0.  # type: float
        self.fill_price = float('nan') if self.type == OrderType.market else self.price
        self.time_posted = None  # type: pd.Timestamp
        self.status = OrderStatus.opened  # type: OrderStatus
        self.status_msg = None  # type: OrderSubmissionError

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

    @staticmethod
    def _sign(x):
        return -1 if x < 0 else +1

    def fill(self, size):
        # type: (float) -> bool
        """ :return: True if fully filled, False otherwise  """
        assert self._sign(size) == self._sign(self.signed_qty)
        remaining = self.signed_qty - self.filled
        if abs(size) >= abs(remaining):
            size = remaining
            self.status = OrderStatus.filled
            self.filled += size
            return True
        self.filled += size
        return False

    def type(self):
        raise AttributeError("interface class")


class OrderStatus(Enum):
    opened = 'OPENED'
    filled = 'FILLED'
    canceled = 'CANCELED'


class OrderSubmissionError(Enum):
    insufficient_funds = 1
    invalid_price = 2
    end_of_sim = 3
    cancel_requested = 4
    unknown = 5


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


def get_or_none(dicti, key):
    try:
        return dicti[key]
    except KeyError:
        return None


def to_str(number, precision=TWOPLACES):
    return str(Decimal(str(number)).quantize(precision))
