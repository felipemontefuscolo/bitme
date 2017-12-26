import base64
import uuid
from cdecimal import Decimal
import pandas as pd
from enum import Enum

TWOPLACES = Decimal(10) ** -2
EIGHPLACES = Decimal(10) ** -8


# Order container util
class Orders:
    def __init__(self, id_to_order_map=None):
        # type: (dict) -> None
        if id_to_order_map is None:
            self.data = dict()  # id -> OrderCommon
        else:
            self.data = dict(id_to_order_map)
        pass

    def __getitem__(self, index):
        # type: (index) -> OrderCommon
        return self.data[index]

    def __iter__(self):
        return iter(self.data.values())

    def _gen_order_id(self):
        return str('bitme_' + base64.b64encode(uuid.uuid4().bytes).decode('utf8').rstrip('=\n'))

    def size(self):
        return len(self.data)

    def merge(self, orders):
        # type: (Orders) -> None
        self.data.update(orders.data)

    def pop_cancels(self):
        c = Orders()
        nc = Orders()
        for o in self.data.values():
            if o.status is OrderStatus.canceled:
                c.add(o)
            else:
                nc.add(o)
        self.data = nc.data
        return c

    # replace old order with same id
    def add(self, order):
        self.data[order.id] = order

    def clean_filled(self):
        self.data = dict([(oid, order) for oid, order in self.data.iteritems() if order.size > 0])

    def to_csv(self, header=True):
        # type: () -> str
        r = ['time,side,size,price'] if header else []
        for o in self.data.values():
            try:
                price = str(o.price)
            except AttributeError:
                price = 'market'  # market order
            r += [','.join([o.ts.strftime('%Y-%m-%dT%H:%M:%S'), o.side, str(o.size), price])]
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
        self.order_type = get_or_none(kargs, 'order_type')  # type: OrderType
        self.time_in_force = get_or_none(kargs, 'time_in_foce')  # type: TimeInForce
        self.contingency_type = get_or_none(kargs, 'contingency_type')  # type: ContingencyType

        # data change by the exchange
        self.filled = 0.  # type: float
        self.time_posted = None  # type: pd.Timestamp
        self.status = OrderStatus.opened  # type: OrderStatus
        self.status_msg = None  # type: OrderSubmissionError

    def qty_sign(self):
        self._sign(self.signed_qty)

    def is_sell(self):
        return self.signed_qty < 0

    def is_buy(self):
        return self.signed_qty > 0

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
    opened = 1
    filled = 2
    canceled = 3


class OrderSubmissionError(Enum):
    insufficient_funds = 1
    invalid_price = 2
    end_of_sim = 3
    unknown = 4


class OrderType(Enum):
    market = 1
    limit = 2
    stop = 3


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
