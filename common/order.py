import math
from enum import Enum
# Order container util
from typing import Iterable, Dict

import pandas as pd

from api.symbol import Symbol


def is_order_closed(order):
    return order.status == OrderStatus.filled or order.status == OrderStatus.canceled


def drop_closed_orders_dict(orders: Dict) -> Dict:
    return dict(filter(lambda x: not is_order_closed(x[1]), orders.items()))


def drop_orders(orders_orig: Dict, order_ids_to_drop: Iterable) -> Dict:
    """
    :param orders_orig: dict id -> order
    :param order_ids_to_drop: list,set,map,.. anything where set(.) gives you the order id's
    :return:
    """
    keys = orders_orig.keys() - set(order_ids_to_drop)
    return {k: orders_orig[k] for k in keys}


def filter_symbol(orders: Dict, symbol: Symbol):
    return dict(filter(lambda x: x[1].symbol == symbol, orders.items()))


class OrderStatus(Enum):
    pending = 'PENDING'
    opened = 'OPENED'
    filled = 'FILLED'  # fully filled
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


class OrderCommon:
    def __init__(self,
                 symbol: Symbol,
                 type: OrderType,
                 tactic,
                 signed_qty: float = float('nan'),
                 price: float = float('nan'),
                 stop_price: float = float('nan'),
                 linked_order_id: str = None,
                 time_in_force: TimeInForce = None,
                 contingency_type: ContingencyType = None):
        # self.id = str('bitme_' + base64.b64encode(uuid.uuid4().bytes).decode('utf8').rstrip('=\n'))  # type: str
        self.id = str('zaloe_' + str(OrderCommon._count))  # type: str
        OrderCommon._count += 1

        self.symbol = symbol
        self.signed_qty = signed_qty
        self.price = price
        self.stop_price = stop_price
        self.linked_order_id = linked_order_id
        self.type = type
        self.time_in_force = time_in_force
        self.contingency_type = contingency_type
        self.tactic = tactic

        # data change by the exchange
        self.filled = 0.  # type: float
        self.fill_price = float('nan') if self.type == OrderType.market else self.price
        self.time_posted = None  # type: pd.Timestamp
        self.status = OrderStatus.pending  # type: OrderStatus
        self.status_msg = None  # type: OrderCancelReason
        self.bitmex_id = None  # type: str

        # sanity check
        q = abs(self.signed_qty) * 2 + 1.e-10
        assert abs(q - math.floor(q)) < 1.e-8

    _count = 0

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
