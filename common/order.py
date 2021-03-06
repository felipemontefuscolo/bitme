import math
from enum import Enum
# Order container util
from typing import Iterable, Dict, Union, List

import pandas as pd
import numpy as np

from api.symbol import Symbol


class OrderStatus(Enum):
    Pending = 'Pending'  # not in bitmex
    PartiallyFilled = 'PartiallyFilled'
    Filled = 'Filled'
    New = 'New'
    Canceled = 'Canceled'
    Rejected = 'Rejected'

    def __str__(self):
        return self.name

    @staticmethod
    def is_open(raw: dict):
        status = raw['ordStatus']
        return status == 'PartiallyFilled' or status == 'New'


# in bitmex: ordRejReason
class OrderCancelReason(Enum):
    insufficient_funds = "insufficient funds"
    invalid_price = "invalid price"
    cross_during_post_only = "cross during post only"
    end_of_sim = "end of sim"
    requested_by_user = "requested by user"
    liquidation = "liquidation"
    unknown = "unknown"

    def __str__(self):
        return self.name


class OrderType(Enum):
    Market = 'Market'
    Limit = 'Limit'
    Stop = 'Stop'

    def __str__(self):
        return self.name


class TimeInForce(Enum):
    Day = 'Day'
    GoodTillCancel = 'GoodTillCancel'
    ImmediateOrCancel = 'ImmediateOrCancel'
    FillOrKill = 'FillOrKill'

    def __str__(self):
        return self.name


class ContingencyType(Enum):
    # TODO: this is probably wrong. They should be camelCase
    one_cancels_the_other = 'one_cancels_the_other'
    one_triggers_the_other = 'one_triggers_the_other'
    one_updates_the_other_absolute = 'one_updates_the_other_absolute'
    one_updates_the_other_proportional = 'one_updates_the_other_proportional'

    def __str__(self):
        return self.name


class OrderCommon:
    def __init__(self,
                 symbol: Symbol,
                 type: OrderType,
                 client_id: str = None,
                 signed_qty: float = float('nan'),
                 price: float = float('nan'),
                 stop_price: float = float('nan'),
                 linked_order_id: str = None,
                 time_in_force: TimeInForce = None,
                 contingency_type: ContingencyType = None):

        self.symbol = symbol  # type: Symbol
        self.signed_qty = signed_qty  # type: float
        self.price = price  # type: float
        self.stop_price = stop_price  # type: float
        self.linked_order_id = linked_order_id  # type: str
        self.type = type  # type: OrderType
        if time_in_force is None:
            self.time_in_force = time_in_force  # type: TimeInForce
        elif type == OrderType.Market:
            self.time_in_force = TimeInForce.FillOrKill
        else:
            self.time_in_force = TimeInForce.GoodTillCancel
        self.contingency_type = contingency_type  # type: ContingencyType

        # data changed by the exchange
        self.leaves_qty = abs(signed_qty)
        self.time_posted = None  # type: pd.Timestamp
        self.status = OrderStatus.Pending  # type: OrderStatus
        self.status_msg = None  # type: OrderCancelReason
        self.bitmex_id = None  # type: str
        self.client_id = client_id  # type: str

        # sanity check
        if not np.isnan(signed_qty):
            q = abs(self.signed_qty) * 2 + 1.e-10
            assert abs(q - math.floor(q)) < 1.e-8

        # used in sim only and only valid for limit orders
        # this is an extra information to help with the fill simulator
        self._made_spread = None  # type: bool

    def filled(self):
        return abs(self.signed_qty) - self.leaves_qty

    def side(self) -> int:
        return -1 if self.signed_qty < 0 else +1

    def side_str(self) -> str:
        return 'Sell' if self.signed_qty < 0 else 'Buy'

    def is_open(self):
        if self.status == OrderStatus.New or self.status == OrderStatus.PartiallyFilled:
            assert self.leaves_qty > 0
            return True
        return False

    def is_sell(self):
        return self.signed_qty < 0

    def is_buy(self):
        return self.signed_qty > 0

    def fill(self, qty: '> 0'):
        # type: (float) -> bool
        """ :return: True if fully filled, False otherwise  """

        assert qty > 0
        assert self.is_open()

        self.leaves_qty = max(0, self.leaves_qty - qty)
        if self.leaves_qty == 0:
            self.status = OrderStatus.Filled
            return True
        self.status = OrderStatus.PartiallyFilled
        return False

    def __str__(self):
        return str(self.to_line())

    def __repr__(self):
        return str(self.to_line())

    def to_line(self):
        side = 'buy' if self.signed_qty > 0 else 'sell'
        return ','.join([
            str(self.time_posted),
            str(self.symbol),
            str(self.client_id),
            str(side),
            str(self.signed_qty),
            str(self.leaves_qty),
            str(self.price) if (self.price is not None and not np.isnan(self.price)) else '',
            str(self.type.name),
            str(self.status.name),
            str(self.e_str(self.status_msg))
        ])

    @staticmethod
    def get_header():
        return 'timestamp,symbol,id,side,qty,leaves_qty,price,type,status,status_msg'

    @staticmethod
    def e_str(x: Union[Enum, str]) -> str:
        try:
            return str(x.value)
        except AttributeError:
            return str(x).replace(',', '...').replace('\n', '').replace('\r', '')

    def to_bitmex(self) -> dict:
        a = {}
        if self.client_id:
            a['clOrdID'] = self.client_id
        if self.symbol:
            a['symbol'] = self.symbol.name
        if self.signed_qty and not np.isnan(self.signed_qty):
            a['orderQty'] = abs(self.signed_qty)
            a['side'] = 'Buy' if self.signed_qty > 0 else 'Sell'
        if self.price and not np.isnan(self.price):
            a['price'] = self.price
        if self.stop_price and not np.isnan(self.stop_price):
            a['stopPx'] = self.stop_price
        if self.linked_order_id:
            assert self.contingency_type
            a['clOrdLinkID'] = str(self.linked_order_id)
        if self.type:
            a['ordType'] = self.type.name
            if self.type == OrderType.Limit:
                assert a.get('price')
                a['execInst'] = 'ParticipateDoNotInitiate'  # post-only, postonly, post only
        if self.time_in_force:
            a['timeInForce'] = self.time_in_force.name
        else:
            if self.type == OrderType.Market:
                a['timeInForce'] = TimeInForce.FillOrKill.name
            elif self.type == OrderType.Limit or OrderType.Stop:
                a['timeInForce'] = TimeInForce.GoodTillCancel.name
        if self.contingency_type:
            assert self.linked_order_id
            a['contingencyType'] = self.contingency_type.name
        return a

    # update this object and return itself
    def update_from_bitmex(self, order: dict) -> 'OrderCommon':
        assert self.symbol.name == order['symbol']
        if 'ordType' in order:
            assert self.type == OrderType[order['ordType']]

        if self.client_id is None:
            self.client_id = order.get('clOrdID')
        else:
            if self.client_id != order['clOrdID']:
                raise ValueError("Updating from order with different id. Self: {}, other: {}".format(self.client_id,
                                                                                                     order['clOrdID']))

        if self.bitmex_id is None:
            self.bitmex_id = order['orderID']
        else:
            if self.bitmex_id != order['orderID']:
                raise ValueError(
                    "Updating from order with different bitmex id. Self: {}, other: {}".format(self.bitmex_id,
                                                                                               order['orderID']))
        if order['side'] == 'Buy':
            side = +1
        elif order['side'] == 'Sell':
            side = -1
        else:
            raise AttributeError('Unknown side: {}'.format(order['side']))

        self.status = OrderStatus[order['ordStatus']]

        if 'orderQty' in order:
            self.signed_qty = side * order['orderQty']
        self.price = order.get('price', self.price)
        self.stop_price = order.get('stopPx', self.stop_price)
        self.linked_order_id = order.get('clOrdLinkID', self.linked_order_id)

        if 'timeInForce' in order:
            self.time_in_force = TimeInForce[order['timeInForce']] if order['timeInForce'] else None
        if 'contingencyType' in order:
            self.contingency_type = ContingencyType[order['contingencyType']] if order['contingencyType'] else None
        self.status_msg = order.get('text', self.status_msg)
        self.leaves_qty = order.get('leavesQty', self.leaves_qty)

        if not self.time_posted:
            self.time_posted = pd.Timestamp(order['transactTime'])

        return self


OrderContainerType = Dict[str, OrderCommon]  # id -> order


def drop_closed_orders_dict(orders: Dict) -> OrderContainerType:
    return dict(filter(lambda x: x[1].is_open(), orders.items()))


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


def get_orders_id(orders: Union[OrderContainerType, List[OrderCommon], List[str], str]) -> List[str]:
    if len(orders) < 1:
        return
    if isinstance(orders, dict):
        ids = [o.client_id for o in orders.values()]
    elif isinstance(orders[0], OrderCommon):
        ids = [o.client_id for o in orders]
    elif isinstance(orders, str):
        ids = [orders]
    elif isinstance(orders, list):
        ids = orders
    else:
        raise ValueError("Don't know how to get id from: " + str(orders))
    return ids
