from abc import ABCMeta, abstractmethod
from enum import Enum

import pandas as pd

from api import Symbol
# from tactic.tactic_interface import TacticInterface


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


class Order(metaclass=ABCMeta):
    client_id = None  # type: str
    price = None  # type: float
    stop_price = None  # type: float
    signed_qty = None  # type: float
    symbol = None  # type: Symbol
    type = None  # type: OrderType
    time_in_force = None  # type: TimeInForce
    contingency_type = None  # type: ContingencyType
    tactic = None  # type: TacticInterface
    linked_order_id = None  # type: str

    #
    #
    # read only data

    @property
    @abstractmethod
    def filled(self) -> float:
        raise AttributeError('interface class')

    @property
    @abstractmethod
    def filled_price(self) -> float:
        raise AttributeError('interface class')

    @property
    @abstractmethod
    def time_posted(self) -> pd.Timestamp:
        raise AttributeError('interface class')

    @property
    @abstractmethod
    def status(self) -> OrderStatus:
        raise AttributeError('interface class')

    @property
    @abstractmethod
    def status_msg(self) -> OrderCancelReason:
        raise AttributeError('interface class')

    @property
    @abstractmethod
    def fill_price(self) -> float:
        raise AttributeError('interface class')
