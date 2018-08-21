import queue

from common import Fill, OrderCommon
from common.quote import Quote
from common.trade import Trade
from tactic import TacticInterface


class TacticEventHandler:

    def __init__(self, tactic: TacticInterface):
        self.tactic = tactic
        self.queue = queue.Queue()
        self.tactic_method = dict()
        self.type_subscriptions = []

    def subscribe_to_type(self, type_):
        try:
            self.tactic_method[type_] = {
                Quote: self.tactic.handle_quote,
                Trade: self.tactic.handle_trade,
                Fill: self.tactic.handle_fill,
                OrderCommon: self.tactic.handle_cancel
            }[type_]
        except KeyError:
            raise ValueError('Type "{}" not supported'.format(type_))

    def unsubscribe_to_type(self, type_):
        if type_ in self.tactic_method:
            del self.tactic_method[type_]

    def start(self):
        while True:
            event = self.queue.get(block=True, timeout=None)
            method = self.tactic_method.get(type(event))
            if method:
                method(event)
