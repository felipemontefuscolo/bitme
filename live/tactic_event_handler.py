import queue

import pandas as pd
from common import Fill, OrderCommon, OrderStatus
from common.quote import Quote
from common.trade import Trade
from tactic import TacticInterface


class TacticEventHandler:

    def __init__(self, tactic: TacticInterface):
        assert isinstance(tactic, TacticInterface)
        self.tactic = tactic
        self.queue = queue.Queue()
        self.tactic_method = {
            Quote: self.tactic.handle_quote,
            Trade: self.tactic.handle_trade,
            Fill: self.tactic.handle_fill,
            OrderCommon: self._handle_order,
            pd.DataFrame: self.tactic.handle_1m_candles
        }

    def run_forever(self):
        while True:
            event = self.queue.get(block=True, timeout=None)
            if event is None:
                continue
            method = self.tactic_method[type(event)]
            method(event)

    def _handle_order(self, order: OrderCommon):
        if order.status == OrderStatus.Canceled or order.status == OrderStatus.Rejected:
            return self.tactic.handle_cancel(order)
        elif order.status == OrderStatus.Filled:
            return self.tactic.handle_order_completed(order)

        raise AttributeError("Tactic should only handle cancels or fully filled orders")
