import logging
import queue

import pandas as pd

from common import Fill, OrderCommon
from common.quote import Quote
from common.trade import Trade
from tactic import TacticInterface

logger = logging.getLogger('root')


class TacticEventHandler:

    def __init__(self, tactic: TacticInterface, live):
        self.live = live
        assert isinstance(tactic, TacticInterface)
        self.tactic = tactic
        self.queue = queue.Queue()
        self.tactic_method = {
            Quote: self.tactic.handle_quote,
            Trade: self.tactic.handle_trade,
            Fill: self.tactic.handle_fill,
            OrderCommon: self.tactic.handle_cancel,
            pd.DataFrame: self.tactic.handle_1m_candles
        }

    def run_forever(self):
        try:
            while True:
                event = self.queue.get(block=True, timeout=None)
                method = self.tactic_method[type(event)]
                method(event)
        except Exception as e:
            # let's make the code exit if a tactic throws
            import sys
            # the main thread will capture this info
            exc_info = sys.exc_info()
            logger.error("CALLING live error thrower")
            self.live.throw_exception_from_tactic(exc_info)
