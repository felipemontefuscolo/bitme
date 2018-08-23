from common.quote import Quote
from tactic import TacticInterface, ExchangeInterface, Symbol, OrderCommon, Fill
import pandas as pd


class BitmexDummyTactic(TacticInterface):
    """
    This class is associated to orders issued by Bitmex
    """

    def handle_quote(self, quote: Quote) -> None:
        pass

    def handle_order_completed(self, order: OrderCommon) -> None:
        pass

    def handle_liquidation(self, pnl: float):
        pass

    def id(self):
        return 'DUMMY'

    def init(self, exchange: ExchangeInterface, preferences: dict) -> None:
        pass

    def get_symbol(self) -> Symbol:
        pass

    def handle_1m_candles(self, candles1m: pd.DataFrame) -> None:
        pass

    def handle_submission_error(self, failed_order: OrderCommon) -> None:
        pass

    def handle_fill(self, fill: Fill) -> None:
        pass

    def handle_cancel(self, order: OrderCommon) -> None:
        pass

    def handle_trade(self):
        pass
