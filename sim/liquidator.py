from api.symbol import Symbol
from tactic.tactic_interface import TacticInterface


class Liquidator(TacticInterface):
    def __init__(self):
        pass

    def initialize(self, exchange, preferences) -> None:
        pass

    def finalize(self) -> None:
        pass

    def get_symbol(self) -> Symbol:
        pass

    def handle_trade(self, trade) -> None:
        pass

    def handle_quote(self, quote) -> None:
        pass

    def handle_fill(self, fill) -> None:
        pass

    def handle_1m_candles(self, candles) -> None:
        pass

    def handle_cancel(self, order) -> None:
        pass

    def handle_liquidation(self, pnl: float):
        pass

    @staticmethod
    def id() -> str:
        return "LIQ"
