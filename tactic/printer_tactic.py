from tactic import TacticInterface, ExchangeInterface, Symbol, OrderCommon, Fill
import pandas as pd

class PrinterTactic(TacticInterface):
    """
    This class is associated to orders issued by Bitmex
    """
    def initialize(self, exchange: ExchangeInterface, preferences: dict) -> None:
        pass

    def get_symbol(self) -> Symbol:
        pass

    def handle_1m_candles(self, candles1m: pd.DataFrame) -> None:
        pass

    def handle_fill(self, fill: Fill) -> None:
        print("---PrinterTactic : got a fill:")
        print(str(fill))
        pass

    def handle_cancel(self, order: OrderCommon) -> None:
        print("---PrinterTactic : got a cancel")
        print(str(order))
        pass
