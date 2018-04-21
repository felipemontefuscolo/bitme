class TacticInterface:
    def __init__(self):
        pass

    def init(self, exchange, preferences):
        # type: (ExchangeCommon, dict) -> None
        raise AttributeError("interface class")

    def get_symbol(self):
        # type: () -> Enum
        raise AttributeError("interface class")

    def handle_candles(self, exchange):
        # type: (ExchangeCommon) -> None
        raise AttributeError("interface class")

    def handle_submission_error(self, failed_order):
        # type: (OrderCommon) -> None
        raise AttributeError("interface class")

    def handle_fill(self, exchange, fill):
        # type: (ExchangeCommon, Fill) -> None
        raise AttributeError("interface class")

    def handle_cancel(self, exchange, order):
        # type: (ExchangeCommon, OrderCommon) -> None
        raise AttributeError("interface class")

    def id(self):
        # type: () -> str
        return self.__class__.__name__