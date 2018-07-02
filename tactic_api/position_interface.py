from tactic_api.symbol import Symbol
import pandas as pd
import json


class PositionInterface:
    def __init__(self, symbol):
        self.symbol = symbol

    """
        Only isolated margin is supported (see isolated vs cross here: https://www.bitmex.com/app/isolatedMargin)
        It means that when a position is opened, a fixed amount is taken as collateral. Any gains are only credited
        after the position is closed.
    """

    symbol = Symbol.XBTUSD
    avg_entry_price = None  # type: float
    break_even_price = None  # type: float
    liquidation_price = None  # type: float
    leverage = None  # type: int
    current_qty = None  # type: float
    side = None  # type: int
    """ In XBT """
    realized_pnl = None  # type: float
    has_started = False  # type: bool
    has_closed = False  # type: bool
    current_timestamp = None  # type: pd.Timestamp
    open_timestamp = None  # type: pd.Timestamp

    def __str__(self):
        # m = {
        #     'symbol': str(self.symbol),
        #     'avg_entry_price': str(self.avg_entry_price),
        #     'break_even_price': str(self.break_even_price),
        #     'liquidation_price': self.liquidation_price,
        #     'leverage': self.leverage,
        #     'current_qty': self.current_qty,
        #     'side': self.side,
        #     'realized_pnl': self.realized_pnl,
        #     'is_open': self.is_open,
        #     'current_timestamp': self.current_timestamp,
        #     'open_timestamp': self.open_timestamp,
        # }
        m = {}
        for i in dir(PositionInterface):
            if '__' not in i:
                m[i] = str(self.__getattribute__(i))

        return json.dumps(m, indent=4, sort_keys=True)
