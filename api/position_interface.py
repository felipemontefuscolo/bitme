from api.symbol import Symbol
import pandas as pd
import json

from common import BITCOIN_TO_SATOSHI


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
    realized_pnl = float('nan')  # type: float
    is_open = False  # type: bool
    current_timestamp = None  # type: pd.Timestamp
    open_timestamp = None  # type: pd.Timestamp

    def __str__(self):
        m = {i: str(self.__getattribute__(i)) for i in dir(PositionInterface) if '__' not in i}

        return json.dumps(m, indent=4, sort_keys=True)

    def update_from_bitmex(self, raw: dict):

        assert self.symbol == Symbol[raw['symbol']]

        self.avg_entry_price = raw.get('avgEntryPrice', self.avg_entry_price)
        self.break_even_price = raw.get('breakEvenPrice', self.break_even_price)
        self.liquidation_price = raw.get('liquidationPrice', self.liquidation_price)
        self.leverage = raw.get('leverage', self.leverage)
        self.current_qty = raw.get('currentQty', self.current_qty)
        if not self.current_qty:
            self.side = None
        else:
            self.side = +1 if self.current_qty > 0 else -1
        if 'realisedPnl' in raw:
            self.realized_pnl = raw['realisedPnl'] / BITCOIN_TO_SATOSHI
        self.is_open = raw.get('isOpen', self.is_open)
        # TODO: those timestamps don't seem accurate! maybe use our own timestamp?
        self.current_timestamp = pd.Timestamp(raw.get('currentTimestamp', self.current_timestamp))
        self.open_timestamp = pd.Timestamp(raw.get('openingTimestamp', self.open_timestamp))

        return self
