from numpy import sign
from sympy import sign

from tactic_api.position_interface import PositionInterface
from tactic_api.symbol import Symbol
from common.constants import ZERO_TOL


def prevent_call_closed(fn):
    def wrapped(self, *args, **kwargs):
        if self.has_closed:
            raise ValueError("Can't call this method if position was closed")
        else:
            return fn(self, *args, **kwargs)
    return wrapped


class PositionSim(PositionInterface):

    def __init__(self, symbol: Symbol):
        super().__init__(symbol)

    def __str__(self):
        return super().__str__()

    @prevent_call_closed
    def update(self, qty, price, leverage, current_timestamp, fee, mark_price=None) -> PositionInterface:

        # TODO: support mark price
        if mark_price is None:
            mark_price = price

        self.leverage = leverage

        just_opened = False
        if not self.has_started:
            assert not self.open_timestamp
            assert not self.current_timestamp
            assert not self.current_qty
            assert not self.side
            just_opened = True
            self.open_timestamp = current_timestamp
            self.current_qty = 0.
            self.avg_entry_price = 0.
            self.realized_pnl = 0.
            self.has_started = True
            assert abs(qty) > ZERO_TOL
            self.side = int(sign(qty))

        self.current_timestamp = current_timestamp

        new_qty = self.current_qty + qty

        if self.current_qty * new_qty < -ZERO_TOL:
            raise ValueError("It does not support side change")

        fake_price = price * (1. + int(sign(qty)) * fee)

        if sign(qty) == sign(self.current_qty) or just_opened:
            self.avg_entry_price = (self.avg_entry_price * self.current_qty + fake_price * qty) / (
                    self.current_qty + qty)
        else:
            # TODO: take into account rebates. In this case, the line below could be outside the condition if/else
            self.realized_pnl += (1. / self.avg_entry_price - 1. / fake_price) * abs(qty) * self.side

        self.current_qty = new_qty

        self.break_even_price = (self.current_qty * self.avg_entry_price) / \
                                (self.current_qty + self.realized_pnl * self.avg_entry_price)

        if sign(qty) == sign(self.current_qty) or just_opened:
            self.liquidation_price = self._calc_liq_price(
                avg_entry=self.avg_entry_price,
                curr_qty=self.current_qty,
                lev=self.leverage
            )
        #print(self.avg_entry_price, self.liquidation_price, self.current_qty, self.realized_pnl, qty, fake_price)

        # close position
        if abs(self.current_qty) < ZERO_TOL:
            self.has_closed = True

        return self

    @staticmethod
    def _calc_liq_price(avg_entry, curr_qty, lev, funding_rate=0.001):
        # TODO: support funding rate
        maintenance_margin = float(sign(curr_qty)) * (1. / lev - (0.5 + 0.075 + funding_rate) / 100)
        return avg_entry / max(abs(1. + maintenance_margin), ZERO_TOL)

    @prevent_call_closed
    def would_change_side(self, qty):
        return self.current_qty * (qty + self.current_qty) < ZERO_TOL
