from numpy import sign
from sympy import sign

from api.position_interface import PositionInterface
from api.symbol import Symbol
from common.constants import ZERO_TOL


class PositionSim(PositionInterface):

    def __init__(self, symbol: Symbol, on_position_close):
        """
        :param symbol:
        :param on_position_close: a callback that takes PositionSim as argument. It will be called when this position
                                  close. After that, this position will self restart
        """
        super().__init__(symbol)
        self.on_position_close = on_position_close
        self.open_timestamp = None

    def __str__(self):
        return super().__str__()

    def update(self, signed_qty, price, leverage, current_timestamp, fee, mark_price=None) -> PositionInterface:

        # TODO: support mark price
        if mark_price is None:
            mark_price = price

        self.leverage = leverage

        just_opened = False
        if not self.is_open:
            assert not self.open_timestamp
            assert not self.signed_qty
            assert not self.side
            just_opened = True
            self.open_timestamp = current_timestamp
            self.signed_qty = 0.
            self.avg_entry_price = 0.
            self.realized_pnl = 0.
            self.is_open = True
            assert abs(signed_qty) > ZERO_TOL
            self.side = int(sign(signed_qty))

        new_qty = self.signed_qty + signed_qty

        if self.signed_qty * new_qty < -ZERO_TOL:
            raise ValueError("It does not support side change")

        fake_price = price * (1. + int(sign(signed_qty)) * fee)

        if sign(signed_qty) == sign(self.signed_qty) or just_opened:
            self.avg_entry_price = (self.avg_entry_price * self.signed_qty + fake_price * signed_qty) /\
                                   (self.signed_qty + signed_qty + 1.e-4)
        else:
            # TODO: take into account rebates. In this case, the line below could be outside the condition if/else
            d_price = fake_price - self.avg_entry_price
            self.realized_pnl += d_price / (self.avg_entry_price * fake_price) * self.leverage * abs(signed_qty) * self.side

        self.signed_qty = new_qty

        self.break_even_price = (self.signed_qty * self.avg_entry_price) / \
                                (self.signed_qty + self.realized_pnl * self.avg_entry_price)

        if sign(signed_qty) == sign(self.signed_qty) or just_opened:
            self.liquidation_price = self._calc_liq_price(
                avg_entry=self.avg_entry_price,
                curr_qty=self.signed_qty,
                lev=self.leverage
            )

        # close position
        if abs(self.signed_qty) < ZERO_TOL:
            self.is_open = False
            self.on_position_close(self)
            self.__init__(self.symbol, self.on_position_close)

        return self

    @staticmethod
    def _calc_liq_price(avg_entry, curr_qty, lev, funding_rate=0.001):
        # TODO: support funding rate
        maintenance_margin = float(sign(curr_qty)) * (1. / lev - (0.5 + 0.075 + funding_rate) / 100)
        return avg_entry / max(abs(1. + maintenance_margin), ZERO_TOL)

    def would_change_side(self, qty):
        return self.signed_qty * (qty + self.signed_qty) < ZERO_TOL
