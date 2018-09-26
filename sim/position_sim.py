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

    def split_quantity(self, signed_qty: int) -> tuple:
        """
        Split signed_qty in two quantities (qty1 and qty2) such that qty1 + qty2 = signed_qty and update(qty1) will
        never change position side
        :return: tuple with qty1 and qty2
        """
        cur_pos = self.signed_qty
        if cur_pos == 0:
            qty1 = signed_qty
        elif cur_pos > 0:
            qty1 = max(-cur_pos, signed_qty)
        else:
            qty1 = min(-cur_pos, signed_qty)
        qty2 = signed_qty - qty1

        assert (qty1 + self.signed_qty) * self.signed_qty >= 0, "logic error"
        if abs(qty2) >= 1:
            assert qty1 + self.signed_qty == 0, "logic error"

        return qty1, qty2

    def update(self, signed_qty, *args, **kwargs):
        """
        This version of update supports position side change
        """
        qty1, qty2 = self.split_quantity(signed_qty)

        self.update_unsafe(qty1, *args, **kwargs)
        if abs(qty2) >= 1:
            assert not self.is_open
            self.update_unsafe(qty2, *args, **kwargs)

    def update_unsafe(self, signed_qty, price, leverage, current_timestamp, fee, mark_price=None) -> PositionInterface:

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
            assert abs(signed_qty) > ZERO_TOL, "signed_qty is {}".format(signed_qty)
            self.side = int(sign(signed_qty))

        new_qty = self.signed_qty + signed_qty

        if self.signed_qty * new_qty < -ZERO_TOL:
            raise ValueError("It does not support side change. Current qty={}, new_qty={}, fill_qty={}"
                             .format(self.signed_qty, new_qty, signed_qty))

        fake_price = price * (1. + int(sign(signed_qty)) * fee)

        if sign(signed_qty) == sign(self.signed_qty) or just_opened:
            self.avg_entry_price = (self.avg_entry_price * self.signed_qty + fake_price * signed_qty) / \
                                   (self.signed_qty + signed_qty + 1.e-4)
        else:
            # TODO: take into account rebates. In this case, the line below could be outside the condition if/else
            d_price = fake_price - self.avg_entry_price
            self.realized_pnl += d_price / (self.avg_entry_price * fake_price) * self.leverage * abs(
                signed_qty) * self.side

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
            if self.on_position_close:
                self.on_position_close(self)
            self.__init__(self.symbol, self.on_position_close)

        return self

    @staticmethod
    def _calc_liq_price(avg_entry, curr_qty, lev, funding_rate=0.001):
        # TODO: support funding rate
        maintenance_margin = float(sign(curr_qty)) * (1. / lev - (0.5 + 0.075 + funding_rate) / 100)
        return avg_entry / max(abs(1. + maintenance_margin), ZERO_TOL)

    def would_change_side(self, signed_qty):
        return self.signed_qty * (signed_qty + self.signed_qty) < ZERO_TOL
