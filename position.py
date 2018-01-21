from sympy import sign

from constants import ZERO_TOL


class Position:
    """
    Only isolated margin is supported (see isolated vs cross here: https://www.bitmex.com/app/isolatedMargin)
    It means that when a position is opened, a fixed amount is taken as collateral. Any gains are only credited
    after the position is closed.
    """

    def __init__(self):
        self.buy_qty = 0.
        self.buy_vol = 0.
        self.sell_qty = 0.  # always negative
        self.sell_vol = 0.
        self.realized_pnl = 0.
        self.liq_price = 0.
        self.close_ts = None
        self.side = None
        pass

    def __repr__(self):
        return str(self.to_json())

    def to_json(self):
        params = {
            'buy_qty': str(self.buy_qty),
            'buy_vol': str(self.buy_vol),
            'sell_qty': str(self.sell_qty),
            'sell_vol': str(self.sell_vol),
            'realized_pnl': str(self.realized_pnl),
            'liq_price': str(self.liq_price),
            'close_ts': str(self.close_ts.strftime('%Y-%m-%dT%H:%M:%S') if self.close_ts else "None"),
            'side': str(self.side),  # USD
        }
        return params

    def close_position(self):
        assert self.is_closeable()
        pnl = self.realized_pnl
        self.__init__()
        return pnl

    def update(self, qty, price, multiplier, fee):
        qty = float(qty)
        price = float(price)
        multiplier = float(multiplier)
        fee = float(fee)
        if self.side is None:
            self.side = sign(qty)
        """
        Should be updated on every fill.
        """
        net_qty = self.net_qty()
        if net_qty * (qty + net_qty) < 0:  # changing position direction
            raise ValueError("provided qty changes position side. This case should be handled outside this method.")

        if qty > 0:
            self.buy_qty += qty
            self.buy_vol += qty * price * (1. + fee)
        else:
            self.sell_qty += qty
            self.sell_vol += qty * price * (1. - fee)

        self.liq_price = self.calc_liq_price(multiplier)

        self.realized_pnl = self.calc_realized_pnl(multiplier)

        return self

    def calc_liq_price(self, multiplier):
        eprice = self.entry_price()
        liq_price = eprice * multiplier / (self.side * .75 + multiplier)
        net_qty = self.net_qty()
        if abs(net_qty) > ZERO_TOL and min(self.buy_qty, -self.sell_qty) > ZERO_TOL:
            # limit is calculated such that pnl = -cost = -abs(total qty) / (entry price * multi)
            if self.side > 0:
                limit = (self.buy_qty * (eprice * multiplier ** 2) /
                         (1. + multiplier ** 2) + self.sell_vol) / net_qty
                liq_price = min(liq_price, limit)
            else:
                limit = (self.sell_qty * (eprice * multiplier ** 2) /
                         (1. + multiplier ** 2) - self.buy_vol) / net_qty
                liq_price = max(liq_price, limit)
        elif abs(net_qty) < ZERO_TOL:
            if self.side > 0:
                liq_price = 0.
            else:
                liq_price = float('inf')
        return liq_price

    def calc_realized_pnl(self, multiplier):
        # pnl = Contracts * Multiplier * (1/Entry Price - 1/Exit Price)

        dprice = self.buy_qty / max(self.buy_vol, 1.e-8) - self.sell_qty / min(self.sell_vol, -1.e-8)

        return multiplier * min(abs(self.buy_qty), abs(self.sell_qty)) * dprice

    def entry_price(self):
        # this is not exactly true, but it's a good approximation
        if self.side > 0:
            return self.buy_vol / self.buy_qty
        else:
            return self.sell_vol / self.sell_qty

    def net_qty(self):
        return self.buy_qty + self.sell_qty

    def position(self):
        return self.buy_qty + self.sell_qty

    def is_closeable(self):
        return abs(self.buy_qty + self.sell_qty) < 1.e-10

    def is_open(self):
        return self.side is not None

    def does_change_side(self, qty):
        net_qty = self.net_qty()
        return net_qty * (qty + net_qty) < 0

    def does_reduce_position(self, qty):
        return self.position_change(qty) < 0

    def position_change(self, signed_qty):
        # type: (float) -> float
        """
        :param signed_qty:
        :return: a positive number if signed_qty increase position (positively or negatively) and negative otherwise
        """
        return abs(self.buy_qty + self.sell_qty + signed_qty) - abs(self.buy_qty + self.sell_qty)

