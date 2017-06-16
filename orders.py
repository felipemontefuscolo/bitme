from cdecimal import Decimal

TWOPLACES = Decimal(10) ** -2
EIGHPLACES = Decimal(10) ** -8


class Orders:
    def __init__(self):
        self.buys = []
        self.sells = []
        pass

    def post_limit_order(self, side, price, size, product_id, time_posted):
        order = LimitOrder(side, price, size, product_id, time_posted)
        if side == "buy":
            self.buys += [order]
        else:
            self.sells += [order]


class LimitOrder:
    def __init__(self, side, price, size, product_id, time_posted):
        self.side = side
        self.price = price
        self.size = size
        self.product_id = product_id
        self.ts = time_posted
        pass

    def __repr__(self):
        return str(self.to_json())

    def __str__(self):
        return str(self.to_json())

    def to_json(self):
        params = {
            'side': self.side,
            'price': to_str(self.price, TWOPLACES),  # USD
            'size': to_str(self.size, EIGHPLACES),  # BTC
            'post_only': 'true',
            'product_id': self.product_id,
            'type': 'limit',
            'overdraft_enabled': 'true'
        }
        return params


class MarketOrder:
    def __init__(self, side, size, product_id, time_posted):
        self.side = side
        self.size = size
        self.product_id = product_id
        self.ts = time_posted
        pass

    def __repr__(self):
        return str(self.to_json())

    def __str__(self):
        return str(self.to_json())

    def to_json(self):
        params = {
            'side': self.side,
            'size': to_str(self.size, EIGHPLACES),  # BTC
            'product_id': self.product_id,
            'type': 'market',
            'overdraft_enabled': 'true'
        }
        return params


class StopOrder:
    def __init__(self, side, price, size, product_id, time_posted):
        self.side = side
        self.price = price
        self.size = size
        self.product_id = product_id
        self.ts = time_posted
        pass

    def __repr__(self):
        return str(self.to_json())

    def __str__(self):
        return str(self.to_json())

    def to_json(self):
        params = {
            'side': self.side,
            'price': to_str(self.price, TWOPLACES),  # USD
            'size': to_str(self.size, EIGHPLACES),  # BTC
            'product_id': self.product_id,
            'type': 'stop',
            'overdraft_enabled': 'true'
        }
        return params


def to_str(number, precision=TWOPLACES):
    return str(Decimal(str(number)).quantize(precision))
