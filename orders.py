from cdecimal import Decimal

TWOPLACES = Decimal(10) ** -2
EIGHPLACES = Decimal(10) ** -8


class Orders:
    order_id = 0

    def __init__(self):
        self.data = []
        pass

    def __iter__(self):
        return iter(self.data)

    def size(self):
        return len(self.data)

    def post_limit_order(self, side, price, size, product_id, time_posted):
        order = LimitOrder(self.order_id, side, price, size, product_id, time_posted)
        self.order_id += 1
        self.data += [order]

    def merge(self, orders):
        # type: (Orders) -> None
        self.data += orders.data

    def clean_filled(self):
        self.data = [order for order in self.data if order.size > 0]

    def printf(self):
        print("buys")
        for i in self.data:
            if i.side == 'buy':
                print(i.to_json())
        print("\nsells")
        for i in self.data:
            if i.side == 'sell':
                print(i.to_json())

    def remove_no_fund_orders(self, position_coin, position_usd):
        self.data = [o for o in self.data if (o.side[0] == 's' and o.size <= position_coin) or
                                             (o.side[0] == 'b' and o.size * o.price <= position_usd)]


class _OrderCommon:
    def __init__(self, order_id, side, size, product_id, order_type, time_posted):
        self.order_id = order_id
        self.side = side
        self.size = size
        self.order_type = order_type
        self.ts = time_posted
        self.product_id = product_id
        pass

    def fill(self, size):
        filled = min(self.size, abs(size))
        self.size -= filled
        return filled


class LimitOrder(_OrderCommon):
    def __init__(self, order_id, side, price, size, product_id, time_posted):
        _OrderCommon.__init__(self, order_id, side, size, product_id, 'limit', time_posted)
        self.price = price
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

    @staticmethod
    def is_ioc():
        return False


class MarketOrder(_OrderCommon):
    def __init__(self, order_id, side, size, product_id, time_posted):
        _OrderCommon.__init__(self, order_id, side, size, product_id, 'market', time_posted)
        raise RuntimeError("Not implemented. Need to implement full depth book first")
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

    @staticmethod
    def is_ioc():
        return True


class StopOrder(_OrderCommon):
    def __init__(self, order_id, side, price, size, product_id, time_posted):
        _OrderCommon.__init__(self, order_id, side, size, product_id, 'stop', time_posted)
        self.price = price
        raise RuntimeError("Not implemented. Need to implement full depth book first")
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

    @staticmethod
    def is_ioc():
        return False


def to_str(number, precision=TWOPLACES):
    return str(Decimal(str(number)).quantize(precision))
