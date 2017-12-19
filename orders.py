import base64
import uuid
from cdecimal import Decimal
import pandas as pd

TWOPLACES = Decimal(10) ** -2
EIGHPLACES = Decimal(10) ** -8


# Order container util
class Orders:
    def __init__(self):
        self.data = dict()  # id -> OrderCommon
        pass

    def __iter__(self):
        return iter(self.data.values())

    def _gen_order_id(self):
        return str('bitme_' + base64.b64encode(uuid.uuid4().bytes).decode('utf8').rstrip('=\n'))

    def size(self):
        return len(self.data)

    def post_limit_order(self, side, price, size, product_id, time_posted):
        id = self._gen_order_id()
        self.data[id] = LimitOrder(id, side, price, size, product_id, time_posted)

    def post_market_order(self, side, size, product_id, time_posted):
        id = self._gen_order_id()
        self.data[id] = MarketOrder(id, side, size, product_id, time_posted)

    def merge(self, orders):
        # type: (Orders) -> None
        self.data.update(orders.data)

    def clean_filled(self):
        self.data = dict([(oid, order) for oid, order in self.data.iteritems() if order.size > 0])

    def remove_no_fund_orders(self, position_coin, position_usd):
        self.data = dict([(i, o) for i, o in self.data.iteritems()
                          if (o.side[0] == 's' and o.size <= position_coin) or
                             (o.side[0] == 'b' and o.size * o.price <= position_usd)])

    def to_csv(self, header=True):
        # type: () -> str
        r = ['time,side,size,price'] if header else []
        for o in self.data.values():
            try:
                price = str(o.price)
            except AttributeError:
                price = 'market'  # market order
            r += [','.join([o.ts.strftime('%Y-%m-%dT%H:%M:%S'), o.side, str(o.size), price])]
        return '\n'.join(r)


class OrderCommon:
    def __init__(self, order_id, side, size, product_id, order_type, time_posted):
        # type: (int, str, int, str, str, pd.Timestamp) -> None
        self.id = order_id
        self.side = side
        self.size = size
        self.type = order_type
        self.ts = time_posted
        self.product_id = product_id

    def fill(self, size):
        filled = min(abs(self.size), abs(size))
        self.size -= filled
        return filled

    def set_filled(self):
        self.size = 0
        pass


class LimitOrder(OrderCommon):
    def __init__(self, order_id, side, price, size, product_id, time_posted):
        OrderCommon.__init__(self, order_id, side, size, product_id, 'limit', time_posted)
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


class MarketOrder(OrderCommon):
    def __init__(self, order_id, side, size, product_id, time_posted):
        OrderCommon.__init__(self, order_id, side, size, product_id, 'market', time_posted)
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


class StopOrder(OrderCommon):
    def __init__(self, order_id, side, price, size, product_id, time_posted):
        OrderCommon.__init__(self, order_id, side, size, product_id, 'stop', time_posted)
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
