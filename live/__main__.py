import base64
import datetime
import json
import logging
import time
import uuid
from collections import defaultdict
from typing import Iterable, Dict, List, Union

import numpy as np
import pandas as pd
import pytz
import requests

from api import ExchangeInterface, PositionInterface
from common import Symbol, REQUIRED_COLUMNS, create_df_for_candles, fix_bitmex_bug, BITCOIN_TO_SATOSHI, OrderCommon, \
    OrderType, TimeInForce, OrderContainerType, get_orders_id, OrderStatus
from live import errors
from live.auth import APIKeyAuthWithExpires
from live.settings import settings
from live.ws.ws_thread import BitMEXWebsocket
from tactic import TacticInterface
from tactic.bitmex_dummy_tactic import BitmexDummyTactic
from utils import log

MAX_NUM_CANDLES_BITMEX = 500  # bitmex REST API limit
INITIAL_NUM_CANDLES = 10

logger = log.setup_custom_logger('root')


def authentication_required(fn):
    """Annotation for methods that require auth."""

    def wrapped(self, *args, **kwargs):
        if not self.apiKey:
            msg = "You must be authenticated to use this method"
            raise errors.AuthenticationError(msg)
        else:
            return fn(self, *args, **kwargs)

    return wrapped


class LiveBitMex(ExchangeInterface):

    def __init__(self):
        ExchangeInterface.__init__(self)
        self.span = 10  # minutes;
        assert self.span <= MAX_NUM_CANDLES_BITMEX

        self.timeout = settings.TIMEOUT

        self.candles = create_df_for_candles()  # type: pd.DataFrame
        self.positions = defaultdict(
            lambda: list())  # type: Dict[Symbol, List[PositionInterface]]

        # all orders, opened and closed
        self.orders = dict()  # type: OrderContainerType
        self.open_orders = dict()  # type: OrderContainerType
        self.bitmex_dummy_tactic = BitmexDummyTactic()

        self.tactics = []

        self.logger = logging.getLogger('root')
        self.base_url = settings.BASE_URL
        self.symbol = Symbol.XBTUSD  # TODO: support other symbols
        self.postOnly = settings.POST_ONLY
        if settings.API_KEY is None:
            raise Exception("Please set an API key and Secret to get started. See " +
                            "https://github.com/BitMEX/sample-market-maker/#getting-started for more information.")
        self.apiKey = settings.API_KEY
        self.apiSecret = settings.API_SECRET
        if len(settings.ORDERID_PREFIX) > 13:
            raise ValueError("settings.ORDERID_PREFIX must be at most 13 characters long!")
        self.orderIDPrefix = settings.ORDERID_PREFIX
        self.retries = 0  # initialize counter

        # Prepare HTTPS session
        self.session = requests.Session()
        # These headers are always sent
        self.session.headers.update({'user-agent': 'liquidbot-' + 'alpha'})
        self.session.headers.update({'content-type': 'application/json'})
        self.session.headers.update({'accept': 'application/json'})

        # Create websocket for streaming data
        self.ws = BitMEXWebsocket()

        # dict of subscription -> callback
        callback_maps = {'tradeBin1m': self.on_tradeBin1m,
                         'position': self.on_position,
                         'order': self.on_order}

        self.init_candles()

        self.ws.connect(endpoint=self.base_url,
                        symbol=str(self.symbol),
                        should_auth=True,
                        callback_maps=callback_maps)

        pass

    def init_candles(self):
        end = pd.Timestamp.now(tz=pytz.UTC).floor(freq='1min')
        start = end - pd.Timedelta(minutes=INITIAL_NUM_CANDLES)
        query = {'binSize': '1m',
                 'partial': False,
                 'symbol': str(self.symbol),
                 'reversed': False,
                 'startTime': str(start),
                 'endTime': str(end)
                 }
        raw = self._curl_bitmex(path='trade/bucketed', query=query, verb='GET')
        for candle in raw:
            self.candles.loc[pd.Timestamp(candle['timestamp'])] = [candle[j] for j in REQUIRED_COLUMNS]
        if self.candles.index[0] > self.candles.index[1]:
            self.candles = self.candles[-1::-1]
        self.candles = fix_bitmex_bug(self.candles)

    def register_tactic(self, tactic: TacticInterface):
        self.tactics.append(tactic)

    ######################
    # CALLBACKS
    ######################

    def on_order(self, raw):
        id = raw.get('clOrdID')
        if id in self.orders:
            order = self.orders[id]
            order.update_from_bitmex(raw)
        else:
            order = OrderCommon(symbol=Symbol[raw.get('symbol')], type=OrderType[raw['ordType']],
                                tactic=self.bitmex_dummy_tactic)
            order.id = raw.get('clOrdID')
            order.bitmex_id = raw['orderID']
            order.update_from_bitmex(raw)
            self.orders[order.id] = order

        if order.is_open():
            self.open_orders[order.id] = order
        else:
            try:
                del self.open_orders[order.id]
            except KeyError:
                pass

    def on_tradeBin1m(self, raw):
        self.candles.loc[pd.Timestamp(raw['timestamp'])] = [raw[j] for j in REQUIRED_COLUMNS]
        # self.print_ws_output(raw)

    def on_position(self, raw: dict):
        if not raw:
            return
        if not raw.get('symbol'):
            return
        symbol = Symbol.value_of(raw['symbol'])
        if symbol is None:
            return

        pos = PositionInterface(self.symbol)
        pos.avg_entry_price = raw.get('avgEntryPrice')  # type: float
        pos.break_even_price = raw.get('breakEvenPrice')  # type: float
        pos.liquidation_price = raw.get('liquidationPrice')  # type: float
        pos.leverage = raw.get('leverage')  # type: int
        pos.current_qty = raw.get('currentQty')  # type: float
        pos.side = None if pos.current_qty is None else (+1 if pos.current_qty >= 0 else -1)  # type: int
        pos.realized_pnl = float(raw.get('realisedPnl', float('nan'))) / BITCOIN_TO_SATOSHI  # type: float
        pos.is_open = raw.get('isOpen')  # type: bool
        # TODO: those timestamps don't seem accurate! maybe use our own timestamp?
        pos.current_timestamp = pd.Timestamp(raw.get('currentTimestamp'))  # type: pd.Timestamp
        pos.open_timestamp = pd.Timestamp(raw.get('openingTimestamp'))  # type: pd.Timestamp
        assert pos.current_timestamp >= pos.open_timestamp

        self.positions[symbol].append(pos)

    def print_ws_output(self, raw):
        print(json.dumps(raw, indent=4, sort_keys=True))

    ######################
    # FROM INTERFACE
    ######################

    def get_candles1m(self) -> pd.DataFrame:
        return self.candles

    def current_time(self) -> pd.Timestamp:
        return pd.Timestamp.now()

    def get_tick_info(self, symbol=None) -> dict:
        """
        :param symbol:
        :return: dict, example: {"buy": 6630.0, "last": 6633.0, "mid": 6630.0, "sell": 6630.5}
        """
        if symbol is None:
            symbol = str(self.symbol)
        return self.ws.get_ticker(symbol)

    @authentication_required
    def get_position(self, symbol: Symbol = None) -> PositionInterface:
        if symbol is None:
            symbol = self.symbol
        return self.positions[symbol][-1]

    @authentication_required
    def get_closed_positions(self, symbol: Symbol = None) -> List[PositionInterface]:
        if symbol is None:
            symbol = self.symbol
        return [pos for pos in self.positions[symbol] if (not pos.is_open) and pos.avg_entry_price]

    @authentication_required
    def set_leverage(self, symbol, leverage, rethrow_errors=False):
        """Set the leverage on an isolated margin position"""
        path = "position/leverage"
        postdict = {
            'symbol': symbol,
            'leverage': leverage
        }
        return self._curl_bitmex(path=path, postdict=postdict, verb="POST", rethrow_errors=rethrow_errors)

    @authentication_required
    def cancel_orders(self, orders: Union[OrderContainerType, List[OrderCommon], List[str]]) -> OrderContainerType:
        """
        :return: Dict of cancelled orders
        """
        # TODO: add a warning if the order ids passed by the user don't exist
        ids = get_orders_id(orders)
        if not ids:
            return dict()
        raws = self._curl_bitmex(path="order", postdict={'clOrdID': ','.join(ids)}, verb="DELETE")

        cancelled = {raw['clOrdID']: self.orders[raw['clOrdID']] for raw in raws}

        self.check_all_closed(cancelled.values())

        return cancelled

    @authentication_required
    def post_orders(self, orders: List[OrderCommon]) -> List[OrderCommon]:
        # sanity check
        if not all([o.status == OrderStatus.Pending for o in orders]):
            raise ValueError("New orders should have status OrderStatus.Pending")

        if not all([o.id not in self.orders for o in orders]):
            raise ValueError("Orders contain an id that already exists")

        for o in orders:
            self.orders[o.id] = o

        converted = [self.convert_order_to_bitmex(self.check_order_sanity(order)) for order in orders]
        raws = self._curl_bitmex(path='order/bulk', postdict={'orders': converted}, verb='POST')

        result = [self.orders[r['clOrdID']].update_from_bitmex(r) for r in raws]  # type: List[OrderCommon]

        return result

    @authentication_required
    def get_opened_orders(self, symbol=None) -> OrderContainerType:
        return self.open_orders

    ######################
    # END INTERFACE
    ######################

    def check_all_closed(self, orders_cancelled: Iterable[OrderCommon]) -> None:
        # expecting orders_cancelled to be changed externally by the websocket
        t = 0.
        n_opened = sum(order.is_open() for order in orders_cancelled)
        while n_opened > 0:
            time.sleep(0.1)
            t += 0.1
            n_opened = sum(order.is_open() for order in orders_cancelled)
            if t >= self.timeout / 2:
                opened = [order.id for order in orders_cancelled if order.is_open()]
                raise TimeoutError(
                    "Orders cancelled through http (ids={}) were not confirmed in websocket within {} seconds".format(
                        ','.join(opened), self.timeout / 2))

    @staticmethod
    def check_order_sanity(order: OrderCommon) -> OrderCommon:
        assert order.id
        return order

    @staticmethod
    def convert_order_to_bitmex(order: OrderCommon) -> dict:
        a = {}
        if order.id:
            a['clOrdID'] = order.id
        if order.symbol:
            a['symbol'] = order.symbol.name
        if order.signed_qty and not np.isnan(order.signed_qty):
            a['orderQty'] = order.signed_qty
        if order.price and not np.isnan(order.price):
            a['price'] = order.price
        if order.stop_price and not np.isnan(order.stop_price):
            a['stopPx'] = order.stop_price
        if order.linked_order_id:
            assert order.contingency_type
            a['clOrdLinkID'] = str(order.linked_order_id)
        if order.type:
            a['ordType'] = order.type.name
            if order.type == OrderType.Limit:
                assert a.get('price')
                a['execInst'] = 'ParticipateDoNotInitiate'  # post-only, postonly, post only
        if order.time_in_force:
            a['timeInForce'] = order.time_in_force.name
        else:
            if order.type == OrderType.Market:
                a['timeInForce'] = TimeInForce.FillOrKill.name
            elif order.type == OrderType.Limit or OrderType.Stop:
                a['timeInForce'] = TimeInForce.GoodTillCancel.name
        if order.contingency_type:
            assert order.linked_order_id
            a['contingencyType'] = order.contingency_type.name
        return a

    def instruments(self, filter=None):
        query = {}
        if filter is not None:
            query['filter'] = json.dumps(filter)
        return self._curl_bitmex(path='instrument', query=query, verb='GET')

    def recent_trades(self):
        """Get recent trades.

                Returns
                -------
                A list of dicts:
                      {u'amount': 60,
                       u'date': 1306775375,
                       u'price': 8.7401099999999996,
                       u'tid': u'93842'},

                """
        return self.ws.recent_trades()

    @authentication_required
    def funds(self):
        """Get your current balance."""
        return self.ws.funds()

    @authentication_required
    def place_order(self, quantity, price):
        """Place an order."""
        if price < 0:
            raise Exception("Price must be positive.")

        endpoint = "order"
        # Generate a unique clOrdID with our prefix so we can identify it.
        clOrdID = self.orderIDPrefix + base64.b64encode(uuid.uuid4().bytes).decode('utf8').rstrip('=\n')
        postdict = {
            'symbol': self.symbol,
            'orderQty': quantity,
            'price': price,
            'clOrdID': clOrdID
        }
        return self._curl_bitmex(path=endpoint, postdict=postdict, verb="POST")

    @authentication_required
    def amend_bulk_orders(self, orders):
        """Amend multiple orders."""
        # Note rethrow; if this fails, we want to catch it and re-tick
        return self._curl_bitmex(path='order/bulk', postdict={'orders': orders}, verb='PUT', rethrow_errors=True)

    @authentication_required
    def create_bulk_orders(self, orders):
        """Create multiple orders."""
        for order in orders:
            order['clOrdID'] = self.orderIDPrefix + base64.b64encode(uuid.uuid4().bytes).decode('utf8').rstrip('=\n')
            order['symbol'] = self.symbol
            if self.postOnly:
                order['execInst'] = 'ParticipateDoNotInitiate'
        return self._curl_bitmex(path='order/bulk', postdict={'orders': orders}, verb='POST')

    @authentication_required
    def open_orders(self):
        """Get open orders."""
        return self.ws.open_orders(self.orderIDPrefix)

    @authentication_required
    def http_open_orders(self):
        """Get open orders via HTTP. Used on close to ensure we catch them all."""
        path = "order"
        orders = self._curl_bitmex(
            path=path,
            query={
                'filter': json.dumps({'ordStatus.isTerminated': False, 'symbol': self.symbol}),
                'count': 500
            },
            verb="GET"
        )
        # Only return orders that start with our clOrdID prefix.
        return [o for o in orders if str(o['clOrdID']).startswith(self.orderIDPrefix)]

    @authentication_required
    def cancel(self, orderID):
        """Cancel an existing order."""
        path = "order"
        postdict = {
            'orderID': orderID,
        }
        return self._curl_bitmex(path=path, postdict=postdict, verb="DELETE")

    @authentication_required
    def withdraw(self, amount, fee, address):
        path = "user/requestWithdrawal"
        postdict = {
            'amount': amount,
            'fee': fee,
            'currency': 'XBt',
            'address': address
        }
        return self._curl_bitmex(path=path, postdict=postdict, verb="POST", max_retries=0)

    def _curl_bitmex(self, path, query=None, postdict=None, timeout=None, verb=None, rethrow_errors=False,
                     max_retries=None):
        """Send a request to BitMEX Servers."""
        # Handle URL
        url = self.base_url + path

        if timeout is None:
            timeout = self.timeout

        # Default to POST if data is attached, GET otherwise
        if not verb:
            verb = 'POST' if postdict else 'GET'

        # By default don't retry POST or PUT. Retrying GET/DELETE is okay because they are idempotent.
        # In the future we could allow retrying PUT, so long as 'leavesQty' is not used (not idempotent),
        # or you could change the clOrdID (set {"clOrdID": "new", "origClOrdID": "old"}) so that an amend
        # can't erroneously be applied twice.
        if max_retries is None:
            max_retries = 0 if verb in ['POST', 'PUT'] else 3

        # Auth: API Key/Secret
        auth = APIKeyAuthWithExpires(self.apiKey, self.apiSecret)

        def exit_or_throw(e):
            if rethrow_errors:
                raise e
            else:
                exit(1)

        def retry():
            self.retries += 1
            if self.retries > max_retries:
                raise Exception("Max retries on %s (%s) hit, raising." % (path, json.dumps(postdict or '')))
            return self._curl_bitmex(path, query, postdict, timeout, verb, rethrow_errors, max_retries)

        # Make the request
        response = None
        try:
            self.logger.info("sending req to %s: %s" % (url, json.dumps(postdict or query or '')))
            req = requests.Request(verb, url, json=postdict, auth=auth, params=query)
            prepped = self.session.prepare_request(req)
            response = self.session.send(prepped, timeout=timeout)
            # Make non-200s throw
            response.raise_for_status()

        except requests.exceptions.HTTPError as e:
            if response is None:
                raise e

            # 401 - Auth error. This is fatal.
            if response.status_code == 401:
                self.logger.error("API Key or Secret incorrect, please check and restart.")
                self.logger.error("Error: " + response.text)
                if postdict:
                    self.logger.error(postdict)
                # Always exit, even if rethrow_errors, because this is fatal
                exit(1)

            # 404, can be thrown if order canceled or does not exist.
            elif response.status_code == 404:
                if verb == 'DELETE':
                    self.logger.error("Order not found: %s" % postdict['orderID'])
                    return
                self.logger.error("Unable to contact the BitMEX API (404). " +
                                  "Request: %s \n %s" % (url, json.dumps(postdict)))
                exit_or_throw(e)

            # 429, ratelimit; cancel orders & wait until X-Ratelimit-Reset
            elif response.status_code == 429:
                self.logger.error("Ratelimited on current request. Sleeping, then trying again. Try fewer " +
                                  "order pairs or contact support@bitmex.com to raise your limits. " +
                                  "Request: %s \n %s" % (url, json.dumps(postdict)))

                # Figure out how long we need to wait.
                ratelimit_reset = response.headers['X-Ratelimit-Reset']
                to_sleep = int(ratelimit_reset) - int(time.time())
                reset_str = datetime.datetime.frompd.Timestamp(int(ratelimit_reset)).strftime('%X')

                # We're ratelimited, and we may be waiting for a long time. Cancel orders.
                self.logger.warning("Canceling all known orders in the meantime.")
                self.cancel([o.bitmex_id for o in self.open_orders.values()])

                self.logger.error("Your ratelimit will reset at %s. Sleeping for %d seconds." % (reset_str, to_sleep))
                time.sleep(to_sleep)

                # Retry the request.
                return retry()

            # 503 - BitMEX temporary downtime, likely due to a deploy. Try again
            elif response.status_code == 503:
                self.logger.warning("Unable to contact the BitMEX API (503), retrying. " +
                                    "Request: %s \n %s" % (url, json.dumps(postdict)))
                time.sleep(3)
                return retry()

            elif response.status_code == 400:
                error = response.json()['error']
                message = error['message'].lower() if error else ''

                # Duplicate clOrdID: that's fine, probably a deploy, go get the order(s) and return it
                if 'duplicate clordid' in message:
                    orders = postdict['orders'] if 'orders' in postdict else postdict

                    IDs = json.dumps({'clOrdID': [order['clOrdID'] for order in orders]})
                    orderResults = self._curl_bitmex('/order', query={'filter': IDs}, verb='GET')

                    print("!!!!!!!!!!!!!!!!!  found duplicate ids !!!!!!!!!!!!!!!!!")
                    print("postdict = {}".format(postdict))
                    print("orders = {}".format(orders))
                    print("orderResults = {}".format(orderResults))

                    for i, order in enumerate(orderResults):
                        assert 'orderQty' in order

                        if (abs(order['leavesQty']) != abs(postdict['leavesQty']) or
                                order['side'] != postdict['side'] or
                                order.get('price') != postdict.get('price') or
                                order['symbol'] != postdict['symbol']):
                            raise Exception(
                                'Attempted to recover from duplicate clOrdID, but order returned from API ' +
                                'did not match POST.\nPOST data: %s\nReturned order: %s' % (
                                    json.dumps(orders[i]), json.dumps(order)))
                    # All good
                    return orderResults

                elif 'insufficient available balance' in message:
                    self.logger.error('Account out of funds. The message: %s' % error['message'])
                    exit_or_throw(Exception('Insufficient Funds'))

            # If we haven't returned or re-raised yet, we get here.
            self.logger.error("Unhandled Error: %s: %s" % (e, response.text))
            self.logger.error("Endpoint was: %s %s: %s" % (verb, path, json.dumps(postdict)))
            exit_or_throw(e)

        except requests.exceptions.Timeout as e:
            # Timeout, re-run this request
            self.logger.warning("Timed out on request: %s (%s), retrying..." % (path, json.dumps(postdict or '')))
            return retry()

        except requests.exceptions.ConnectionError as e:
            self.logger.warning("Unable to contact the BitMEX API (%s). Please check the URL. Retrying. " +
                                "Request: %s %s \n %s" % (e, url, json.dumps(postdict)))
            time.sleep(1)
            return retry()

        # Reset retry counter on success
        self.retries = 0

        return response.json()


def test1():
    live = LiveBitMex()
    orders = live.post_orders([OrderCommon(symbol=Symbol.XBTUSD,
                                           type=OrderType.Limit,
                                           tactic=BitmexDummyTactic(),
                                           signed_qty=-100,
                                           price=20000.5)])
    if orders:
        assert orders[0].is_open()
    time.sleep(.5)

    oo = live.get_opened_orders()
    assert len(oo) == 1

    for oid, o in oo.items():
        print(o)

    exit(0)


if __name__ == "__main__":
    test1()

    # live = LiveBitMex()
    # print(live.get_candles1m())
    # g = live.get_position()
    # g = None
    # for i in range(120):
    #     time.sleep(1)

    # print('oi')

    # print(type(g))
    # print(g)
