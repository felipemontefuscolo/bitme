import argparse
import base64
import datetime
import json
import logging
import time
import uuid
from collections import defaultdict
from typing import Iterable, Dict, List, Union, Set

import numpy as np
import os
import pandas as pd
import pytz
import requests
import shutil

import sys

from api import ExchangeInterface, PositionInterface
from common import Symbol, REQUIRED_COLUMNS, create_df_for_candles, fix_bitmex_bug, BITCOIN_TO_SATOSHI, OrderCommon, \
    OrderType, TimeInForce, OrderContainerType, get_orders_id, OrderStatus, Fill, FillType
from common.quote import Quote
from common.trade import Trade
from live import errors
from live.auth import APIKeyAuthWithExpires
from live.settings import settings
from live.ws.ws_thread import BitMEXWebsocket
from tactic import TacticInterface
from tactic.TacticTest1 import TacticTest1
from tactic.TacticTest2 import TacticTest2
from tactic.TacticTest3 import TacticTest3
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
    SYMBOLS = list(Symbol)

    def __init__(self, log_dir: str):
        ExchangeInterface.__init__(self)

        self.log_dir = log_dir

        self._init_files(log_dir)

        self.timeout = settings.TIMEOUT

        self.candles = create_df_for_candles()  # type: pd.DataFrame
        self.positions = defaultdict(list)  # type: Dict[Symbol, List[PositionInterface]]
        self.cum_pnl = defaultdict(float)  # type: Dict[Symbol, float]]

        # all orders, opened and closed
        self.orders = dict()  # type: OrderContainerType
        self.open_orders = dict()  # type: OrderContainerType
        self.bitmex_dummy_tactic = BitmexDummyTactic()
        self.fills = dict()  # type: Dict[str, Fill]
        self.user_order_cancels = set()  # type: Set[str]
        self.quotes = {s: Quote(symbol=s) for s in self.SYMBOLS}  # type: Dict[Symbol, Quote]
        self.trades = {s: Trade(symbol=s) for s in self.SYMBOLS}  # type: Dict[Symbol, Trade]

        self.tactics_map = {}  # type: Dict[str, TacticInterface]
        self.trades_subscribers = {}  # type: Dict[str, TacticInterface]
        self.quotes_subscribers = {}  # type: Dict[str, TacticInterface]

        self.logger = logging.getLogger('root')
        self.base_url = settings.BASE_URL
        self.symbol = Symbol.XBTUSD  # TODO: support other symbols
        self.postOnly = settings.POST_ONLY
        if settings.API_KEY is None:
            raise Exception("Please set an API key and Secret to get started. See " +
                            "https://github.com/BitMEX/sample-market-maker/#getting-started for more information.")
        self.apiKey = settings.API_KEY
        self.apiSecret = settings.API_SECRET

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
                         'order': self.on_order,
                         'execution': self.on_fill,
                         'quote': self.on_quote,
                         'trade': self.on_trade}

        self.init_candles()

        self._curl_bitmex(path='order/all', postdict={}, verb='DELETE')

        self.ws.connect(endpoint=self.base_url,
                        symbol=str(self.symbol),
                        should_auth=True,
                        callback_maps=callback_maps)

        pass

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        self.close()
        pass

    def _init_files(self, log_dir):
        self.fills_file = open(os.path.join(log_dir, 'fills.csv'), 'w')
        self.orders_file = open(os.path.join(log_dir, 'orders.csv'), 'w')
        self.pnl_file = open(os.path.join(log_dir, 'pnl.csv'), 'w')
        self.candles_file = open(os.path.join(log_dir, 'candles.csv'), 'w')

        self.fills_file.write(Fill.get_header() + '\n')
        self.orders_file.write(OrderCommon.get_header() + '\n')
        self.pnl_file.write('time,symbol,pnl,cum_pnl\n')
        self.candles_file.write(','.join(REQUIRED_COLUMNS) + '\n')

    def close(self):
        self.fills_file.close()
        self.orders_file.close()
        self.pnl_file.close()
        self.candles_file.close()

    def _log_fill(self, fill: Fill):
        self.fills_file.write(fill.to_line() + '\n')

    def _log_order(self, order: OrderCommon):
        self.orders_file.write(order.to_line() + '\n')

    def _log_and_update_pnl(self, closed_position: PositionInterface):
        pnl = closed_position.realized_pnl
        self.cum_pnl[closed_position.symbol] += closed_position.realized_pnl
        self.pnl_file.write(','.join([str(closed_position.current_timestamp.strftime('%Y-%m-%dT%H:%M:%S')),
                                      closed_position.symbol.name,
                                      str(pnl),
                                      str(self.cum_pnl[closed_position.symbol])])
                            + '\n')

    def _log_candle(self, vals: list):
        self.candles_file.write(','.join([str(i) for i in vals]) + '\n')

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
        self.tactics_map[tactic.id()] = tactic

    def get_tactic_from_order_id(self, order_id: str) -> TacticInterface:
        tactic_id = order_id.split('_')[0]
        return self.tactics_map[tactic_id]

    ######################
    # CALLBACKS
    ######################

    def on_trade(self, raw):
        symbol = Symbol[raw['symbol']]
        trade = self.trades[symbol]
        trade.update_from_bitmex(raw)
        for tactic in self.trades_subscribers.values():
            tactic.handle_trade(trade)

    def on_quote(self, raw):
        symbol = Symbol[raw['symbol']]
        quote = self.quotes[symbol]
        quote.update_from_bitmex(raw)
        for tactic in self.quotes_subscribers.values():
            tactic.handle_quote(quote)

    def on_fill(self, raw):
        fill_id = raw['execID']
        if not raw['avgPx']:
            # bitmex has this first empty fill that I don't know what it is
            # just skipp it
            return
        # self.print_ws_output(raw)
        if fill_id in self.fills:
            # self.fills[fill_id].update_from_bitmex(raw)  # type: Fill
            raise AttributeError("Never got an fill update from bitmex, not expecting getting it now ...")
        else:
            order = self.orders[raw['clOrdID']]
            self.fills[fill_id] = Fill(order=order,
                                       qty_filled=raw['lastQty'],
                                       price_fill=raw['lastPx'],
                                       fill_time=pd.Timestamp(raw['transactTime']),
                                       fill_type=FillType.complete if raw['leavesQty'] == 0 else FillType.partial)

        fill = self.fills[fill_id]
        # this actually works because order's fields are a subset of fill's fields
        fill.order.update_from_bitmex(raw)

        fill.order.tactic.handle_fill(fill)

        self._log_fill(fill)

    def on_order(self, raw):
        try:
            id = raw['clOrdID']
        except KeyError:
            raise AttributeError('all our orders should have a client id ("clOrdID")')

        try:
            order = self.orders[id]
        except KeyError:
            # probably an order from a previous run, let's remove it from the way
            raise AttributeError('the only for that to happen is that our order-id-gen generated the same id'
                                 'of an order in the previous run. The change for this to happen is super low, probably'
                                 'it is an logic error.')

        order.update_from_bitmex(raw)  # type: OrderCommon
        order.confirmed_by_websocket = True

        if order.is_open():
            self.open_orders[order.id] = order
        else:
            try:
                del self.open_orders[order.id]
            except KeyError:
                pass

        if (order.status == OrderStatus.Canceled and order.id not in self.user_order_cancels) \
                or order.status == OrderStatus.Rejected:
            order.tactic.handle_cancel(order)

    def rename_old_order(self, raw):
        old_id = raw['clOrdID']
        any_id = 'remove_' + base64.b64encode(uuid.uuid4().bytes).decode('utf8').rstrip('=\n')
        raw['clOrdID'] = any_id
        raw['origClOrdID'] = old_id
        result = self._curl_bitmex(path='order', postdict=raw, verb='PUT')
        self.logger.info('changed stale order clOrdID={} to {}'.format(old_id, result['clOrdID']))
        return

    def on_tradeBin1m(self, raw):
        row = [raw[j] for j in REQUIRED_COLUMNS]
        self.candles.loc[pd.Timestamp(raw['timestamp'])] = row

        # dead-man's switch
        self._curl_bitmex(path='order/cancelAllAfter', postdict={'timeout': 90000}, verb='POST')

        for tac in self.tactics_map.values():
            tac.handle_1m_candles(self.candles)

        self._log_candle(row)

        # self.print_ws_output(raw)

    def on_position(self, raw: dict):

        symbol = Symbol[raw['symbol']]

        positions = self.positions[symbol]

        print("PRINTING RAW FROM WEBSOCKET")
        print(raw)

        if positions and positions[-1].is_open:
            positions[-1].update_from_bitmex(raw)
            if not raw['isOpen']:
                # if here, it means the position was opened and it just closed
                self._log_and_update_pnl(positions[-1])
        elif raw['isOpen']:
            pos = PositionInterface(symbol)
            self.positions[symbol].append(pos.update_from_bitmex(raw))

    def print_ws_output(self, raw):
        print("PRINTING RAW")
        print(json.dumps(raw, indent=4, sort_keys=True))

    ######################
    # FROM INTERFACE
    ######################

    def subscribe_to_trades(self, tactic):
        self.trades_subscribers[tactic.id] = tactic

    def subscribe_to_quotes(self, tactic):
        self.quotes_subscribers[tactic.id] = tactic

    def get_candles1m(self) -> pd.DataFrame:
        return self.candles

    def current_time(self) -> pd.Timestamp:
        return pd.Timestamp.now()

    def get_quote(self, symbol: Symbol) -> Quote:
        """
        :param symbol:
        :return: dict, example: {"buy": 6630.0, "last": 6633.0, "mid": 6630.0, "sell": 6630.5}
        """
        return self.quotes[symbol]

    def get_last_trade(self, symbol: Symbol):
        return self.trades[symbol]

    @authentication_required
    def get_position(self, symbol: Symbol = None) -> PositionInterface:
        if symbol is None:
            symbol = self.symbol
        path = "position"
        postdict = {'filter': {'symbol': symbol.name}}
        raw = self._curl_bitmex(path=path, postdict=postdict, verb="GET")
        # if raw and raw[-1]['isOpen']:
        #     pos = PositionInterface(symbol)
        #     pos.update_from_bitmex(raw[-1])
        #     return pos
        # return PositionInterface(symbol)
        if raw:
            pos = PositionInterface(symbol)
            pos.update_from_bitmex(raw[-1])
            return pos
        else:
            return PositionInterface(symbol)

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
        self.user_order_cancels = set(ids).union(self.user_order_cancels)
        raws = self._curl_bitmex(path="order", postdict={'clOrdID': ','.join(ids)}, verb="DELETE")

        cancelled = {raw['clOrdID']: self.orders[raw['clOrdID']] for raw in raws}

        self.check_all_closed(cancelled.values())
        for o in cancelled.values():
            self._log_order(o)

        return cancelled

    @authentication_required
    def send_orders(self, orders: List[OrderCommon]) -> List[OrderCommon]:
        # sanity check
        if not all([o.status == OrderStatus.Pending for o in orders]):
            raise ValueError("New orders should have status OrderStatus.Pending")

        for i in range(len(orders)):
            orders[i].id = '{}_{}'.format(orders[i].id,
                                          base64.b64encode(uuid.uuid4().bytes).decode('utf8').rstrip('=\n'))

        if not all([o.id not in self.orders for o in orders]):
            raise ValueError("Orders contain an id that already exists")

        for o in orders:
            self.orders[o.id] = o

        converted = [self.convert_order_to_bitmex(self.check_order_sanity(order)) for order in orders]
        raws = self._curl_bitmex(path='order/bulk', postdict={'orders': converted}, verb='POST')

        result = [self.orders[r['clOrdID']].update_from_bitmex(r) for r in raws]  # type: List[OrderCommon]

        for r in result:
            self._log_order(r)

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
        a = {'leavesQty': order.leaves_qty}
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

                if 'duplicate clordid' in message:
                    self.logger.info("found duplicate id")
                    orders = postdict['orders'] if 'orders' in postdict else postdict

                    IDs = json.dumps({'clOrdID': [order['clOrdID'] for order in orders]})
                    orderResults = self._curl_bitmex('/order', query={'filter': IDs}, verb='GET')

                    raise AttributeError("Found duplicate orders id. Sent request orders are: {}\n"
                                         "Existing orders are: {}".format(orders, orderResults))

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


def get_args(input_args=None):
    parser = argparse.ArgumentParser(description='Simulation')
    parser.add_argument('-l', '--log-dir', type=str, help='log directory', required=True)
    parser.add_argument('-x', '--pref', action='append', help='args for tactics, given in the format "key=value"')

    args = parser.parse_args(args=input_args)

    if args.log_dir is not None:
        if os.path.isfile(args.log_dir):
            raise ValueError(args.log_dir + " is a file")
        args.log_dir = os.path.abspath(args.log_dir)
        if os.path.isdir(args.log_dir):
            shutil.rmtree(args.log_dir)
        os.makedirs(args.log_dir)

    if not args.pref:
        args.pref = list()
    for i in range(len(args.pref)):
        args.pref[i] = args.pref[i].split("=")
    args.pref = dict(args.pref)

    return args


def test_common_1(tactic, sleep_time, input_args):
    args = get_args(input_args)

    with LiveBitMex(args.log_dir) as live:
        live.register_tactic(tactic)
        for tac in live.tactics_map.values():
            tac.init(live, args.pref)

        live.tactics_map[tactic.id()].handle_1m_candles(None)

        for i in range(sleep_time):
            print('sleeping ... ' + str(i))
            time.sleep(1)

        live.ws.log_summary()

    return 0


def test1(input_args=None):
    return test_common_1(TacticTest1(), 10, input_args)


def test2(input_args=None):
    tactic = TacticTest2()
    ret = test_common_1(tactic, 3, input_args)
    time.sleep(1)
    assert tactic.got_the_cancel
    return ret


def test3(input_args=None):
    # assert that tactic doesn't get 'handle_cancel' if the cancel came from the tactic iteself
    args = get_args(input_args)
    tactic = TacticTest3()
    with LiveBitMex(args.log_dir) as live:
        live.register_tactic(tactic)
        for tac in live.tactics_map.values():
            tac.init(live, args.pref)

        live.tactics_map[tactic.id()].handle_1m_candles(None)
        time.sleep(1)
        live.tactics_map[tactic.id()].handle_1m_candles(None)

    assert not tactic.got_the_cancel

    return 0


def test_print_quote(input_args=None):
    args = get_args(input_args)

    with LiveBitMex(args.log_dir) as live:
        for i in range(3):
            print('sleeping ... ' + str(i))
            time.sleep(1)
            print(live.get_quote(Symbol.XBTUSD).__dict__)

        live.ws.log_summary()

    return 0


def test_print_trade(input_args=None):
    args = get_args(input_args)

    with LiveBitMex(args.log_dir) as live:
        for i in range(3):
            print('sleeping ... ' + str(i))
            time.sleep(1)
            print(live.trades[Symbol.XBTUSD])
            # print(live.get_last_trade(Symbol.XBTUSD).__dict__)

        live.ws.log_summary()

    return 0


def main(input_args=None):
    args = get_args(input_args)

    # live = LiveBitMex()
    # print(live.get_candles1m())
    # g = live.get_position()
    # g = None
    # for i in range(120):
    #     time.sleep(1)

    # print('oi')

    # print(type(g))
    # print(g)


if __name__ == "__main__":
    sys.exit(test_print_trade())
