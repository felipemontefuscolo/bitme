import argparse
import base64
import copy
import datetime
import json
import logging
import os
import queue
import shutil
import sys
import threading
import time
import uuid
from collections import defaultdict
from typing import Iterable, Dict, List, Union, Set

import numpy as np
import pandas as pd
import pytz
import requests

from api import ExchangeInterface, PositionInterface
from common import Symbol, REQUIRED_COLUMNS, create_df_for_candles, fix_bitmex_bug, OrderCommon, \
    OrderType, OrderContainerType, get_orders_id, OrderStatus, Fill, BITCOIN_TO_SATOSHI
from common.quote import Quote
from common.trade import Trade
from live import errors
from live.auth import APIKeyAuthWithExpires
from live.settings import settings
from live.tactic_event_handler import TacticEventHandler
from live.ws.ws_thread import BitMEXWebsocket
from tactic import TacticInterface
from tactic.TacticLimitOrderTest import TacticLimitOrderTest
from tactic.TacticMarketOrderTest import TacticMarketOrderTest
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

    def __init__(self, log_dir: str, tac_prefs: dict, run_time_seconds=None):
        ExchangeInterface.__init__(self)

        self.exc_info = None  # if a thread needs to throws, it should populate this variable
        self.run_loop_in_main_thread = True
        self.log_dir = log_dir
        self.run_time_seconds = run_time_seconds
        self.tac_prefs = tac_prefs
        self.started = False
        self.finished = False
        self.threads = list()  # first thread is the run loop, the rest are the tactics loop

        self.timeout = settings.TIMEOUT

        self.candles = create_df_for_candles()  # type: pd.DataFrame
        self.positions = defaultdict(list)  # type: Dict[Symbol, List[PositionInterface]]
        self.cum_pnl = defaultdict(float)  # type: Dict[Symbol, float]
        self.pnl_history = defaultdict(list)  # type: Dict[Symbol, List[float]]
        self.last_margin = dict()  # type: Dict[str, dict]
        self.last_closed_margin = dict()  # type: Dict[str, dict]

        # opened orders, but can hold closed/filled orders temporarily
        self.all_orders = dict()  # type: OrderContainerType
        self.bitmex_dummy_tactic = BitmexDummyTactic()
        self.solicited_cancels = set()  # type: Set[str]
        self.quotes = {s: Quote(symbol=s) for s in self.SYMBOLS}  # type: Dict[Symbol, Quote]
        self.trades = {s: Trade(symbol=s) for s in self.SYMBOLS}  # type: Dict[Symbol, Trade]

        self.tactics_map = {}  # type: Dict[str, TacticInterface]
        self.tactic_event_handlers = {}  # type: Dict[str, TacticEventHandler]

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
        self.events_queue = queue.Queue()
        self.ws = BitMEXWebsocket()

        self.callback_maps = {
            # Bitmex websocket names
            'tradeBin1m': self.on_tradeBin1m,
            'position': self.on_position,
            'order': self.on_order,
            'execution': self.on_fill,
            'quote': self.on_quote,
            'trade': self.on_trade,
            'margin': self.on_margin}

    def __enter__(self):
        self.start_main_loop()
        return self

    def __exit__(self, type, value, traceback):
        self.end_main_loop()
        pass

    def _init_files(self, log_dir):
        self.fills_file = open(os.path.join(log_dir, 'fills.csv'), 'w')
        self.orders_file = open(os.path.join(log_dir, 'orders.csv'), 'w')
        self.pnl_file = open(os.path.join(log_dir, 'pnl.csv'), 'w')
        self.candles_file = open(os.path.join(log_dir, 'candles.csv'), 'w')

        self.fills_file.write(Fill.get_header() + '\n')
        self.orders_file.write(OrderCommon.get_header() + '\n')
        self.pnl_file.write('time,symbol,pnl,cum_pnl\n')
        self.candles_file.write(','.join(['timestamp'] + REQUIRED_COLUMNS) + '\n')

    def close_files(self):
        assert self.finished
        self.fills_file.close()
        self.orders_file.close()
        self.pnl_file.close()
        self.candles_file.close()

    def _log_fill(self, fill: Fill):
        self.fills_file.write(fill.to_line() + '\n')

    def _log_order(self, order: OrderCommon):
        self.orders_file.write(order.to_line() + '\n')

    # def _log_and_update_pnl(self, closed_position: PositionInterface):
    #     pnl = closed_position.realized_pnl
    #     self.cum_pnl[closed_position.symbol] += closed_position.realized_pnl
    #     self.pnl_file.write(','.join([str(closed_position.current_timestamp.strftime('%Y-%m-%dT%H:%M:%S')),
    #                                   closed_position.symbol.name,
    #                                   str(pnl),
    #                                   str(self.cum_pnl[closed_position.symbol])])
    #                         + '\n')

    def _log_and_update_pnl(self, pnl: float, symbol: Symbol, timestamp: pd.Timestamp):
        self.pnl_history[symbol].append(pnl)
        self.cum_pnl[symbol] += pnl
        self.pnl_file.write(','.join([str(timestamp.strftime('%Y-%m-%dT%H:%M:%S')),
                                      symbol.name,
                                      str(pnl),
                                      str(self.cum_pnl[symbol])])
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
        # TODO: we shouldn't allow tactic trading the same symbol

        self.tactics_map[tactic.id()] = tactic
        tactic.initialize(self, self.tac_prefs)
        tac_event_handler = TacticEventHandler(tactic, self)

        t = threading.Thread(target=tac_event_handler.run_forever)
        t.daemon = True
        t.start()

        self.threads.append(t)
        self.tactic_event_handlers[tactic.id()] = tac_event_handler

    def throw_exception_from_tactic(self, exc_info):
        self.exc_info = exc_info

    def get_tactic_from_order_id(self, order_id: str) -> TacticInterface:
        tactic_id = order_id.split('_')[0]
        return self.tactics_map[tactic_id]

    def start_main_loop(self, run_time_seconds=None):
        if self.started:
            raise AttributeError('Live already started')
        if run_time_seconds:
            self.run_time_seconds = run_time_seconds
        self.started = True

        self.init_candles()
        self._curl_bitmex(path='order/all', postdict={}, verb='DELETE')
        self._init_files(self.log_dir)

        self.ws.connect(endpoint=self.base_url,
                        symbol=str(self.symbol),
                        should_auth=True,
                        events_queue=self.events_queue)

        if self.run_loop_in_main_thread:
            self._run()
        else:
            t = threading.Thread(target=self._run)
            t.daemon = True
            t.start()
            self.threads = [t] + self.threads

    def end_main_loop(self):
        if not self.run_loop_in_main_thread:
            self.threads[0].join()  # wait for the main-loop thread to finish
        self.close_files()

        logger.info("Total pnl (in XBT): ")
        for k, v in self.cum_pnl.items():
            logger.info("{}: {}".format(k, v))

    def _run(self):
        start = time.time()

        while self.run_time_seconds is None or time.time() - start <= self.run_time_seconds:
            event = self.events_queue.get()
            name, action, raw = event
            method = self.callback_maps.get(name)
            if method:
                # if name != 'quote' and name != 'trade':
                #     print("GOT RAW {} {}".format(name, action))
                #     print(raw)
                # if name == 'margin':
                #     print("GOT RAW {} {}".format(name, action))
                #     print(raw)
                method(raw, action)
            if self.exc_info:
                raise self.exc_info[1].with_traceback(self.exc_info[2])

        for tac in self.tactics_map.values():
            tac.finalize()

        self.finished = True
        del self.threads

    ######################
    # CALLBACKS
    ######################

    def on_margin(self, raw, action):
        # margin is the best most reliable way to keep track of the pnl

        last_margin = self.last_margin.get(raw['currency'])
        if last_margin is None:
            assert action == 'partial'
            self.last_margin[raw['currency']] = copy.copy(raw)
            self.last_closed_margin[raw['currency']] = copy.copy(raw)
            return

        last_closed_margin = self.last_closed_margin[raw['currency']]

        if last_margin['riskValue'] > 0 and raw['riskValue'] == 0:
            # assuming that this condition implies that the position was closed
            assert raw['walletBalance'] == raw['withdrawableMargin']
            pnl_satoshi = raw['walletBalance'] - last_closed_margin['walletBalance']
            self._log_and_update_pnl(pnl_satoshi / BITCOIN_TO_SATOSHI,
                                     Symbol.XBTUSD,
                                     pd.Timestamp(raw['timestamp']))
            last_closed_margin.update(raw)

        last_margin.update(raw)
        pass

    def on_trade(self, raw: dict, action: str):
        symbol = Symbol[raw['symbol']]
        trade = self.trades[symbol]
        trade.update_from_bitmex(raw)
        for tac_handler in self.tactic_event_handlers.values():
            tac_handler.queue.put(trade, block=True, timeout=None)

    def on_quote(self, raw: dict, action: str):
        symbol = Symbol[raw['symbol']]
        quote = self.quotes[symbol]
        quote.update_from_bitmex(raw)
        for tac_handler in self.tactic_event_handlers.values():
            tac_handler.queue.put(quote, block=True, timeout=None)

    def on_fill(self, raw: dict, action: str):
        if not raw['avgPx']:
            # bitmex has this first empty fill that I don't know what it is
            # just skipp it
            return
        # self.print_ws_output(raw)

        fill = Fill.create_from_raw(raw)

        order_id = raw.get('clOrdID')

        if order_id:
            tac_name = order_id.split('_')[0]
            self.tactic_event_handlers[tac_name].queue.put(fill, block=True, timeout=None)

        self._log_fill(fill)

    def _notify_cancel_if_the_case(self, order_: OrderCommon):
        if order_.status == OrderStatus.Canceled:
            if order_.client_id not in self.solicited_cancels:
                tac_name = order_.client_id.split('_')[0]
                self.tactic_event_handlers[tac_name].queue.put(order_, block=True, timeout=None)
            else:
                self.solicited_cancels.remove(order_.client_id)

    def on_order(self, raw: dict, action: str):
        # DEV NOTE: this is the only method who can delete orders from self.all_orders
        # DEV NOTE: this is the only method who broadcast orders to tactics

        order_id = raw.get('clOrdID')

        if not order_id:
            self.logger.error('Got an order without "clOrdID", probably a bitmex order (e.g., liquidation)')
            self.logger.error('order: {}'.format(raw))
            raise AttributeError('If we were liquidated, we should change our tactics')

        if OrderStatus[raw['ordStatus']] == OrderStatus.Rejected:
            raise ValueError('A order should never be rejected, please fix the tactic. Order: {}'.format(raw))

        order = self.all_orders.get(order_id)

        if order:
            if action != 'update':
                raise AttributeError('Got action "{}". This should be an "update", since this order already '
                                     'exist in our data and we don\'t expect a "delete"'
                                     .format(action))

            if not order.is_open():
                # This is the case when a closed (e.g. Canceled) order where inserted, now we clean it up in the insert
                # We also assume that bitmex will not update a canceled/closed order twice
                del self.all_orders[order.client_id]
                return

            order.update_from_bitmex(raw)  # type: OrderCommon

            if not order.is_open():
                del self.all_orders[order.client_id]

                self._notify_cancel_if_the_case(order)

        else:
            symbol = Symbol[raw['symbol']]
            type_ = OrderType[raw['ordType']]
            order = OrderCommon(symbol=symbol, type=type_).update_from_bitmex(raw)

            if action != 'insert' and order.is_open():
                raise AttributeError('Got action "{}". This should be an "insert", since this data does\'nt exist '
                                     'in memory and is open. Order: {}'
                                     .format(action, raw))

            # yes, always add an order to the list, even if it is closed. If so, it will be removed in the "update" feed
            self.all_orders[order_id] = order

            self._notify_cancel_if_the_case(order)

    def _rename_prev_run_order(self, raw):
        old_id = raw['clOrdID']
        any_id = 'remove_' + base64.b64encode(uuid.uuid4().bytes).decode('utf8').rstrip('=\n')
        raw['clOrdID'] = any_id
        raw['origClOrdID'] = old_id
        result = self._curl_bitmex(path='order', postdict=raw, verb='PUT')
        self.logger.info('changed stale order clOrdID={} to {}'.format(old_id, result['clOrdID']))
        return

    def on_tradeBin1m(self, raw: dict, action: str):
        row = [raw[j] for j in REQUIRED_COLUMNS]
        timestamp = pd.Timestamp(raw['timestamp'])
        self.candles.loc[timestamp] = row

        # dead-man's switch
        self._curl_bitmex(path='order/cancelAllAfter', postdict={'timeout': 90000}, verb='POST')

        self._log_candle([timestamp] + row)

        for tac_handler in self.tactic_event_handlers.values():
            tac_handler.queue.put(self.candles, block=True, timeout=None)

    def on_position(self, raw: dict, action: str):
        # DEV NOTE: bitmex unfortunately computes pnl,openPosition based on a fixed timestamp instead of
        # since when the position was opened.
        # This implies we have to track those values ourselves and update our position accordingly

        symbol = Symbol[raw['symbol']]

        positions = self.positions[symbol]

        # self._print_ws_output(raw)

        # if positions and positions[-1].is_open:
        #     positions[-1].update_from_bitmex(raw)
        #     if not raw['isOpen']:
        #         # if here, it means the position was opened and it just closed
        #         self._log_and_update_pnl(positions[-1])
        # elif raw['isOpen']:
        #     pos = PositionInterface(symbol)
        #     self.positions[symbol].append(pos.update_from_bitmex(raw))

    def _print_ws_output(self, raw):
        logger.info("PRINTING RAW")
        logger.info(json.dumps(raw, indent=4, sort_keys=True))

    @staticmethod
    def _update_position(pos: PositionInterface, raw: dict):
        assert pos.symbol == Symbol[raw['symbol']]

    ######################
    # FROM INTERFACE
    ######################

    def get_balance_xbt(self) -> float:
        return self.last_margin['XBt']['withdrawableMargin']

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
        raw = self._curl_bitmex(
            path="position",
            query={
                'filter': json.dumps({'symbol': symbol.name})
            },
            verb="GET")
        if raw:
            pos = PositionInterface(symbol)
            pos.update_from_bitmex(raw[-1])
            return pos
        else:
            return PositionInterface(symbol)

    @authentication_required
    def get_pnl_history(self, symbol: Symbol) -> List[float]:
        return copy.copy(self.pnl_history[symbol])

    @authentication_required
    def set_leverage(self, symbol: Symbol, leverage: float) -> None:
        """Set the leverage on an isolated margin position"""
        path = "position/leverage"
        postdict = {
            'symbol': symbol.name,
            'leverage': leverage
        }
        self._curl_bitmex(path=path, postdict=postdict, verb="POST", rethrow_errors=False)

    @authentication_required
    def cancel_orders(self, orders: Union[OrderContainerType, List[OrderCommon], List[str]]) -> OrderContainerType:
        # DEV NOTE: it should NOT change any internal state
        """
        :return: Dict of cancelled orders
        """
        # TODO: add a warning if the order ids passed by the user don't exist
        # DEV NOTE: it should not log order, let on_orders log instead

        ids = get_orders_id(orders)
        if not ids:
            return dict()
        self.solicited_cancels = set(ids).union(self.solicited_cancels)
        self._curl_bitmex(path="order", postdict={'clOrdID': ','.join(ids)}, verb="DELETE")

    @authentication_required
    def send_orders(self, orders: List[OrderCommon]):
        # DEV NOTE: it logs all sent orders, good or bad orders
        # DEV NOTE: it should NOT change any internal state

        # sanity check
        if not all([o.status == OrderStatus.Pending for o in orders]):
            raise ValueError("New orders should have status OrderStatus.Pending")

        for o in orders:
            if o.client_id:
                tokens = o.client_id.split('_')
                tactic_id = tokens[0]
            if not o.client_id or len(tokens) < 2 or tactic_id not in self.tactics_map:
                raise ValueError('The order client_id should be a string in the form TACTICID_HASH')

        converted = [self.convert_order_to_bitmex(self.check_order_sanity(order)) for order in orders]
        raws = self._curl_bitmex(path='order/bulk', postdict={'orders': converted}, verb='POST')

        for r in raws:
            symbol = Symbol[r['symbol']]
            type_ = OrderType[r['ordType']]
            tactic_id = r['clOrdID'].split('_')[0]
            tactic = self.tactics_map[tactic_id]
            self._log_order(OrderCommon(symbol, type_).update_from_bitmex(r))

    @authentication_required
    def get_opened_orders(self, symbol: Symbol, client_id_prefix: str) -> OrderContainerType:

        raws = self._curl_bitmex(path='order',
                                 query={
                                     'symbol': symbol.name,
                                     'filter': '{"open": true}'
                                 },
                                 verb='GET')

        def is_owner(client_id: str):
            return client_id.split('_')[0] == client_id_prefix

        return {r['clOrdID']: OrderCommon(symbol, OrderType[r['ordType']]).update_from_bitmex(r)
                for r in raws if r['leavesQty'] != 0 and is_owner(r['clOrdID'])}

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
                opened = [order.client_id for order in orders_cancelled if order.is_open()]
                raise TimeoutError(
                    "Orders cancelled through http (ids={}) were not confirmed in websocket within {} seconds".format(
                        ','.join(opened), self.timeout / 2))

    @staticmethod
    def check_order_sanity(order: OrderCommon) -> OrderCommon:
        assert order.client_id
        return order

    @staticmethod
    def convert_order_to_bitmex(order: OrderCommon) -> dict:
        a = {'leavesQty': order.leaves_qty}
        if order.client_id:
            a['clOrdID'] = order.client_id
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
                self._curl_bitmex(path="order/all", postdict={}, verb="DELETE")

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
    parser.add_argument('--test', action='store_true', help='run basic tests (WARNING: it maybe spend money!!!!!!!!!)')

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


def print_sleep():
    i = 0
    while True:
        print('sleeping ... {}'.format(i))
        time.sleep(1)
        i += 1


def test_common_1(tactic, sleep_time, input_args):
    args = get_args(input_args)

    t = threading.Thread(target=print_sleep)
    t.daemon = True

    live = LiveBitMex(args.log_dir, args.pref, sleep_time + 1)
    live.register_tactic(tactic)
    t.start()
    with live:
        live.ws.log_summary()

    return 0


def test_market_order(n_trades, n_positions, input_args=None):
    return test_common_1(TacticMarketOrderTest(n_trades=n_trades, n_positions=n_positions), 9, input_args)


def test_limit_order_and_cancels(input_args=None):
    return test_common_1(TacticLimitOrderTest(), 5, input_args)


def test2(input_args=None):
    tactic = TacticMarketOrderTest()
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
            tac.initialize(live, args.pref)

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

    with LiveBitMex(args.log_dir, 10) as live:
        for i in range(5):
            print('sleeping ... ' + str(i))
            time.sleep(1)
            print(live.get_last_trade(Symbol.XBTUSD))
            # print(live.get_last_trade(Symbol.XBTUSD).__dict__)

        live.ws.log_summary()

    return 0


def test_all(args):
    test_market_order(n_trades=2, n_positions=2)
    test_limit_order_and_cancels()

    logger.info("ALL TESTS PASSED !!!")
    return 0


def main(input_args=None):
    args = get_args(input_args)

    raise NotImplementedError()


if __name__ == "__main__":
    args = get_args()
    if args.test:
        sys.exit(test_all(args))
    else:
        sys.exit(main())
