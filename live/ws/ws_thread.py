import sys
import websocket
import threading
import traceback
import ssl
from time import sleep
import json
import decimal
import logging

from typing import Union

from common.quote import Quote
from live.settings import settings
from live.auth.APIKeyAuth import generate_nonce, generate_signature
from utils.log import setup_custom_logger
from utils.utils import to_nearest
from future.utils import iteritems
from future.standard_library import hooks

with hooks():  # Python 2/3 compat
    from urllib.parse import urlparse, urlunparse


# Connects to BitMEX websocket for streaming realtime data.
# The Marketmaker still interacts with this as if it were a REST Endpoint, but now it can get
# much more realtime data without heavily polling the API.
#
# The Websocket offers a bunch of data as raw properties right on the object.
# On connect, it synchronously asks for a push of all this data then returns.
# Right after, the MM can start using its data. It will be updated in realtime, so the MM can
# poll as often as it wants.
class BitMEXWebsocket():
    # Don't grow a table larger than this amount. Helps cap memory usage.
    MAX_TABLE_LEN = 500

    def __init__(self):
        self.logger = logging.getLogger('root')
        self.__reset()
        self.symbol = None
        self.should_auth = True

        self.callback_maps = None  # type: dict

    def __del__(self):
        self.exit()

    def connect(self,
                endpoint="",
                symbol="XBTUSD",
                should_auth=True,
                callback_maps=None):
        """
        :param endpoint:
        :param symbol:
        :param should_auth:
        :param callback_maps: 'bitmex subscription' -> callback .. if subs is present, this class won't store its data
        :return:
        """

        if callback_maps is None:
            self.callback_maps = {}
        else:
            self.callback_maps = callback_maps

        self.logger.debug("Connecting WebSocket.")
        self.symbol = symbol
        self.should_auth = should_auth

        # We can subscribe right in the connection querystring, so let's build that.
        # Subscribe to all pertinent endpoints
        subscriptions = [sub + ':' + symbol for sub in ["quote", "trade"]]
        subscriptions += ['tradeBin1m:' + symbol]
        # subscriptions += ["instrument"]  # We want all of them
        if self.should_auth:
            subscriptions += [sub + ':' + symbol for sub in ["order", "execution"]]
            subscriptions += ["margin", "position"]

        # Get WS URL and connect.
        url_parts = list(urlparse(endpoint))
        url_parts[0] = url_parts[0].replace('http', 'ws')
        url_parts[2] = "/realtime?subscribe=" + ",".join(subscriptions)
        ws_url = urlunparse(url_parts)
        self.logger.info("Connecting to %s" % ws_url)
        self.__connect(ws_url)
        self.logger.info('Connected to WS. Waiting for data images, this may take a moment...')

        # Connected. Wait for partials
        self.__wait_for_symbol(symbol)
        if self.should_auth:
            self.__wait_for_account()
        self.logger.info('Got all market data. Starting.')

    #
    # Data methods
    #
    def get_instrument(self, symbol):
        instruments = self.data['instrument']
        matching_instruments = [i for i in instruments if i['symbol'] == symbol]
        if len(matching_instruments) == 0:
            raise Exception("Unable to find instrument or index with symbol: " + symbol)
        instrument = matching_instruments[0]
        # Turn the 'tickSize' into 'tickLog' for use in rounding
        # http://stackoverflow.com/a/6190291/832202
        instrument['tickLog'] = decimal.Decimal(str(instrument['tickSize'])).as_tuple().exponent * -1
        return instrument

    def get_ticker(self, symbol):
        """Return a ticker object. Generated from instrument."""

        instrument = self.get_instrument(symbol)

        # If this is an index, we have to get the data from the last trade.
        if instrument['symbol'][0] == '.':
            ticker = {'mid': instrument['markPrice'], 'buy': instrument['markPrice'], 'sell': instrument['markPrice'],
                      'last': instrument['markPrice']}
        # Normal instrument
        else:
            bid = instrument['bidPrice'] or instrument['lastPrice']
            ask = instrument['askPrice'] or instrument['lastPrice']
            ticker = {
                "last": instrument['lastPrice'],
                "buy": bid,
                "sell": ask,
                "mid": (bid + ask) * 0.5
            }

        # The instrument has a tickSize. Use it to round values.
        return {k: to_nearest(float(v or 0), instrument['tickSize']) for k, v in iteritems(ticker)}

    def funds(self):
        return self.data['margin'][0]

    def market_depth(self, symbol):
        raise NotImplementedError('orderBook is not subscribed; use askPrice and bidPrice on instrument')
        # return self.data['orderBook25'][0]

    def open_orders(self, cl_ord_id_prefix):
        orders = self.data['order']
        # Filter to only open orders (leavesQty > 0) and those that we actually placed
        return [o for o in orders if str(o['clOrdID']).startswith(cl_ord_id_prefix) and o['leavesQty'] > 0]

    def recent_trades(self):
        return self.data['trade']

    def trades1min_bin(self):
        return self.data['tradeBin1m']

    #
    # Lifecycle methods
    #
    def error(self, err):
        self._error = err
        self.logger.error(err)
        self.exit()

    def exit(self):
        self.exited = True
        self.ws.close()

    def log_summary(self):
        self.logger.info('Stored data:')
        for table, data in self.data.items():
            self.logger.info(' table: {}, len(data)={}'.format(table, len(data)))

    #
    # Private methods
    #

    def __connect(self, wsURL):
        """Connect to the websocket in a thread."""
        self.logger.debug("Starting thread")

        ssl_defaults = ssl.get_default_verify_paths()
        sslopt_ca_certs = {'ca_certs': ssl_defaults.cafile}
        self.ws = websocket.WebSocketApp(wsURL,
                                         on_message=self.__on_message,
                                         on_close=self.__on_close,
                                         on_open=self.__on_open,
                                         on_error=self.__on_error,
                                         header=self.__get_auth()
                                         )

        setup_custom_logger('websocket', log_level=logging.INFO)
        self.wst = threading.Thread(target=lambda: self.ws.run_forever(sslopt=sslopt_ca_certs))
        self.wst.daemon = True
        self.wst.start()
        self.logger.info("Started thread")

        # Wait for connect before continuing
        conn_timeout = 10
        while (not self.ws.sock or not self.ws.sock.connected) and conn_timeout and not self._error:
            sleep(1)
            conn_timeout -= 1

        if not conn_timeout or self._error:
            self.logger.error(
                "Couldn't connect to WS! Exiting. (timeout, error) = ({}, {})".format(conn_timeout == 0, self._error))
            self.exit()
            sys.exit(1)

    def __get_auth(self):
        """Return auth headers. Will use API Keys if present in settings."""

        if self.should_auth is False:
            return []

        self.logger.info("Authenticating with API Key.")
        # To auth to the WS using an API key, we generate a signature of a nonce and
        # the WS API endpoint.
        nonce = generate_nonce()
        return [
            "api-nonce: " + str(nonce),
            "api-signature: " + generate_signature(settings.API_SECRET, 'GET', '/realtime', nonce, ''),
            "api-key:" + settings.API_KEY
        ]

    def __wait_for_account(self):
        """On subscribe, this data will come down. Wait for it."""
        # Wait for the keys to show up from the ws
        while not {'margin', 'position', 'order'} <= set(self.data):
            sleep(0.1)

    def __wait_for_symbol(self, symbol):
        """On subscribe, this data will come down. Wait for it."""
        while not {'tradeBin1m', 'trade', 'quote'} <= set(self.data):
            sleep(0.1)

    # def __send_command(self, command, args):
    #    '''Send a raw command.'''
    #    self.ws.send(json.dumps({"op": command, "args": args or []}))

    def __on_message(self, ws, message):
        """Handler for parsing WS messages."""
        message = json.loads(message)
        self.logger.debug(json.dumps(message))

        table = message['table'] if 'table' in message else None
        action = message['action'] if 'action' in message else None
        try:
            if 'subscribe' in message:
                if message['success']:
                    self.logger.debug("Subscribed to %s." % message['subscribe'])
                else:
                    self.error("Unable to subscribe to %s. Error: \"%s\" Please check and restart." %
                               (message['request']['args'][0], message['error']))
            elif 'status' in message:
                if message['status'] == 400:
                    self.error(message['error'])
                if message['status'] == 401:
                    self.error("API Key incorrect, please check and restart.")
            elif action:

                if table not in self.data:
                    self.data[table] = []

                if table not in self.keys:
                    self.keys[table] = []

                # There are four possible actions from the WS:
                # 'partial' - full table image
                # 'insert'  - new row
                # 'update'  - update row
                # 'delete'  - delete row
                if action == 'partial':
                    self.logger.debug("%s: partial" % table)
                    self.data[table] += message['data']
                    # Keys are communicated on partials to let you know how to uniquely identify
                    # an item. We use it for updates.
                    self.keys[table] = message['keys']

                    callback = self.callback_maps.get(table)
                    if callback:
                        for data in message['data']:
                            callback(data)

                elif action == 'insert':
                    self.logger.debug('%s: inserting %s' % (table, message['data']))
                    self.data[table] += message['data']

                    callback = self.callback_maps.get(table)
                    if callback:
                        for data in message['data']:
                            callback(data)

                    # Limit the max length of the table to avoid excessive memory usage.
                    # Don't trim orders because we'll lose valuable state if we do.
                    if table not in ['order', 'orderBookL2'] and len(self.data[table]) > BitMEXWebsocket.MAX_TABLE_LEN:
                        self.data[table] = self.data[table][(BitMEXWebsocket.MAX_TABLE_LEN // 2):]

                elif action == 'update':
                    self.logger.debug('%s: updating %s' % (table, message['data']))
                    # Locate the item in the collection and update it.
                    for updateData in message['data']:
                        item = find_item_by_key(self.keys[table], self.data[table], updateData)
                        if not item:
                            continue  # No item found to update. Could happen before push

                        # # Log executions
                        # if table == 'order':
                        #     is_canceled = 'ordStatus' in updateData and updateData['ordStatus'] == 'Canceled'
                        #     if 'cumQty' in updateData and not is_canceled:
                        #         contExecuted = updateData['cumQty'] - item['cumQty']
                        #         if contExecuted > 0:
                        #             instrument = self.get_instrument(item['symbol'])
                        #             self.logger.info("Execution: %s %d Contracts of %s at %.*f" %
                        #                              (item['side'], contExecuted, item['symbol'],
                        #                               instrument['tickLog'], item['price']))

                        # Update this item.
                        item.update(updateData)

                        callback = self.callback_maps.get(table)
                        if callback:
                            callback(item)

                        # Remove canceled / filled orders
                        if (table == 'order' or table == 'execution') and item['leavesQty'] <= 0:
                            self.data[table].remove(item)

                elif action == 'delete':
                    raise NotImplementedError("I was not expecting getting 'delete' for table={}".format(table))
                    # TODO: please don't delete the comments below, it may be useful later
                    # self.logger.info('%s: deleting %s' % (table, message['data']))
                    # # Locate the item in the collection and remove it.
                    # for deleteData in message['data']:
                    #     item = find_item_by_key(self.keys[table], self.data[table], deleteData)
                    #     self.data[table].remove(item)
                else:
                    raise Exception("Unknown action: %s" % action)

        except:
            self.logger.error(traceback.format_exc())

    def __on_open(self, ws):
        self.logger.debug("Websocket Opened.")

    def __on_close(self, ws):
        self.logger.info('Websocket Closed')
        self.exit()

    def __on_error(self, ws, error):
        if not self.exited:
            self.error(error)

    def __reset(self):
        self.data = {}
        self.keys = {}
        self.exited = False
        self._error = None


def find_item_by_key(keys, table, match_data):
    for item in table:
        matched = True
        for key in keys:
            if item[key] != match_data[key]:
                matched = False
        if matched:
            return item


if __name__ == "__main__":
    # create console handler and set level to debug
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    # create formatter
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    # add formatter to ch
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    ws = BitMEXWebsocket()
    ws.logger = logger
    ws.connect("https://testnet.bitmex.com/api/v1")
    while ws.ws.sock.connected:
        sleep(1)
