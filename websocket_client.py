import json
import logging
import threading
import time
import websocket
import sqlite3
import os
import queue
from datetime import datetime

logger = logging.getLogger(__name__)


class BinanceWebSocketClient:
    def __init__(self, db_path='data', min_price=0.50):
        self.db_path = db_path
        self.connections = {}
        self.futures_base_url = "wss://fstream.binance.com/ws/"
        self.spot_base_url = "wss://stream.binance.com:9443/ws/"
        self.combined_stream_url = "wss://fstream.binance.com/stream?streams="
        self.data_queue = queue.Queue(maxsize=10000)  # Buffer for incoming data
        self.processing_thread = None
        self.running = False
        self.symbols = set()
        self.min_price = min_price  # Minimum price filter
        self.excluded_symbols = set()  # Symbols to exclude (low liquidity)

        # Create database if it doesn't exist
        self.init_database()

        logger.info("Binance WebSocket client initialized")

    def init_database(self):
        """Initialize database tables for market data"""
        os.makedirs(self.db_path, exist_ok=True)

        conn = sqlite3.connect(os.path.join(self.db_path, 'market_data.db'))
        cursor = conn.cursor()

        # Create market data table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS market_data (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol TEXT NOT NULL,
            price REAL NOT NULL,
            volume REAL NOT NULL,
            price_change_24h REAL,
            volume_change_24h REAL,
            timestamp INTEGER NOT NULL
        )
        ''')

        # Create table for storing kline/candlestick data
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS kline_data (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol TEXT NOT NULL,
            timeframe TEXT NOT NULL,
            open_time INTEGER NOT NULL,
            close_time INTEGER NOT NULL,
            open_price REAL NOT NULL,
            high_price REAL NOT NULL,
            low_price REAL NOT NULL,
            close_price REAL NOT NULL,
            volume REAL NOT NULL,
            timestamp INTEGER NOT NULL
        )
        ''')

        # Create index for faster queries
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_market_symbol_time ON market_data (symbol, timestamp)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_kline_symbol_time ON kline_data (symbol, timeframe, open_time)')

        conn.commit()
        conn.close()

        logger.info("Database initialized for market data")

    def start(self):
        """Start all WebSocket connections and processing thread"""
        if self.running:
            return

        self.running = True

        # Start processing thread
        self.processing_thread = threading.Thread(target=self._process_queue)
        self.processing_thread.daemon = True
        self.processing_thread.start()

        # Step 1: Get all available USDT pairs first
        self._fetch_symbols()

        # Step 2: Start market tickers stream for all symbols
        self._connect_market_tickers()

        # Step 3: Start kline streams for major symbols
        timeframes = ["1m", "5m", "15m", "1h", "4h"]
        for symbol in self.symbols:
            if symbol in self.excluded_symbols:
                continue

            for timeframe in timeframes:
                self._connect_kline_stream(symbol.lower(), timeframe)

        logger.info(f"Started WebSocket connections for {len(self.symbols)} symbols")

    def stop(self):
        """Stop all WebSocket connections"""
        self.running = False

        for stream_name, ws in self.connections.items():
            try:
                ws.close()
                logger.info(f"Closed WebSocket: {stream_name}")
            except Exception as e:
                logger.error(f"Error closing WebSocket {stream_name}: {e}")

        self.connections = {}
        self.symbols = set()

        if self.processing_thread and self.processing_thread.is_alive():
            self.processing_thread.join(timeout=5)

        logger.info("Stopped all WebSocket connections")

    def _fetch_symbols(self):
        """Fetch all available USDT futures pairs from Binance"""
        import requests

        try:
            # Fetch futures symbols
            response = requests.get('https://fapi.binance.com/fapi/v1/exchangeInfo')
            data = response.json()

            for symbol_info in data.get('symbols', []):
                symbol = symbol_info.get('symbol')
                status = symbol_info.get('status')

                # Filter for active USDT pairs with price above minimum
                if status == 'TRADING' and symbol.endswith('USDT'):
                    # Get current price
                    try:
                        price_response = requests.get(f'https://fapi.binance.com/fapi/v1/ticker/price?symbol={symbol}')
                        price_data = price_response.json()
                        price = float(price_data.get('price', 0))

                        if price >= self.min_price:
                            self.symbols.add(symbol)
                        else:
                            self.excluded_symbols.add(symbol)
                            logger.info(f"Excluded {symbol} due to price below {self.min_price} USDT")
                    except Exception as e:
                        logger.error(f"Error fetching price for {symbol}: {e}")

            logger.info(f"Fetched {len(self.symbols)} active USDT trading pairs")

        except Exception as e:
            logger.error(f"Error fetching symbols: {e}")

    def _connect_market_tickers(self):
        """Connect to the all market tickers stream"""
        stream_name = "!ticker@arr"

        def on_message(ws, message):
            try:
                data = json.loads(message)
                # For !ticker@arr, data is an array of ticker objects
                if isinstance(data, list):
                    for ticker in data:
                        symbol = ticker.get('s')

                        # Filter only our tracked symbols
                        if symbol in self.symbols:
                            price = float(ticker.get('c', 0))  # Last price
                            volume = float(ticker.get('q', 0))  # 24h volume
                            price_change = float(ticker.get('p', 0))  # 24h price change
                            price_change_pct = float(ticker.get('P', 0))  # 24h price change percent

                            # Put in queue for processing
                            self.data_queue.put({
                                'type': 'ticker',
                                'symbol': symbol,
                                'price': price,
                                'volume': volume,
                                'price_change_24h': price_change,
                                'price_change_pct_24h': price_change_pct,
                                'timestamp': int(time.time() * 1000)
                            })
            except Exception as e:
                logger.error(f"Error processing ticker message: {e}")

        def on_error(ws, error):
            logger.error(f"WebSocket error ({stream_name}): {error}")

        def on_close(ws, close_status_code, close_msg):
            logger.info(f"WebSocket closed ({stream_name}): {close_msg}")
            if self.running:
                # Reconnect after delay
                time.sleep(5)
                self._connect_market_tickers()

        def on_open(ws):
            logger.info(f"WebSocket connected: {stream_name}")

        # Create connection
        ws = websocket.WebSocketApp(
            f"{self.futures_base_url}{stream_name}",
            on_message=on_message,
            on_error=on_error,
            on_close=on_close,
            on_open=on_open
        )

        # Start in a separate thread
        wst = threading.Thread(target=ws.run_forever)
        wst.daemon = True
        wst.start()

        # Store connection
        self.connections[stream_name] = ws

    def _connect_kline_stream(self, symbol, timeframe):
        """Connect to kline/candlestick stream for specific symbol and timeframe"""
        stream_name = f"{symbol}@kline_{timeframe}"

        def on_message(ws, message):
            try:
                data = json.loads(message)

                if 'k' in data:
                    kline = data['k']

                    symbol = data.get('s')
                    is_closed = kline.get('x', False)  # Whether the candle is closed

                    # Only process completed candles
                    if is_closed:
                        timeframe = kline.get('i')
                        open_time = kline.get('t')
                        close_time = kline.get('T')
                        open_price = float(kline.get('o', 0))
                        high_price = float(kline.get('h', 0))
                        low_price = float(kline.get('l', 0))
                        close_price = float(kline.get('c', 0))
                        volume = float(kline.get('v', 0))

                        # Put in queue for processing
                        self.data_queue.put({
                            'type': 'kline',
                            'symbol': symbol,
                            'timeframe': timeframe,
                            'open_time': open_time,
                            'close_time': close_time,
                            'open_price': open_price,
                            'high_price': high_price,
                            'low_price': low_price,
                            'close_price': close_price,
                            'volume': volume,
                            'timestamp': int(time.time() * 1000)
                        })
            except Exception as e:
                logger.error(f"Error processing kline message: {e}")

        def on_error(ws, error):
            logger.error(f"WebSocket error ({stream_name}): {error}")

        def on_close(ws, close_status_code, close_msg):
            logger.info(f"WebSocket closed ({stream_name}): {close_msg}")
            if self.running:
                # Reconnect after delay
                time.sleep(5)
                self._connect_kline_stream(symbol, timeframe)

        def on_open(ws):
            logger.info(f"WebSocket connected: {stream_name}")

        # Create connection
        ws = websocket.WebSocketApp(
            f"{self.futures_base_url}{stream_name}",
            on_message=on_message,
            on_error=on_error,
            on_close=on_close,
            on_open=on_open
        )

        # Start in a separate thread
        wst = threading.Thread(target=ws.run_forever)
        wst.daemon = True
        wst.start()

        # Store connection
        self.connections[stream_name] = ws

    def _process_queue(self):
        """Process data from queue and store in database"""
        while self.running or not self.data_queue.empty():
            try:
                # Get data from queue with timeout
                data = self.data_queue.get(timeout=1)

                if data['type'] == 'ticker':
                    self._store_ticker_data(data)
                elif data['type'] == 'kline':
                    self._store_kline_data(data)

                # Mark task as done
                self.data_queue.task_done()
            except queue.Empty:
                # Queue timeout, continue loop
                continue
            except Exception as e:
                logger.error(f"Error processing data: {e}")

                # Still mark as done to avoid blocking
                try:
                    self.data_queue.task_done()
                except:
                    pass

    def _store_ticker_data(self, data):
        """Store ticker data in database"""
        try:
            conn = sqlite3.connect(os.path.join(self.db_path, 'market_data.db'))
            cursor = conn.cursor()

            cursor.execute('''
            INSERT INTO market_data
            (symbol, price, volume, price_change_24h, timestamp)
            VALUES (?, ?, ?, ?, ?)
            ''', (
                data['symbol'],
                data['price'],
                data['volume'],
                data['price_change_24h'],
                data['timestamp']
            ))

            conn.commit()
            conn.close()
        except Exception as e:
            logger.error(f"Error storing ticker data: {e}")

    def _store_kline_data(self, data):
        """Store kline/candlestick data in database"""
        try:
            conn = sqlite3.connect(os.path.join(self.db_path, 'market_data.db'))
            cursor = conn.cursor()

            cursor.execute('''
            INSERT INTO kline_data
            (symbol, timeframe, open_time, close_time, open_price, high_price, 
             low_price, close_price, volume, timestamp)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                data['symbol'],
                data['timeframe'],
                data['open_time'],
                data['close_time'],
                data['open_price'],
                data['high_price'],
                data['low_price'],
                data['close_price'],
                data['volume'],
                data['timestamp']
            ))

            conn.commit()
            conn.close()
        except Exception as e:
            logger.error(f"Error storing kline data: {e}")

    def get_recent_price(self, symbol, minutes=5):
        """Get recent price data for a symbol"""
        try:
            conn = sqlite3.connect(os.path.join(self.db_path, 'market_data.db'))
            cursor = conn.cursor()

            # Calculate timestamp threshold
            threshold = int(time.time() * 1000) - (minutes * 60 * 1000)

            cursor.execute('''
            SELECT price, volume, timestamp
            FROM market_data
            WHERE symbol = ? AND timestamp > ?
            ORDER BY timestamp DESC
            LIMIT 100
            ''', (symbol, threshold))

            rows = cursor.fetchall()
            conn.close()

            return rows
        except Exception as e:
            logger.error(f"Error retrieving recent prices: {e}")
            return []

    def get_kline_data(self, symbol, timeframe, limit=100):
        """Get historical kline data for a symbol and timeframe"""
        try:
            conn = sqlite3.connect(os.path.join(self.db_path, 'market_data.db'))
            cursor = conn.cursor()

            cursor.execute('''
            SELECT open_time, open_price, high_price, low_price, close_price, volume
            FROM kline_data
            WHERE symbol = ? AND timeframe = ?
            ORDER BY open_time DESC
            LIMIT ?
            ''', (symbol, timeframe, limit))

            rows = cursor.fetchall()
            conn.close()

            return rows
        except Exception as e:
            logger.error(f"Error retrieving kline data: {e}")
            return []