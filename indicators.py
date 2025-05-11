import numpy as np
import pandas as pd
import sqlite3
import os
import logging
from datetime import datetime
import time

logger = logging.getLogger(__name__)


class TechnicalIndicators:
    def __init__(self, db_path='data'):
        self.db_path = db_path

        # Make sure directory exists
        os.makedirs(os.path.join(db_path, 'technical_analysis'), exist_ok=True)

        # Initialize database
        self.init_database()

        logger.info("Technical indicators module initialized")

    def init_database(self):
        """Initialize database for storing technical indicators"""
        conn = sqlite3.connect(os.path.join(self.db_path, 'technical_analysis.db'))
        cursor = conn.cursor()

        # Create indicators table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS indicators (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol TEXT NOT NULL,
            timeframe TEXT NOT NULL,
            timestamp INTEGER NOT NULL,
            rsi REAL,
            macd REAL,
            macd_signal REAL,
            macd_histogram REAL,
            ema_short REAL,
            ema_medium REAL,
            ema_long REAL,
            ema_crossover_short_medium INTEGER,
            ema_crossover_medium_long INTEGER,
            adx REAL,
            volume_change_24h REAL,
            price_change_24h REAL
        )
        ''')

        # Create index for faster queries
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_indicator_lookup ON indicators (symbol, timeframe, timestamp)')

        conn.commit()
        conn.close()

        logger.info("Technical analysis database initialized")

    def calculate_indicators(self, symbol, timeframe='1h', limit=100):
        """Calculate all technical indicators for a symbol and timeframe"""
        # Get kline data from market_data database
        conn_market = sqlite3.connect(os.path.join(self.db_path, 'market_data.db'))
        df = pd.read_sql_query('''
        SELECT open_time, open_price, high_price, low_price, close_price, volume, timestamp
        FROM kline_data
        WHERE symbol = ? AND timeframe = ?
        ORDER BY open_time ASC
        LIMIT ?
        ''', conn_market, params=(symbol, timeframe, limit + 100))  # Extra data for calculations

        conn_market.close()

        if len(df) < 50:  # Need minimum data for accurate indicators
            logger.warning(f"Not enough data for {symbol} {timeframe} indicators")
            return None

        # Calculate indicators
        df = self._calculate_rsi(df)
        df = self._calculate_macd(df)
        df = self._calculate_ema(df)
        df = self._calculate_adx(df)
        df = self._calculate_volume_change(df)

        # Store in database
        conn = sqlite3.connect(os.path.join(self.db_path, 'technical_analysis.db'))
        cursor = conn.cursor()

        # Only store the most recent data points (limit)
        df = df.tail(limit)

        for _, row in df.iterrows():
            if pd.isna(row['rsi']) or pd.isna(row['macd']):
                continue

            cursor.execute('''
            INSERT OR REPLACE INTO indicators
            (symbol, timeframe, timestamp, rsi, macd, macd_signal, macd_histogram,
            ema_short, ema_medium, ema_long, ema_crossover_short_medium,
            ema_crossover_medium_long, adx, volume_change_24h, price_change_24h)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                symbol,
                timeframe,
                int(row['timestamp']),
                float(row['rsi']),
                float(row['macd']),
                float(row['macd_signal']),
                float(row['macd_histogram']),
                float(row['ema_short']),
                float(row['ema_medium']),
                float(row['ema_long']),
                int(row['ema_crossover_short_medium']),
                int(row['ema_crossover_medium_long']),
                float(row['adx']),
                float(row['volume_change_24h']),
                float(row['price_change_24h'])
            ))

        conn.commit()
        conn.close()

        return df.tail(1).to_dict('records')[0]

    def _calculate_rsi(self, df, period=14):
        """Calculate Relative Strength Index"""
        delta = df['close_price'].diff()
        gain = delta.mask(delta < 0, 0)
        loss = -delta.mask(delta > 0, 0)

        avg_gain = gain.rolling(window=period).mean()
        avg_loss = loss.rolling(window=period).mean()

        # Calculate RS (Relative Strength)
        rs = avg_gain / avg_loss

        # Calculate RSI
        df['rsi'] = 100 - (100 / (1 + rs))

        return df

    def _calculate_macd(self, df, fast=12, slow=26, signal=9):
        """Calculate MACD (Moving Average Convergence Divergence)"""
        # Calculate EMAs
        ema_fast = df['close_price'].ewm(span=fast, adjust=False).mean()
        ema_slow = df['close_price'].ewm(span=slow, adjust=False).mean()

        # Calculate MACD line
        df['macd'] = ema_fast - ema_slow

        # Calculate signal line
        df['macd_signal'] = df['macd'].ewm(span=signal, adjust=False).mean()

        # Calculate MACD histogram
        df['macd_histogram'] = df['macd'] - df['macd_signal']

        return df

    def _calculate_ema(self, df, short=9, medium=21, long=50):
        """Calculate EMA crossovers"""
        # Calculate EMAs
        df['ema_short'] = df['close_price'].ewm(span=short, adjust=False).mean()
        df['ema_medium'] = df['close_price'].ewm(span=medium, adjust=False).mean()
        df['ema_long'] = df['close_price'].ewm(span=long, adjust=False).mean()

        # Calculate crossovers
        df['ema_crossover_short_medium'] = 0  # 0 = no crossover
        df['ema_crossover_medium_long'] = 0

        # Short-Medium crossover (1 = bullish, -1 = bearish)
        for i in range(1, len(df)):
            if (df['ema_short'].iloc[i - 1] < df['ema_medium'].iloc[i - 1] and
                    df['ema_short'].iloc[i] >= df['ema_medium'].iloc[i]):
                df.at[df.index[i], 'ema_crossover_short_medium'] = 1  # Bullish crossover
            elif (df['ema_short'].iloc[i - 1] > df['ema_medium'].iloc[i - 1] and
                  df['ema_short'].iloc[i] <= df['ema_medium'].iloc[i]):
                df.at[df.index[i], 'ema_crossover_short_medium'] = -1  # Bearish crossover

        # Medium-Long crossover
        for i in range(1, len(df)):
            if (df['ema_medium'].iloc[i - 1] < df['ema_long'].iloc[i - 1] and
                    df['ema_medium'].iloc[i] >= df['ema_long'].iloc[i]):
                df.at[df.index[i], 'ema_crossover_medium_long'] = 1  # Bullish crossover
            elif (df['ema_medium'].iloc[i - 1] > df['ema_long'].iloc[i - 1] and
                  df['ema_medium'].iloc[i] <= df['ema_long'].iloc[i]):
                df.at[df.index[i], 'ema_crossover_medium_long'] = -1  # Bearish crossover

        return df

    def _calculate_adx(self, df, period=14):
        """Calculate Average Directional Index"""
        # Calculate True Range
        df['tr1'] = abs(df['high_price'] - df['low_price'])
        df['tr2'] = abs(df['high_price'] - df['close_price'].shift())
        df['tr3'] = abs(df['low_price'] - df['close_price'].shift())
        df['tr'] = df[['tr1', 'tr2', 'tr3']].max(axis=1)

        # Calculate directional movement
        df['dm_plus'] = 0.0
        df['dm_minus'] = 0.0

        for i in range(1, len(df)):
            high_diff = df['high_price'].iloc[i] - df['high_price'].iloc[i - 1]
            low_diff = df['low_price'].iloc[i - 1] - df['low_price'].iloc[i]

            if high_diff > low_diff and high_diff > 0:
                df.at[df.index[i], 'dm_plus'] = high_diff
            else:
                df.at[df.index[i], 'dm_plus'] = 0

            if low_diff > high_diff and low_diff > 0:
                df.at[df.index[i], 'dm_minus'] = low_diff
            else:
                df.at[df.index[i], 'dm_minus'] = 0

        # Calculate smoothed values
        df['smoothed_tr'] = df['tr'].rolling(window=period).sum()
        df['smoothed_dm_plus'] = df['dm_plus'].rolling(window=period).sum()
        df['smoothed_dm_minus'] = df['dm_minus'].rolling(window=period).sum()

        # Calculate directional indicators
        df['di_plus'] = 100 * df['smoothed_dm_plus'] / df['smoothed_tr']
        df['di_minus'] = 100 * df['smoothed_dm_minus'] / df['smoothed_tr']

        # Calculate directional index
        df['dx'] = 100 * abs(df['di_plus'] - df['di_minus']) / (df['di_plus'] + df['di_minus'])

        # Calculate ADX
        df['adx'] = df['dx'].rolling(window=period).mean()

        return df

    def _calculate_volume_change(self, df, period=24):
        """Calculate volume and price change over period"""
        # Calculate 24h volume change
        df['volume_change_24h'] = df['volume'].pct_change(periods=period) * 100

        # Calculate 24h price change
        df['price_change_24h'] = df['close_price'].pct_change(periods=period) * 100

        return df

    def get_latest_indicators(self, symbol, timeframe='1h'):
        """Get latest technical indicators for a symbol and timeframe"""
        conn = sqlite3.connect(os.path.join(self.db_path, 'technical_analysis.db'))
        cursor = conn.cursor()

        cursor.execute('''
        SELECT rsi, macd, macd_signal, macd_histogram, ema_short, ema_medium, ema_long,
               ema_crossover_short_medium, ema_crossover_medium_long, adx,
               volume_change_24h, price_change_24h, timestamp
        FROM indicators
        WHERE symbol = ? AND timeframe = ?
        ORDER BY timestamp DESC
        LIMIT 1
        ''', (symbol, timeframe))

        row = cursor.fetchone()
        conn.close()

        if not row:
            return None

        return {
            'symbol': symbol,
            'timeframe': timeframe,
            'rsi': row[0],
            'macd': row[1],
            'macd_signal': row[2],
            'macd_histogram': row[3],
            'ema_short': row[4],
            'ema_medium': row[5],
            'ema_long': row[6],
            'ema_crossover_short_medium': row[7],
            'ema_crossover_medium_long': row[8],
            'adx': row[9],
            'volume_change_24h': row[10],
            'price_change_24h': row[11],
            'timestamp': row[12]
        }

    def generate_features_for_ai(self, symbol, timeframe='1h'):
        """Generate features for AI model based on technical indicators"""
        # Get latest indicators
        indicators = self.get_latest_indicators(symbol, timeframe)

        if not indicators:
            return None

        # Insert into AI model features database
        conn = sqlite3.connect(os.path.join(self.db_path, 'ai_model.db'))
        cursor = conn.cursor()

        # Create table if it doesn't exist
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS model_features (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol TEXT NOT NULL,
            price_change_1m REAL,
            price_change_5m REAL,
            price_change_15m REAL,
            price_change_1h REAL,
            volume_change_1m REAL,
            volume_change_5m REAL,
            rsi_value REAL,
            macd_histogram REAL,
            ema_crossover INTEGER, 
            timestamp INTEGER NOT NULL
        )
        ''')

        # Get price changes for multiple timeframes
        price_changes = self._get_price_changes(symbol)

        # Insert features
        cursor.execute('''
        INSERT INTO model_features
        (symbol, price_change_1m, price_change_5m, price_change_15m, price_change_1h,
         volume_change_1m, volume_change_5m, rsi_value, macd_histogram, ema_crossover, timestamp)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            symbol,
            price_changes.get('1m', 0),
            price_changes.get('5m', 0),
            price_changes.get('15m', 0),
            price_changes.get('1h', 0),
            price_changes.get('vol_1m', 0),
            price_changes.get('vol_5m', 0),
            indicators['rsi'],
            indicators['macd_histogram'],
            indicators['ema_crossover_medium_long'],  # Use medium-long term crossover
            indicators['timestamp']
        ))

        conn.commit()
        feature_id = cursor.lastrowid
        conn.close()

        return feature_id

    def _get_price_changes(self, symbol):
        """Get price changes for multiple timeframes"""
        conn = sqlite3.connect(os.path.join(self.db_path, 'market_data.db'))
        cursor = conn.cursor()

        timeframes = {
            '1m': 1 * 60 * 1000,  # 1 minute in milliseconds
            '5m': 5 * 60 * 1000,
            '15m': 15 * 60 * 1000,
            '1h': 60 * 60 * 1000
        }

        result = {}
        current_time = int(time.time() * 1000)

        # Get latest price
        cursor.execute('''
        SELECT price, volume, timestamp
        FROM market_data
        WHERE symbol = ?
        ORDER BY timestamp DESC
        LIMIT 1
        ''', (symbol,))

        latest = cursor.fetchone()
        if not latest:
            conn.close()
            return result

        latest_price, latest_volume, latest_timestamp = latest

        # Calculate price changes for each timeframe
        for tf_name, tf_ms in timeframes.items():
            # Get historical price
            cursor.execute('''
            SELECT price, volume
            FROM market_data
            WHERE symbol = ? AND timestamp < ?
            ORDER BY timestamp DESC
            LIMIT 1
            ''', (symbol, latest_timestamp - tf_ms))

            historical = cursor.fetchone()
            if historical:
                historical_price, historical_volume = historical
                price_change = ((latest_price - historical_price) / historical_price) * 100
                volume_change = ((
                                             latest_volume - historical_volume) / historical_volume) * 100 if historical_volume > 0 else 0

                result[tf_name] = price_change
                result[f'vol_{tf_name}'] = volume_change

        conn.close()
        return result