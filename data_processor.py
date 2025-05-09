import pandas as pd
import numpy as np
import logging
import datetime
import warnings
from ta.trend import ADXIndicator
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.volatility import BollingerBands
from ta.trend import MACD, SMAIndicator, EMAIndicator
from ta.volume import VolumeWeightedAveragePrice

# Configure logging
logger = logging.getLogger(__name__)

# Constants
LOOKBACK_PERIOD = 100  # Number of data points to keep
RSI_PERIOD = 14
MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9
EMA_SHORT = 50
EMA_LONG = 200
ADX_PERIOD = 14
ADX_THRESHOLD = 25  # Strong trend if ADX > 25
PRICE_CHANGE_THRESHOLD = 1.2  # Significant price change percentage
VOLUME_INCREASE_THRESHOLD = 20  # Volume increase percentage

# Suppress specific TA library warnings
warnings.filterwarnings("ignore", category=RuntimeWarning,
                        message="invalid value encountered in scalar divide")


class DataProcessor:
    def __init__(self):
        self.symbol_data = {}  # Dictionary to store data for each symbol
        self.last_processed = {}  # Track last processed timestamp for each symbol
        self.logger = logging.getLogger('data_processor')
        self.logger.info("Data processor initialized")

    def update_data(self, ticker_data):
        """
        Update the data for a symbol with the latest ticker data
        """
        symbol = ticker_data['symbol']
        if symbol not in self.symbol_data:
            self.symbol_data[symbol] = []

        # Add new data
        self.symbol_data[symbol].append({
            'timestamp': ticker_data['timestamp'],
            'price': ticker_data['price'],
            'volume': ticker_data['volume']
        })

        # Limit data size to lookback period
        if len(self.symbol_data[symbol]) > LOOKBACK_PERIOD:
            self.symbol_data[symbol] = self.symbol_data[symbol][-LOOKBACK_PERIOD:]

    def calculate_indicators(self, symbol):
        """
        Calculate technical indicators for a given symbol
        """
        if symbol not in self.symbol_data or len(self.symbol_data[symbol]) < 30:
            return None

        # Convert to DataFrame
        df = pd.DataFrame(self.symbol_data[symbol])

        # Calculate price changes
        df['price_pct_change'] = df['price'].pct_change() * 100
        df['volume_pct_change'] = df['volume'].pct_change() * 100

        # Calculate VWAP
        vwap = VolumeWeightedAveragePrice(
            high=df['price'],
            low=df['price'],
            close=df['price'],
            volume=df['volume']
        )
        df['vwap'] = vwap.volume_weighted_average_price()

        # Calculate RSI
        rsi = RSIIndicator(df['price'], window=RSI_PERIOD)
        df['rsi'] = rsi.rsi()

        # Calculate MACD
        macd = MACD(
            df['price'],
            window_fast=MACD_FAST,
            window_slow=MACD_SLOW,
            window_sign=MACD_SIGNAL
        )
        df['macd'] = macd.macd()
        df['macd_signal'] = macd.macd_signal()
        df['macd_diff'] = macd.macd_diff()

        # Calculate Bollinger Bands
        bollinger = BollingerBands(df['price'])
        df['bb_upper'] = bollinger.bollinger_hband()
        df['bb_lower'] = bollinger.bollinger_lband()
        df['bb_middle'] = bollinger.bollinger_mavg()
        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']

        # Calculate Stochastic Oscillator
        stoch = StochasticOscillator(
            high=df['price'] * 1.001,  # Add small padding to avoid identical values
            low=df['price'] * 0.999,
            close=df['price']
        )
        df['stoch_k'] = stoch.stoch()
        df['stoch_d'] = stoch.stoch_signal()

        # Calculate EMAs
        ema_short = EMAIndicator(df['price'], window=EMA_SHORT)
        ema_long = EMAIndicator(df['price'], window=EMA_LONG)
        df['ema_short'] = ema_short.ema_indicator()
        df['ema_long'] = ema_long.ema_indicator()

        # Calculate EMA crossover signal
        # 1: bullish crossover (short crosses above long)
        # -1: bearish crossover (short crosses below long)
        # 0: no recent crossover
        df['ema_crossover'] = 0
        df.loc[df['ema_short'] > df['ema_long'], 'ema_crossover'] = 1
        df.loc[df['ema_short'] < df['ema_long'], 'ema_crossover'] = -1

        # Calculate ADX with safety adjustments to prevent division by zero
        df = self.calculate_adx(df, ADX_PERIOD)

        # Clean NaN values
        df = df.fillna(0)

        return df

    def calculate_adx(self, df, window=14):
        """Calculate ADX indicator with safety adjustments to prevent division by zero"""
        try:
            # Add small variance to high/low to avoid identical values
            adx = ADXIndicator(
                high=df['price'] * 1.0001,  # Add small padding to high price
                low=df['price'] * 0.9999,   # Subtract small padding from low price
                close=df['price'],
                window=window
            )

            # Calculate ADX components with NaN handling
            df['adx'] = adx.adx().fillna(0)
            df['+di'] = adx.adx_pos().fillna(0)
            df['-di'] = adx.adx_neg().fillna(0)

            # Clean any remaining NaN/inf values
            df.replace([np.inf, -np.inf], 0, inplace=True)
            df.fillna(0, inplace=True)

            return df
        except Exception as e:
            self.logger.error(f"Error calculating ADX: {e}")
            return df

    def detect_trend(self, symbol):
        """
        Enhanced trend detection with improved criteria
        """
        if symbol not in self.symbol_data or len(self.symbol_data[symbol]) < 30:
            return None

        # Check if we already processed data recently (within 1 minute)
        current_time = int(datetime.datetime.now().timestamp() * 1000)
        if symbol in self.last_processed and current_time - self.last_processed[symbol] < 60000:
            return None  # Skip processing if we did it recently

        # Calculate indicators
        df = self.calculate_indicators(symbol)
        if df is None:
            return None

        # Get the latest values
        latest = df.iloc[-1]

        # Detect pump (long) signal with enhanced criteria
        pump_signal = False
        if (
                latest['price_pct_change'] > PRICE_CHANGE_THRESHOLD and
                latest['rsi'] > 50 and
                latest['macd_diff'] > 0 and
                latest['price'] > latest['vwap'] and
                latest['ema_crossover'] >= 0 and  # EMA50 >= EMA200
                latest['adx'] > ADX_THRESHOLD  # Strong trend
        ):
            pump_signal = True

        # Detect dump (short) signal with enhanced criteria
        dump_signal = False
        if (
                latest['price_pct_change'] < -PRICE_CHANGE_THRESHOLD and
                latest['rsi'] < 50 and
                latest['macd_diff'] < 0 and
                latest['price'] < latest['vwap'] and
                latest['ema_crossover'] <= 0 and  # EMA50 <= EMA200
                latest['adx'] > ADX_THRESHOLD  # Strong trend
        ):
            dump_signal = True

        # Update last processed time
        self.last_processed[symbol] = current_time

        if pump_signal:
            return {
                'symbol': symbol,
                'trend': 'LONG',
                'price': latest['price'],
                'price_change': latest['price_pct_change'],
                'rsi': latest['rsi'],
                'macd_diff': latest['macd_diff'],
                'adx': latest['adx'],
                'ema_crossover': latest['ema_crossover'],
                'stoch_k': latest['stoch_k'],
                'timestamp': current_time
            }
        elif dump_signal:
            return {
                'symbol': symbol,
                'trend': 'SHORT',
                'price': latest['price'],
                'price_change': latest['price_pct_change'],
                'rsi': latest['rsi'],
                'macd_diff': latest['macd_diff'],
                'adx': latest['adx'],
                'ema_crossover': latest['ema_crossover'],
                'stoch_k': latest['stoch_k'],
                'timestamp': current_time
            }

        return None

    def detect_exit_signal(self, position):
        """
        Enhanced exit signal detection with improved criteria
        """
        symbol = position['symbol']
        trend = position['trend']
        entry_price = position['entry_price']

        # Add this block to prevent immediate exits
        if 'timestamp' in position:
            entry_time = position['timestamp']
            current_time = int(datetime.datetime.now().timestamp() * 1000)
            # Require minimum 15 minutes holding time
            if current_time - entry_time < 15 * 60 * 1000:
                return None

        df = self.calculate_indicators(symbol)
        if df is None or len(df) < 5:  # Need at least a few data points
            return None

        # Get the latest data
        latest = df.iloc[-1]
        current_price = latest['price']

        # Calculate profit/loss percentage
        if trend == 'LONG':
            profit_pct = ((current_price - entry_price) / entry_price) * 100

            # Exit conditions for LONG position with enhanced criteria
            if (
                    # Strong reversal signal
                    (latest['macd_diff'] < -0.0002 and latest['rsi'] < 40) or
                    # Overbought condition
                    (latest['rsi'] > 75 and latest['stoch_k'] > 80) or
                    # Price drops below VWAP significantly
                    (latest['price'] < latest['vwap'] * 0.99) or
                    # EMA crossover turns bearish
                    (latest['ema_crossover'] == -1 and latest['ema_crossover'].shift(1).iloc[0] == 1)
            ):
                return {
                    'symbol': symbol,
                    'exit_price': current_price,
                    'profit_pct': profit_pct,
                    'reason': 'Trend reversal detected',
                    'timestamp': int(datetime.datetime.now().timestamp() * 1000)
                }

        elif trend == 'SHORT':
            profit_pct = ((entry_price - current_price) / entry_price) * 100

            # Exit conditions for SHORT position with enhanced criteria
            if (
                    # Strong reversal signal
                    (latest['macd_diff'] > 0.0002 and latest['rsi'] > 60) or
                    # Oversold condition
                    (latest['rsi'] < 25 and latest['stoch_k'] < 20) or
                    # Price rises above VWAP significantly
                    (latest['price'] > latest['vwap'] * 1.01) or
                    # EMA crossover turns bullish
                    (latest['ema_crossover'] == 1 and latest['ema_crossover'].shift(1).iloc[0] == -1)
            ):
                return {
                    'symbol': symbol,
                    'exit_price': current_price,
                    'profit_pct': profit_pct,
                    'reason': 'Trend reversal detected',
                    'timestamp': int(datetime.datetime.now().timestamp() * 1000)
                }

        return None

    def get_market_data(self, symbol, period=100):
        """
        Extract the most recent market data for AI model training
        """
        if symbol not in self.symbol_data or len(self.symbol_data[symbol]) < period:
            return None

        df = self.calculate_indicators(symbol)
        if df is None:
            return None

        # Get the most recent data points
        return df.tail(period)