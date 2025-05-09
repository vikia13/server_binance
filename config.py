import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# WebSocket configuration
BINANCE_WEBSOCKET_URL = "wss://fstream.binance.com/ws/!ticker@arr"

# Alert parameters
TIME_INTERVAL_MINUTES = 3
PRICE_CHANGE_THRESHOLD = 3.0
MAX_SIGNALS_PER_DAY = 50

# Risk management parameters
RISK_PER_TRADE = 0.03  # 3% of account balance
DEFAULT_ACCOUNT_BALANCE = 1000  # Default account balance in USDT
DEFAULT_LEVERAGE = 10  # Default leverage
STOP_LOSS_PERCENTAGE = 0.05  # 5% stop loss
TAKE_PROFIT_PERCENTAGE = 0.1  # 10% take profit
MAX_DRAWDOWN_PERCENTAGE = 0.05  # 5% maximum drawdown
TRAILING_STOP_ACTIVATION = 0.03  # Activate trailing stop after 3% profit
TRAILING_STOP_DISTANCE = 0.02  # 2% trailing stop distance

# Coin filtering parameters
MIN_PRICE_THRESHOLD = 0.50  # Same as min_price in config.json
EXCLUDED_SYMBOLS = []  # No need to exclude as we filter by price

# Telegram configuration
TELEGRAM_TOKEN = os.environ.get("TELEGRAM_TOKEN")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID")

# Database configuration
DB_PATH = "data"  # Match the default db_path in main_enhanced.py

# Technical indicators parameters
RSI_PERIOD = 14
RSI_OVERBOUGHT = 70
RSI_OVERSOLD = 30
MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9
BOLLINGER_PERIOD = 20
BOLLINGER_STD = 2
EMA_50_PERIOD = 50
EMA_200_PERIOD = 200
ADX_PERIOD = 14
ADX_THRESHOLD = 25
STOCH_RSI_PERIOD = 14
STOCH_RSI_K = 3
STOCH_RSI_D = 3
STOCH_RSI_OVERBOUGHT = 80
STOCH_RSI_OVERSOLD = 20

# AI model parameters
AI_MODEL_PATH = "models/trend_detection_model.pkl"
AI_MEMORY_PATH = "data/ai_memory.json"