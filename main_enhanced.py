import os
import time
import logging
import threading
import json
import argparse
from datetime import datetime
import sys
import sqlite3
from web_dashboard import run_dashboard
from ai_model_enhanced import EnhancedAIModel
from websocket_client import BinanceWebSocketClient
from technical_analysis.indicators import TechnicalIndicators
from signal_generator import SignalGenerator

# Import Telegram components
try:
    from telegram_bot import TradingTelegramBot  # Use the existing class name
    from telegram_adapter import EnhancedTelegramBot
    TELEGRAM_AVAILABLE = True
except ImportError as e:
    TELEGRAM_AVAILABLE = False
    print(f"Telegram functionality not available: {e}")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('app.log')
    ]
)

logger = logging.getLogger(__name__)


class AITradingSystem:
    def __init__(self, db_path='data', config_path='config.json'):
        self.db_path = db_path
        self.config_path = config_path
        self.running = False
        self.timeframes = ["1m", "5m", "15m", "1h", "4h"]
        self.config = self.load_config()
        self.signal_check_interval = 300  # Check for signals every 5 minutes
        self.indicator_update_interval = 900  # Update indicators every 15 minutes
        self.model_retrain_interval = 86400  # Retrain models every 24 hours        # Initialize Telegram bot
        self.telegram_bot = self.init_telegram()
        # Initialize components
        self.init_components()
        # Create database directory if it doesn't exist
        os.makedirs(db_path, exist_ok=True)

    def init_telegram(self):
        """Initialize Telegram bot if available"""
        try:
            # Get token from config file instead of environment variables
            telegram_config = self.config.get("telegram", {})
            telegram_token = telegram_config.get("token") or os.environ.get("TELEGRAM_TOKEN")
            telegram_chat_id = telegram_config.get("chat_id") or os.environ.get("TELEGRAM_CHAT_ID", "")
            if not telegram_token:
                logger.warning("Telegram token not set in config or environment variables")
                return None
            # Create enhanced bot with proper wrapper
            from telegram_adapter import EnhancedTelegramBot
            bot = EnhancedTelegramBot(telegram_token, telegram_chat_id)
            logger.info(f"Telegram bot initialized with chat ID: {telegram_chat_id}")
            return bot
        except Exception as e:
            logger.error(f"Failed to initialize Telegram bot: {e}", exc_info=True)
            return None


    def load_config(self):
        """Load configuration from config file"""
        if not os.path.exists(self.config_path):
            # Create default config without symbols list
            default_config = {
                "min_confidence": 0.2,
                "min_price": 0.2,
                "max_active_signals": 9,
                "signal_throttle_hours": 24
            }
            with open(self.config_path, "w") as f:
                json.dump(default_config, f, indent=4)
            logger.info(f"Created default configuration at {self.config_path}")
            return default_config
        try:
            with open(self.config_path, "r") as f:
                config = json.load(f)
            logger.info(f"Loaded configuration from {self.config_path}")
            return config
        except Exception as e:
            logger.error(f"Error loading configuration: {e}")
            sys.exit(1)

    def init_components(self):
        """Initialize all system components"""
        logger.info("Initializing AI Trading System components...")

        # Initialize AI model
        self.ai_model = EnhancedAIModel(db_path=self.db_path)

        # Initialize WebSocket client
        self.websocket_client = BinanceWebSocketClient(
            db_path=self.db_path,
            min_price=self.config.get("min_price", 0.5)
        )

        # Initialize technical indicators
        self.indicators = TechnicalIndicators(db_path=self.db_path)

        # Initialize signal generator
        self.signal_generator = SignalGenerator(
            db_path=self.db_path,
            ai_model=self.ai_model,
            technical_indicators=self.indicators
        )

        # Set minimum confidence from config
        self.signal_generator.min_confidence = self.config.get("min_confidence", 0.65)

        logger.info("All components initialized")

    def get_available_symbols(self):
        """Get list of available symbols from the market data database"""
        try:
            conn = sqlite3.connect(os.path.join(self.db_path, 'market_data.db'))
            cursor = conn.cursor()

            cursor.execute('SELECT DISTINCT symbol FROM market_data WHERE symbol LIKE "%USDT"')
            symbols = [row[0] for row in cursor.fetchall()]
            conn.close()

            return symbols
        except Exception as e:
            logger.error(f"Error getting available symbols: {e}")
            return []

    def start(self):
        """Start the trading system"""
        if self.running:
            logger.warning("System is already running")
            return

        self.running = True
        logger.info("Starting AI Trading System")

        # Start Telegram bot with detailed logging
        if self.telegram_bot:
            try:
                logger.info(f"Starting Telegram bot of type: {type(self.telegram_bot)}")
                success = self.telegram_bot.start()
                logger.info(f"Telegram bot.start() returned: {success}")

                if success:
                    logger.info("Attempting to send message")
                    sent = self.telegram_bot.send_message("AI Trading System started!")
                    logger.info(f"Message send attempt returned: {sent}")
                else:
                    logger.error("Bot start() returned False")
            except Exception as e:
                logger.error(f"Error starting Telegram bot: {e}", exc_info=True)

        # Start WebSocket client
        self.websocket_client.start()
        logger.info("WebSocket client started")

        # Wait for initial data collection
        logger.info("Waiting for initial data collection (30 seconds)...")
        time.sleep(30)

        # Start main processing threads
        self.start_processing_threads()

        logger.info("AI Trading System is now running")

    def stop(self):
        """Stop the trading system"""
        if not self.running:
            return

        self.running = False
        logger.info("Stopping AI Trading System")

        # Send notification via Telegram if available
        if self.telegram_bot:
            try:
                self.telegram_bot.send_message("AI Trading System is shutting down.")
            except:
                pass

        # Stop WebSocket client
        self.websocket_client.stop()

        logger.info("AI Trading System stopped")

    def start_processing_threads(self):
        """Start all background processing threads"""
        # Thread for updating technical indicators
        indicator_thread = threading.Thread(target=self.indicator_update_loop)
        indicator_thread.daemon = True
        indicator_thread.start()

        # Thread for generating signals
        signal_thread = threading.Thread(target=self.signal_generation_loop)
        signal_thread.daemon = True
        signal_thread.start()

        # Thread for model training
        training_thread = threading.Thread(target=self.model_training_loop)
        training_thread.daemon = True
        training_thread.start()

        # Thread for monitoring active signals
        monitor_thread = threading.Thread(target=self.signal_monitoring_loop)
        monitor_thread.daemon = True
        monitor_thread.start()

    def indicator_update_loop(self):
        """Loop to update technical indicators"""
        while self.running:
            try:
                logger.info("Updating technical indicators")

                # Get current available symbols
                available_symbols = self.get_available_symbols()
                logger.info(f"Processing {len(available_symbols)} symbols for indicator updates")

                for symbol in available_symbols:
                    for timeframe in self.timeframes:
                        self.indicators.calculate_indicators(symbol, timeframe)

                logger.info(f"Indicators updated for {len(available_symbols)} symbols")
            except Exception as e:
                logger.error(f"Error updating indicators: {e}")

            # Wait for next update interval
            time.sleep(self.indicator_update_interval)

    def signal_generation_loop(self):
        """Loop to generate trading signals"""
        # Wait for indicators to be calculated first
        time.sleep(60)

        while self.running:
            try:
                logger.info("Checking for new signals")
                active_signals = self.signal_generator.get_active_signals()

                # Check if we've reached max active signals
                max_active = self.config.get("max_active_signals", 5)
                if len(active_signals) >= max_active:
                    logger.info(f"Maximum active signals reached ({max_active})")
                    time.sleep(self.signal_check_interval)
                    continue

                # Get current available symbols
                available_symbols = self.get_available_symbols()
                logger.info(f"Processing {len(available_symbols)} available symbols for signals")

                # Process each symbol for potential signals
                for symbol in available_symbols:
                    # Generate features for AI model
                    feature_id = self.indicators.generate_features_for_ai(symbol, "1h")

                    if feature_id:
                        # Try to generate a signal
                        signal = self.signal_generator.generate_signal(symbol, feature_id)

                        if signal:
                            logger.info(
                                f"New signal generated: {symbol} {signal['direction']} at {signal['entry_price']}")
                            # Send signal to Telegram if bot is available
                            if self.telegram_bot:
                                try:
                                    self.telegram_bot.send_signal(signal)
                                except Exception as e:
                                    logger.error(f"Error sending signal to Telegram: {e}")

                    # Small delay between symbols
                    time.sleep(1)

            except Exception as e:
                logger.error(f"Error generating signals: {e}")

            # Wait for next check interval
            time.sleep(self.signal_check_interval)

    def model_training_loop(self):
        """Loop to periodically retrain AI models"""
        # Wait before first training
        time.sleep(3600)

        while self.running:
            try:
                logger.info("Starting model retraining")

                # Auto-label outcomes for training data
                self.ai_model.auto_label_outcomes()

                # Get current available symbols
                available_symbols = self.get_available_symbols()
                logger.info(f"Retraining models for {len(available_symbols)} symbols")

                # Retrain models for each symbol
                for symbol in available_symbols:
                    self.ai_model.train_multiple_models(symbol)
                    logger.info(f"Models retrained for {symbol}")
                    time.sleep(10)  # Short delay between symbols

                logger.info("Model retraining complete")

                # Send notification via Telegram if available
                if self.telegram_bot:
                    try:
                        self.telegram_bot.send_message("Model retraining completed successfully.")
                    except:
                        pass

            except Exception as e:
                logger.error(f"Error in model training: {e}")

            # Wait for next training interval
            time.sleep(self.model_retrain_interval)

    def signal_monitoring_loop(self):
        """Loop to monitor active signals for completion"""
        while self.running:
            try:
                active_signals = self.signal_generator.get_active_signals()

                if not active_signals:
                    # No active signals to monitor
                    time.sleep(60)
                    continue

                logger.info(f"Monitoring {len(active_signals)} active signals")

                for signal in active_signals:
                    # Get current price
                    current_price = self._get_current_price(signal['symbol'])

                    if not current_price:
                        continue

                    # Check if stop loss or take profit hit
                    if signal['direction'] == "LONG":
                        # For long positions
                        if current_price <= signal['stop_loss']:
                            # Stop loss hit
                            loss_pct = (current_price - signal['entry_price']) / signal['entry_price'] * 100
                            self.signal_generator.update_signal_status(signal['id'], "LOSS", price=current_price,
                                                                       profit_pct=loss_pct)
                            logger.info(f"Signal {signal['id']} hit stop loss: {loss_pct:.2f}%")

                            # Send notification via Telegram if available
                            if self.telegram_bot:
                                try:
                                    self.telegram_bot.send_message(
                                        f"❌ Stop Loss Hit\nSymbol: {signal['symbol']}\nLoss: {loss_pct:.2f}%"
                                    )
                                except:
                                    pass

                        elif current_price >= signal['take_profit']:
                            # Take profit hit
                            profit_pct = (current_price - signal['entry_price']) / signal['entry_price'] * 100
                            self.signal_generator.update_signal_status(signal['id'], "WIN", price=current_price,
                                                                       profit_pct=profit_pct)
                            logger.info(f"Signal {signal['id']} hit take profit: +{profit_pct:.2f}%")

                            # Send notification via Telegram if available
                            if self.telegram_bot:
                                try:
                                    self.telegram_bot.send_message(
                                        f"✅ Take Profit Hit\nSymbol: {signal['symbol']}\nProfit: +{profit_pct:.2f}%"
                                    )
                                except:
                                    pass
                    else:
                        # For short positions
                        if current_price >= signal['stop_loss']:
                            # Stop loss hit
                            loss_pct = (signal['entry_price'] - current_price) / signal['entry_price'] * 100
                            self.signal_generator.update_signal_status(signal['id'], "LOSS", price=current_price,
                                                                       profit_pct=-loss_pct)
                            logger.info(f"Signal {signal['id']} hit stop loss: {-loss_pct:.2f}%")

                            # Send notification via Telegram if available
                            if self.telegram_bot:
                                try:
                                    self.telegram_bot.send_message(
                                        f"❌ Stop Loss Hit\nSymbol: {signal['symbol']}\nLoss: {-loss_pct:.2f}%"
                                    )
                                except:
                                    pass

                        elif current_price <= signal['take_profit']:
                            # Take profit hit
                            profit_pct = (signal['entry_price'] - current_price) / signal['entry_price'] * 100
                            self.signal_generator.update_signal_status(signal['id'], "WIN", price=current_price,
                                                                       profit_pct=profit_pct)
                            logger.info(f"Signal {signal['id']} hit take profit: +{profit_pct:.2f}%")

                            # Send notification via Telegram if available
                            if self.telegram_bot:
                                try:
                                    self.telegram_bot.send_message(
                                        f"✅ Take Profit Hit\nSymbol: {signal['symbol']}\nProfit: +{profit_pct:.2f}%"
                                    )
                                except:
                                    pass

            except Exception as e:
                logger.error(f"Error monitoring signals: {e}")

            # Wait before next check
            time.sleep(60)

    def _get_current_price(self, symbol):
        """Get current price for a symbol"""
        try:
            conn = sqlite3.connect(os.path.join(self.db_path, 'market_data.db'))
            cursor = conn.cursor()

            cursor.execute('''
            SELECT price FROM market_data
            WHERE symbol = ?
            ORDER BY timestamp DESC
            LIMIT 1
            ''', (symbol,))

            row = cursor.fetchone()
            conn.close()

            return row[0] if row else None
        except Exception as e:
            logger.error(f"Error getting price for {symbol}: {e}")
            return None

def test_telegram():
    system = AITradingSystem()
    if system.telegram_bot:
        success = system.telegram_bot.send_message("Test message from AI Trading System")
        print(f"Telegram message sent: {success}")
    else:
        print("Telegram bot not initialized")

def main():
    """Main entry point for the application"""
    parser = argparse.ArgumentParser(description='AI Trading System')
    parser.add_argument('--db_path', default='data', help='Path to database directory')
    parser.add_argument('--config', default='config.json', help='Path to configuration file')
    parser.add_argument('--web', action='store_true', help='Start web dashboard')
    parser.add_argument('--port', type=int, default=5000, help='Web dashboard port')
    args = parser.parse_args()

    logger.info("Starting AI Trading System")
    trading_system = AITradingSystem(db_path=args.db_path, config_path=args.config)

    try:
        trading_system.start()

        # Start web dashboard if requested
        if args.web:
            import threading
            dashboard_thread = threading.Thread(
                target=run_dashboard,
                args=(trading_system, args.port)
            )
            dashboard_thread.daemon = True
            dashboard_thread.start()
            logger.info(f"Web dashboard started on http://localhost:{args.port}")

        # Keep main thread alive to allow background threads to run
        while True:
            time.sleep(1)

    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received, shutting down")
        trading_system.stop()
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        trading_system.stop()


if __name__ == "__main__":
    main()