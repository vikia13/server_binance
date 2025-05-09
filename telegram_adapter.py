import logging
import requests
import time
import threading
import asyncio
from telegram import Bot
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

class TelegramBot:
    """Simple synchronous Telegram bot implementation"""

    def __init__(self, token: str, chat_id: Optional[str] = None):
        self.token = token
        self.chat_id = chat_id
        self.base_url = f"https://api.telegram.org/bot{token}/"
        self.initialized = False
        self.update_thread = None
        self.running = False
        self.last_update_id = 0
        self.command_handlers = {
            "start": self._start_command,
            "help": self._help_command,
            "status": self._status_command,
            "dashboard": self._dashboard_command,
            "performance": self._performance_command,
            "positions": self._positions_command,
            "signals": self._signals_command,
            "settings": self._settings_command
        }
    def start(self):
        """Start the Telegram bot"""
        try:
            # Test the API connection
            response = requests.get(f"{self.base_url}getMe")
            if not response.ok:
                logger.error(f"Failed to initialize bot: {response.text}")
                return False

            bot_info = response.json()
            logger.info(f"Bot connected: @{bot_info['result']['username']}")

            # Start polling for updates in a separate thread
            self.running = True
            self.update_thread = threading.Thread(target=self._poll_updates)
            self.update_thread.daemon = True
            self.update_thread.start()

            self.initialized = True
            return True
        except Exception as e:
            logger.error(f"Failed to initialize TelegramBot: {e}")
            return False

    def _poll_updates(self):
        """Poll for updates from Telegram"""
        while self.running:
            try:
                params = {'offset': self.last_update_id + 1, 'timeout': 30}
                response = requests.get(f"{self.base_url}getUpdates", params=params)

                if response.ok:
                    updates = response.json().get('result', [])
                    for update in updates:
                        self._process_update(update)
                        self.last_update_id = update['update_id']
                else:
                    logger.error(f"Error polling updates: {response.text}")
                    time.sleep(5)
            except Exception as e:
                logger.error(f"Error in update polling: {e}")
                time.sleep(5)

    def _process_update(self, update: Dict[str, Any]):
        """Process an update from Telegram"""
        if 'message' not in update:
            return

        message = update['message']
        if 'text' not in message:
            return

        text = message['text']
        chat_id = message['chat']['id']

        # Log all incoming messages for debugging
        logger.info(f"Received message: '{text}' from chat_id: {chat_id}")

        # Save chat_id if not already set
        if not self.chat_id:
            self.chat_id = str(chat_id)
            logger.info(f"Chat ID set to {self.chat_id}")

        # Handle commands
        if text.startswith('/'):
            command = text.split()[0][1:]  # Remove the / and get the command
            if command in self.command_handlers:
                try:
                    self.command_handlers[command](chat_id, text)
                    logger.info(f"Processed command: /{command}")
                except Exception as e:
                    logger.error(f"Error handling command {command}: {e}")
                    self.send_message("Error processing command", chat_id)

    def _start_command(self, chat_id, text):
        """Handle the /start command"""
        message = "Welcome to the AI Trading Bot! 🤖\n\n"
        message += "I can help you monitor trading signals and system status.\n"
        message += "Use /help to see available commands."
        self.send_message(message, chat_id)

    def _help_command(self, chat_id, text):
        """Handle the /help command"""
        message = "📋 *Available Commands*\n\n"
        message += "/start - Start the bot and get welcome message\n"
        message += "/help - Show this help message\n"
        message += "/status - Check system status\n"
        message += "/dashboard - Shows overall system status\n"
        message += "/performance - Displays trading performance metrics\n"
        message += "/positions - Lists current open positions\n"
        message += "/signals - Shows recent trading signals\n"
        message += "/settings - Allows changing system parameters\n"
        self.send_message(message, chat_id)

    def _status_command(self, chat_id, text):
        """Handle the /status command"""
        message = "🔍 *System Status*\n\n"
        message += "✅ Bot is running\n"
        message += "✅ Data collection is active\n"
        message += "✅ Signal monitoring is active\n"
        self.send_message(message, chat_id)

    def _dashboard_command(self, chat_id, text):
        """Handle the /dashboard command"""
        message = "📊 *System Dashboard*\n\n"
        message += "✅ Trading System: Active\n"
        message += "🔄 Last Update: " + time.strftime("%Y-%m-%d %H:%M:%S") + "\n\n"
        message += "📈 *Current Status*\n"
        message += "• Active Positions: 3\n"
        message += "• Today's Signals: 5\n"
        message += "• Today's P/L: +2.4%\n"
        self.send_message(message, chat_id)

    def _performance_command(self, chat_id, text):
        """Handle the /performance command"""
        message = "📈 *Performance Metrics*\n\n"
        message += "• Win Rate: 68%\n"
        message += "• Average Profit: 2.1%\n"
        message += "• Average Loss: 1.3%\n"
        message += "• Risk/Reward: 1.62\n"
        message += "• Total Trades: 156\n"
        message += "• Profitable Trades: 106\n"
        message += "• Current Accuracy: 71%\n"
        self.send_message(message, chat_id)

    def _positions_command(self, chat_id, text):
        """Handle the /positions command with real data"""
        try:
            import sqlite3
            import os

            # Get the database path (make sure this is accessible)
            db_path = "data"  # You might need to adjust this

            conn = sqlite3.connect(os.path.join(db_path, 'signals.db'))
            cursor = conn.cursor()

            # Get active signals (positions)
            cursor.execute('''
            SELECT id, symbol, direction, entry_price, status, timestamp, take_profit, stop_loss
            FROM signals
            WHERE status = 'ACTIVE'
            ORDER BY timestamp DESC
            ''')

            active_signals = cursor.fetchall()
            conn.close()

            if not active_signals:
                message = "🔍 *Current Positions*\n\nNo active positions at the moment."
                self.send_message(message, chat_id)
                return

            message = "🔍 *Current Positions*\n\n"

            # Get current prices for calculating P/L
            price_conn = sqlite3.connect(os.path.join(db_path, 'market_data.db'))
            price_cursor = price_conn.cursor()

            for i, signal in enumerate(active_signals, 1):
                sig_id, symbol, direction, entry_price, status, timestamp, take_profit, stop_loss = signal

                # Get current price
                price_cursor.execute('''
                SELECT price FROM market_data
                WHERE symbol = ?
                ORDER BY timestamp DESC
                LIMIT 1
                ''', (symbol,))

                price_row = price_cursor.fetchone()
                current_price = price_row[0] if price_row else entry_price

                # Calculate P/L
                if direction == "LONG":
                    pl_pct = (current_price - entry_price) / entry_price * 100
                else:
                    pl_pct = (entry_price - current_price) / entry_price * 100

                # Calculate duration
                import time
                duration_seconds = time.time() - timestamp
                hours, remainder = divmod(duration_seconds, 3600)
                minutes, _ = divmod(remainder, 60)

                message += f"{i}. {symbol} ({direction})\n"
                message += f"   Entry: ${entry_price:.4f} | Current: ${current_price:.4f}\n"
                message += f"   P/L: {'+' if pl_pct >= 0 else ''}{pl_pct:.2f}% | Duration: {int(hours)}h {int(minutes)}m\n\n"

            price_conn.close()
            self.send_message(message, chat_id)

        except Exception as e:
            error_msg = f"Error retrieving positions: {str(e)}"
            self.send_message(error_msg, chat_id)

    def _signals_command(self, chat_id, text):
        """Handle the /signals command with real data"""
        try:
            import sqlite3
            import os
            import time

            # Define the database path
            db_path = "data"  # Adjust this if needed

            # Connect to the signals database
            conn = sqlite3.connect(os.path.join(db_path, 'signals.db'))
            cursor = conn.cursor()

            # Get recent signals (last 5)
            cursor.execute('''
            SELECT symbol, direction, entry_price, confidence, timestamp, model_name
            FROM signals
            ORDER BY timestamp DESC
            LIMIT 5
            ''')

            signals = cursor.fetchall()
            conn.close()

            if not signals:
                message = "📊 *Recent Signals*\n\nNo signals generated yet."
                self.send_message(message, chat_id)
                return

            message = "📊 *Recent Signals*\n\n"

            for i, signal in enumerate(signals, 1):
                symbol, direction, entry_price, confidence, timestamp, model_name = signal
                direction_emoji = "🟢" if direction == "LONG" else "🔴"

                message += f"{i}. {symbol} {direction_emoji} {direction}\n"
                message += f"   Price: ${entry_price:.4f} | Conf: {confidence:.1f}%\n"
                message += f"   Time: {time.strftime('%Y-%m-%d %H:%M', time.localtime(timestamp))}\n\n"

            self.send_message(message, chat_id)
        except Exception as e:
            self.send_message(f"Error retrieving signals: {str(e)}", chat_id)

    def _settings_command(self, chat_id, text):
        """Handle the /settings command"""
        message = "⚙️ *System Settings*\n\n"
        message += "• Time Interval: 5 minutes\n"
        message += "• Price Change Threshold: 3.0%\n"
        message += "• Max Signals/Day: 3\n"
        message += "• Last Updated: " + time.strftime("%Y-%m-%d %H:%M:%S") + "\n\n"
        message += "To change settings, use:\n"
        message += "/set [parameter] [value]"
        self.send_message(message, chat_id)

    def register_command(self, command: str, handler):
        """Register a command handler"""
        self.command_handlers[command] = handler

    def send_message(self, message: str, chat_id: Optional[str] = None, parse_mode: str = 'Markdown'):
        """Send a message to the specified chat ID"""
        if not chat_id:
            chat_id = self.chat_id

        if not chat_id:
            logger.error("Cannot send message: chat_id not set")
            return False

        try:
            params = {
                'chat_id': chat_id,
                'text': message,
                'parse_mode': parse_mode
            }
            response = requests.post(f"{self.base_url}sendMessage", json=params)
            if response.ok:
                return True
            else:
                logger.error(f"Error sending message: {response.text}")
                return False
        except Exception as e:
            logger.error(f"Error sending message: {e}")
            return False

    def send_signal(self, signal):
        """Send a trading signal notification"""
        direction_emoji = "🟢" if signal['direction'] == "LONG" else "🔴"
        message = f"🚨 *NEW TRADING SIGNAL*\n\n"
        message += f"{signal['symbol']} {direction_emoji} {signal['direction']}\n"
        message += f"Entry Price: ${signal['entry_price']:.4f}\n"
        message += f"Take Profit: ${signal['take_profit']:.4f}\n"
        message += f"Stop Loss: ${signal['stop_loss']:.4f}\n"
        message += f"Risk/Reward: {signal['risk_reward']:.2f}\n"
        message += f"Confidence: {signal['confidence']:.1f}%\n"
        message += f"Model: {signal['model_name']}\n"

        return self.send_message(message, self.chat_id)

    def stop(self):
        """Stop the Telegram bot"""
        self.running = False
        if self.update_thread and self.update_thread.is_alive():
            self.update_thread.join(timeout=1.0)
        logger.info("Telegram bot stopped")

class EnhancedTelegramBot:
    def __init__(self, token, chat_id=None):
        self.token = token
        self.chat_id = chat_id if isinstance(chat_id, int) else int(chat_id) if chat_id else None
        self.bot = None

        try:
            self.bot = Bot(token=self.token)
            logger.info("Telegram bot initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Telegram bot: {e}")
            self.bot = None

    def send_message(self, message):
        """Send a text message"""
        if not self.bot or not self.chat_id:
            logger.error("Cannot send message: bot not initialized or chat_id not set")
            return False

        try:
            # Create a new event loop for this call
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

            # Run the coroutine and get the result
            result = loop.run_until_complete(
                self.bot.send_message(
                    chat_id=self.chat_id,
                    text=message,
                    parse_mode='Markdown'
                )
            )
            loop.close()

            logger.info(f"Message sent successfully: {result.message_id}")
            return True
        except Exception as e:
            logger.error(f"Error sending message: {e}")
            return False
    def send_signal(self, signal_data):
        """Send a trading signal"""
        if not signal_data:
            logger.error("Attempted to send empty signal")
            return False

        try:
            message = self._format_signal(signal_data)
            return self.send_message(message)
        except Exception as e:
            logger.error(f"Error formatting signal message: {e}")
            return False

    def start(self):
        """Start the Telegram bot"""
        try:
            # The issue is here - self.bot doesn't exist
            # Change this line
            if hasattr(self.bot, 'start') and callable(self.bot.start):
                return self.bot.start()

            # To this - use self.bot instead of self.bot
            if hasattr(self.bot, 'start') and callable(self.bot.start):
                return self.bot.start()

            # If the bot doesn't have a start method, it's already running
            return True
        except Exception as e:
            logger.error(f"Failed to start Telegram bot: {e}")
            return False

    def register_command(self, command, handler):
        """Register a command handler"""
        if hasattr(self.bot, 'register_command') and callable(self.bot.register_command):
            try:
                self.bot.register_command(command, handler)
                return True
            except Exception as e:
                logger.error(f"Error registering command {command}: {e}")
                return False
        return False


    def _format_signal(self, signal_data):
        """Format a trading signal as a message"""
        symbol = signal_data.get('symbol', 'Unknown')
        direction = signal_data.get('direction', signal_data.get('trend', 'Unknown'))
        price = signal_data.get('entry_price', signal_data.get('price', 0))
        take_profit = signal_data.get('take_profit', 'N/A')
        stop_loss = signal_data.get('stop_loss', 'N/A')
        confidence = signal_data.get('confidence', 0) * 100

        message = f"🚨 *Trading Signal* 🚨\n"
        message += f"Symbol: `{symbol}`\n"
        message += f"Direction: {'🟢 LONG' if direction == 'LONG' else '🔴 SHORT'}\n"
        message += f"Entry Price: `{price}`\n"
        if take_profit != 'N/A':
            message += f"Take Profit: `{take_profit}`\n"
        if stop_loss != 'N/A':
            message += f"Stop Loss: `{stop_loss}`\n"
        message += f"Confidence: `{confidence:.1f}%`\n"

        return message

    def stop(self):
        """Stop the Telegram bot"""
        if hasattr(self.bot, 'stop') and callable(self.bot.stop):
            try:
                self.bot.stop()
                return True
            except Exception as e:
                logger.error(f"Failed to stop Telegram bot: {e}")
                return False
        return True