import os
import logging
import threading
import time
import json
import sqlite3
import asyncio
from datetime import datetime
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes

logger = logging.getLogger(__name__)
class TelegramBot:
    def __init__(self, token, chat_id=None):
        self.token = token
        self.chat_id = chat_id
        self.application = None

        if self.token:
            try:
                # Create bot first without Application
                import telegram
                self.bot = telegram.Bot(self.token)

                # Then create application
                builder = Application.builder().token(self.token)
                self.application = builder.build()

                # Handler registration unchanged
                self.application.add_handler(CommandHandler("start", self._start_command))
                self.application.add_handler(CommandHandler("help", self._help_command))
                self.application.add_handler(CommandHandler("status", self._status_command))

                logger.info("TelegramBot initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize TelegramBot: {e}")
                self.application = None

    def start(self):
        """Start the bot polling"""
        if not self.application:
            logger.error("Cannot start: application not initialized")
            return False

        try:
            # Run the bot in a separate thread
            self.polling_thread = threading.Thread(target=self._run_polling)
            self.polling_thread.daemon = True
            self.polling_thread.start()
            logger.info("TelegramBot polling started")
            return True
        except Exception as e:
            logger.error(f"Failed to start TelegramBot polling: {e}")
            return False

    def _run_polling(self):
        """Run polling in a separate thread"""
        asyncio.run(self.application.run_polling())

    def stop(self):
        """Stop the bot"""
        if self.application:
            asyncio.run(self.application.shutdown())
            logger.info("TelegramBot stopped")

    def send_message(self, message, parse_mode='Markdown'):
        """Send a message to the configured chat_id"""
        if not self.application or not self.chat_id:
            logger.error("Cannot send message: bot not initialized or chat_id not set")
            return False

        try:
            # Create a new event loop to send message from sync code
            async def send():
                await self.application.bot.send_message(
                    chat_id=self.chat_id,
                    text=message,
                    parse_mode=parse_mode
                )

            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            loop.run_until_complete(send())
            loop.close()
            return True
        except Exception as e:
            logger.error(f"Failed to send message: {e}")
            return False

    async def _start_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /start command"""
        await update.message.reply_text(
            "Bot started! I'll send you trading signals and updates."
        )
        # Store this chat ID if we don't have one yet
        if not self.chat_id:
            self.chat_id = update.effective_chat.id
            logger.info(f"Chat ID set to {self.chat_id}")

    async def _help_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /help command"""
        await update.message.reply_text(
            "Available commands:\n"
            "/start - Start the bot\n"
            "/help - Show this help message\n"
            "/status - Show current status"
        )

    async def _status_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handler for /status command"""
        await update.message.reply_text("Trading system is active and monitoring markets.")

    async def _handle_message(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handler for regular messages"""
        await update.message.reply_text(
            "I don't understand that command. Try /help to see available commands."
        )

    def send_signal(self, signal):
        """Format and send a trading signal"""
        if not signal:
            return False

        try:
            message = f"🚨 *New Trading Signal* 🚨\n"
            message += f"Symbol: `{signal['symbol']}`\n"
            message += f"Direction: {'🟢 LONG' if signal['direction'] == 'LONG' else '🔴 SHORT'}\n"
            message += f"Entry Price: `{signal['entry_price']}`\n"
            message += f"Take Profit: `{signal['take_profit']}`\n"
            message += f"Stop Loss: `{signal['stop_loss']}`\n"

            return self.send_message(message)
        except Exception as e:
            logger.error(f"Error formatting signal: {e}")
            return False


class TradingTelegramBot:
    def __init__(self, db_path='data', config_path='config.json'):
        self.db_path = db_path
        self.config_path = config_path
        self.config = self._load_config()
        self.token = self.config.get('telegram', {}).get('token')
        self.allowed_users = self.config.get('telegram', {}).get('allowed_users', [])

        if not self.token:
            logger.error("Telegram bot token not found in config")
            raise ValueError("Telegram bot token not configured")

        try:
            self.application = Application.builder().token(self.token).build()

            # Register command handlers
            self.application.add_handler(CommandHandler("start", self._start_command))
            self.application.add_handler(CommandHandler("status", self._status_command))
            self.application.add_handler(CommandHandler("signals", self._signals_command))
            self.application.add_handler(CommandHandler("history", self._history_command))
            self.application.add_handler(CommandHandler("stats", self._stats_command))

            # Last processed signal ID to avoid sending duplicates
            self.last_signal_id = self._get_latest_signal_id()
            self.notifier_running = False

            logger.info("TradingTelegramBot initialized")
        except Exception as e:
            logger.error(f"Failed to initialize TradingTelegramBot: {e}")
            self.application = None

    def _load_config(self):
        """Load configuration from file"""
        try:
            with open(self.config_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error loading config: {e}")
            return {}

    def start(self):
        """Start the bot and notification thread"""
        if not self.token or not self.application:
            logger.error("Cannot start Telegram bot: token not configured or initialization failed")
            return False

        try:
            # Start bot in a separate thread
            self.polling_thread = threading.Thread(target=self._run_polling)
            self.polling_thread.daemon = True
            self.polling_thread.start()
            logger.info("TradingTelegramBot polling started")

            # Start notification thread
            self.notifier_running = True
            self.notifier_thread = threading.Thread(target=self._notification_loop)
            self.notifier_thread.daemon = True
            self.notifier_thread.start()
            logger.info("Notification thread started")

            return True
        except Exception as e:
            logger.error(f"Failed to start TradingTelegramBot: {e}")
            return False

    def _run_polling(self):
        """Run polling in a separate thread"""
        asyncio.run(self.application.run_polling())

    def stop(self):
        """Stop the bot and notification thread"""
        self.notifier_running = False
        if self.application:
            asyncio.run(self.application.shutdown())
        logger.info("TradingTelegramBot stopped")

    def _notification_loop(self):
        """Background thread to check for new signals and send notifications"""
        while self.notifier_running:
            try:
                # Check for new signals
                new_signals = self._get_new_signals()
                for signal in new_signals:
                    self._send_signal_notification(signal)
                    self.last_signal_id = max(self.last_signal_id, signal['id'])

                # Check for completed signals
                completed_signals = self._get_completed_signals()
                for signal in completed_signals:
                    self._send_completed_notification(signal)

            except Exception as e:
                logger.error(f"Error in notification loop: {e}")

            # Check every 30 seconds
            time.sleep(30)

    def _get_latest_signal_id(self):
        """Get ID of the latest signal in database"""
        try:
            conn = sqlite3.connect(os.path.join(self.db_path, 'signals.db'))
            cursor = conn.cursor()
            cursor.execute('SELECT MAX(id) FROM signals')
            max_id = cursor.fetchone()[0]
            conn.close()
            return max_id if max_id else 0
        except Exception as e:
            logger.error(f"Error getting latest signal ID: {e}")
            return 0

    def _get_new_signals(self):
        """Get new signals since last check"""
        try:
            conn = sqlite3.connect(os.path.join(self.db_path, 'signals.db'))
            cursor = conn.cursor()

            cursor.execute('''
            SELECT id, symbol, direction, entry_price, stop_loss, take_profit, 
                   risk_reward_ratio, confidence, timestamp
            FROM signals
            WHERE id > ? AND status = 'PENDING'
            ORDER BY timestamp DESC
            ''', (self.last_signal_id,))

            rows = cursor.fetchall()
            conn.close()

            signals = []
            for row in rows:
                signals.append({
                    'id': row[0],
                    'symbol': row[1],
                    'direction': row[2],
                    'entry_price': row[3],
                    'stop_loss': row[4],
                    'take_profit': row[5],
                    'risk_reward_ratio': row[6],
                    'confidence': row[7],
                    'timestamp': row[8]
                })

            return signals
        except Exception as e:
            logger.error(f"Error getting new signals: {e}")
            return []

    def _get_completed_signals(self):
        """Get recently completed signals"""
        try:
            conn = sqlite3.connect(os.path.join(self.db_path, 'signals.db'))
            cursor = conn.cursor()

            # Get signals closed in the last hour that haven't been notified
            one_hour_ago = int(time.time() * 1000) - (60 * 60 * 1000)

            cursor.execute('''
            SELECT id, symbol, direction, entry_price, outcome, profit_pct, closed_at
            FROM signals
            WHERE closed_at > ? AND status IN ('WIN', 'LOSS')
            AND id NOT IN (SELECT signal_id FROM notifications WHERE type = 'completion')
            ORDER BY closed_at DESC
            ''', (one_hour_ago,))

            rows = cursor.fetchall()

            signals = []
            for row in rows:
                signals.append({
                    'id': row[0],
                    'symbol': row[1],
                    'direction': row[2],
                    'entry_price': row[3],
                    'outcome': row[4],
                    'profit_pct': row[5],
                    'closed_at': row[6]
                })

                # Mark as notified
                cursor.execute('''
                INSERT INTO notifications (signal_id, type, timestamp)
                VALUES (?, 'completion', ?)
                ''', (row[0], int(time.time() * 1000)))

            conn.commit()
            conn.close()

            return signals
        except Exception as e:
            logger.error(f"Error getting completed signals: {e}")

            # Create notifications table if it doesn't exist
            try:
                conn = sqlite3.connect(os.path.join(self.db_path, 'signals.db'))
                cursor = conn.cursor()
                cursor.execute('''
                CREATE TABLE IF NOT EXISTS notifications (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    signal_id INTEGER,
                    type TEXT,
                    timestamp INTEGER
                )
                ''')
                conn.commit()
                conn.close()
            except Exception as nested_e:
                logger.error(f"Error creating notifications table: {nested_e}")

            return []

    def _send_signal_notification(self, signal):
        """Send notification about new signal"""
        try:
            emoji = "🟢" if signal['direction'] == "LONG" else "🔴"
            timestamp = datetime.fromtimestamp(signal['timestamp'] / 1000).strftime('%Y-%m-%d %H:%M:%S')

            message = f"{emoji} *NEW SIGNAL #{signal['id']}*\n\n" \
                      f"*Symbol:* {signal['symbol']}\n" \
                      f"*Direction:* {signal['direction']}\n" \
                      f"*Entry:* {signal['entry_price']}\n" \
                      f"*Stop Loss:* {signal['stop_loss']}\n" \
                      f"*Take Profit:* {signal['take_profit']}\n" \
                      f"*Risk/Reward:* {signal['risk_reward_ratio']:.2f}\n" \
                      f"*Confidence:* {signal['confidence']:.2f}\n" \
                      f"*Time:* {timestamp}"

            self._broadcast_message(message)
        except Exception as e:
            logger.error(f"Error sending signal notification: {e}")

    def _send_completed_notification(self, signal):
        """Send notification about completed signal"""
        try:
            if signal['outcome'] == 'WIN':
                emoji = "✅"
                result = "WIN"
            else:
                emoji = "❌"
                result = "LOSS"

            timestamp = datetime.fromtimestamp(signal['closed_at'] / 1000).strftime('%Y-%m-%d %H:%M:%S')

            message = f"{emoji} *SIGNAL #{signal['id']} COMPLETED*\n\n" \
                      f"*Symbol:* {signal['symbol']}\n" \
                      f"*Direction:* {signal['direction']}\n" \
                      f"*Result:* {result}\n" \
                      f"*Profit/Loss:* {signal['profit_pct']:.2f}%\n" \
                      f"*Closed at:* {timestamp}"

            self._broadcast_message(message)
        except Exception as e:
            logger.error(f"Error sending completion notification: {e}")

    def _broadcast_message(self, message):
        """Send message to all allowed users"""
        for user_id in self.allowed_users:
            try:
                # Create a new event loop to send message from sync code
                async def send():
                    await self.application.bot.send_message(
                        chat_id=user_id,
                        text=message,
                        parse_mode='Markdown'
                    )

                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                loop.run_until_complete(send())
                loop.close()
            except Exception as e:
                logger.error(f"Error sending message to user {user_id}: {e}")

    async def _start_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /start command"""
        user_id = update.effective_user.id
        if user_id not in self.allowed_users:
            await update.message.reply_text("Unauthorized access. Your user ID has been logged.")
            logger.warning(f"Unauthorized access attempt from user ID: {user_id}")
            return

        await update.message.reply_text(
            "Welcome to the AI Trading Bot!\n\n"
            "Available commands:\n"
            "/status - Show system status\n"
            "/signals - Show active signals\n"
            "/history - Show signal history\n"
            "/stats - Show performance statistics"
        )

    async def _status_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /status command"""
        user_id = update.effective_user.id
        if user_id not in self.allowed_users:
            return

        try:
            # Get system info
            system_status = self._get_system_status()

            message = f"*System Status*\n\n" \
                      f"Running: {system_status['running']}\n" \
                      f"Symbols monitored: {len(system_status['symbols'])}\n" \
                      f"Active signals: {system_status['active_signals']}\n" \
                      f"Win/Loss ratio: {system_status['win_ratio']:.2f}\n" \
                      f"Last update: {system_status['last_update']}"

            await update.message.reply_text(message, parse_mode='Markdown')
        except Exception as e:
            logger.error(f"Error handling status command: {e}")
            await update.message.reply_text("Error retrieving system status")

    async def _signals_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /signals command"""
        user_id = update.effective_user.id
        if user_id not in self.allowed_users:
            return

        try:
            # Get active signals
            signals = self._get_active_signals()

            if not signals:
                await update.message.reply_text("No active signals at the moment")
                return

            message = "*Active Signals*\n\n"

            for signal in signals[:5]:  # Limit to 5 signals to avoid message size limits
                emoji = "🟢" if signal['direction'] == "LONG" else "🔴"
                message += f"{emoji} *#{signal['id']} {signal['symbol']}*\n" \
                           f"Direction: {signal['direction']}\n" \
                           f"Entry: {signal['entry_price']}\n" \
                           f"SL: {signal['stop_loss']} | TP: {signal['take_profit']}\n\n"

            await update.message.reply_text(message, parse_mode='Markdown')
        except Exception as e:
            logger.error(f"Error handling signals command: {e}")
            await update.message.reply_text("Error retrieving active signals")

    async def _history_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /history command"""
        user_id = update.effective_user.id
        if user_id not in self.allowed_users:
            return

        try:
            # Get recent signal history
            history = self._get_signal_history(5)

            if not history:
                await update.message.reply_text("No signal history available")
                return

            message = "*Recent Signals*\n\n"

            for signal in history:
                if signal['outcome'] == 'WIN':
                    emoji = "✅"
                else:
                    emoji = "❌"

                message += f"{emoji} *#{signal['id']} {signal['symbol']}*\n" \
                           f"{signal['direction']} | {signal['profit_pct']:.2f}%\n\n"

            await update.message.reply_text(message, parse_mode='Markdown')
        except Exception as e:
            logger.error(f"Error handling history command: {e}")
            await update.message.reply_text("Error retrieving signal history")

    async def _stats_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /stats command"""
        user_id = update.effective_user.id
        if user_id not in self.allowed_users:
            return

        try:
            # Get performance statistics
            stats = self._get_performance_stats()

            message = f"*Performance Statistics*\n\n" \
                      f"Total signals: {stats['total']}\n" \
                      f"Win/Loss: {stats['wins']}/{stats['losses']}\n" \
                      f"Win rate: {stats['win_rate']:.2f}%\n" \
                      f"Average profit: {stats['avg_profit']:.2f}%\n" \
                      f"Average loss: {stats['avg_loss']:.2f}%\n" \
                      f"Best trade: {stats['best_trade']:.2f}%\n" \
                      f"Worst trade: {stats['worst_trade']:.2f}%\n"

            await update.message.reply_text(message, parse_mode='Markdown')
        except Exception as e:
            logger.error(f"Error handling stats command: {e}")
            await update.message.reply_text("Error retrieving performance statistics")

    def _get_system_status(self):
        """Get system status information"""
        try:
            conn = sqlite3.connect(os.path.join(self.db_path, 'signals.db'))
            cursor = conn.cursor()

            # Get active signals count
            cursor.execute("SELECT COUNT(*) FROM signals WHERE status IN ('PENDING', 'ACTIVE')")
            active_count = cursor.fetchone()[0]

            # Get win/loss counts
            cursor.execute("SELECT COUNT(*) FROM signals WHERE status = 'WIN'")
            win_count = cursor.fetchone()[0]

            cursor.execute("SELECT COUNT(*) FROM signals WHERE status = 'LOSS'")
            loss_count = cursor.fetchone()[0]

            # Calculate win ratio
            win_ratio = win_count / (win_count + loss_count) if (win_count + loss_count) > 0 else 0

            conn.close()

            return {
                'running': True,  # Assume system is running if the bot is running
                'symbols': self.config.get('symbols', []),
                'active_signals': active_count,
                'win_ratio': win_ratio,
                'last_update': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
        except Exception as e:
            logger.error(f"Error getting system status: {e}")
            return {
                'running': False,
                'symbols': [],
                'active_signals': 0,
                'win_ratio': 0,
                'last_update': 'Unknown'
            }

    def _get_active_signals(self):
        """Get active trading signals"""
        try:
            conn = sqlite3.connect(os.path.join(self.db_path, 'signals.db'))
            cursor = conn.cursor()

            cursor.execute('''
            SELECT id, symbol, direction, entry_price, stop_loss, take_profit, status
            FROM signals
            WHERE status IN ('PENDING', 'ACTIVE')
            ORDER BY timestamp DESC
            ''')

            rows = cursor.fetchall()
            conn.close()

            return [
                {
                    'id': row[0],
                    'symbol': row[1],
                    'direction': row[2],
                    'entry_price': row[3],
                    'stop_loss': row[4],
                    'take_profit': row[5],
                    'status': row[6]
                }
                for row in rows
            ]
        except Exception as e:
            logger.error(f"Error getting active signals: {e}")
            return []

    def _get_signal_history(self, limit=10):
        """Get recent signal history"""
        try:
            conn = sqlite3.connect(os.path.join(self.db_path, 'signals.db'))
            cursor = conn.cursor()

            cursor.execute('''
            SELECT id, symbol, direction, outcome, profit_pct
            FROM signals
            WHERE status IN ('WIN', 'LOSS')
            ORDER BY closed_at DESC
            LIMIT ?
            ''', (limit,))

            rows = cursor.fetchall()
            conn.close()

            return [
                {
                    'id': row[0],
                    'symbol': row[1],
                    'direction': row[2],
                    'outcome': row[3],
                    'profit_pct': row[4]
                }
                for row in rows
            ]
        except Exception as e:
            logger.error(f"Error getting signal history: {e}")
            return []

    def _get_performance_stats(self):
        """Get performance statistics"""
        try:
            conn = sqlite3.connect(os.path.join(self.db_path, 'signals.db'))
            cursor = conn.cursor()

            # Get total signals
            cursor.execute("SELECT COUNT(*) FROM signals")
            total = cursor.fetchone()[0]

            # Get wins and losses
            cursor.execute("SELECT COUNT(*) FROM signals WHERE status = 'WIN'")
            wins = cursor.fetchone()[0]

            cursor.execute("SELECT COUNT(*) FROM signals WHERE status = 'LOSS'")
            losses = cursor.fetchone()[0]

            # Calculate win rate
            win_rate = (wins / (wins + losses) * 100) if (wins + losses) > 0 else 0

            # Get average profit
            cursor.execute("SELECT AVG(profit_pct) FROM signals WHERE status = 'WIN'")
            avg_profit = cursor.fetchone()[0] or 0

            # Get average loss
            cursor.execute("SELECT AVG(profit_pct) FROM signals WHERE status = 'LOSS'")
            avg_loss = cursor.fetchone()[0] or 0

            # Get best and worst trades
            cursor.execute("SELECT MAX(profit_pct) FROM signals")
            best_trade = cursor.fetchone()[0] or 0

            cursor.execute("SELECT MIN(profit_pct) FROM signals WHERE profit_pct IS NOT NULL")
            worst_trade = cursor.fetchone()[0] or 0

            conn.close()

            return {
                'total': total,
                'wins': wins,
                'losses': losses,
                'win_rate': win_rate,
                'avg_profit': avg_profit,
                'avg_loss': avg_loss,
                'best_trade': best_trade,
                'worst_trade': worst_trade
            }
        except Exception as e:
            logger.error(f"Error getting performance stats: {e}")
            return {
                'total': 0,
                'wins': 0,
                'losses': 0,
                'win_rate': 0,
                'avg_profit': 0,
                'avg_loss': 0,
                'best_trade': 0,
                'worst_trade': 0
            }