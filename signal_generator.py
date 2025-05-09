import sqlite3
import os
import logging
import time
import json
from datetime import datetime

logger = logging.getLogger(__name__)


class SignalGenerator:
    def __init__(self, db_path='data', ai_model=None, technical_indicators=None):
        self.db_path = db_path
        self.ai_model = ai_model
        self.technical_indicators = technical_indicators
        self.signal_counter = 0
        self.min_confidence = 0.65  # Minimum confidence threshold

        # Initialize database
        self.init_database()

        logger.info("Signal generator initialized")

    def init_database(self):
        """Initialize database tables for storing signals"""
        os.makedirs(self.db_path, exist_ok=True)

        conn = sqlite3.connect(os.path.join(self.db_path, 'signals.db'))
        cursor = conn.cursor()

        # Create signals table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS signals (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol TEXT NOT NULL,
            direction TEXT NOT NULL,
            entry_price REAL NOT NULL,
            stop_loss REAL NOT NULL,
            take_profit REAL NOT NULL,
            risk_reward_ratio REAL NOT NULL,
            confidence REAL NOT NULL,
            model_name TEXT,
            feature_id INTEGER,
            status TEXT DEFAULT 'PENDING',
            confirmed INTEGER DEFAULT 0,
            outcome TEXT,
            profit_pct REAL,
            timestamp INTEGER NOT NULL,
            closed_at INTEGER
        )
        ''')

        conn.commit()
        conn.close()

        logger.info("Signal database initialized")

    def generate_signal(self, symbol, feature_id):
        """Generate a trading signal based on AI prediction and technical validation"""
        logger.debug(f"Attempting to generate signal for {symbol} with feature_id {feature_id}")

        if not self.ai_model:
            logger.error("AI model not initialized")
            return None

        # Get prediction from AI model
        logger.debug(f"Getting prediction for {symbol}")
        prediction_type, confidence = self.ai_model.predict(symbol, feature_id)
        logger.debug(f"Prediction for {symbol}: {prediction_type} with confidence {confidence:.2f}")

        # Skip if prediction is throttled or neutral
        if prediction_type == "THROTTLED" or prediction_type == "NEUTRAL":
            logger.debug(f"Skipping {symbol}: prediction type is {prediction_type}")
            return None

        # Get latest price and technical indicators
        latest_price = self._get_latest_price(symbol)
        logger.debug(f"Latest price for {symbol}: {latest_price}")

        indicators = self.technical_indicators.get_latest_indicators(symbol) if self.technical_indicators else None
        logger.debug(f"Technical indicators for {symbol}: {indicators}")

        if not latest_price:
            logger.warning(f"No price data available for {symbol}")
            return None

        # Validate signal with technical indicators
        if indicators:
            valid, confidence_boost = self._validate_with_technicals(prediction_type, indicators)
            logger.debug(f"Technical validation for {symbol}: valid={valid}, confidence_boost={confidence_boost}")

            if not valid:
                logger.info(f"Signal rejected by technical validation for {symbol}")
                return None

            # Boost confidence based on technical alignment
            confidence += confidence_boost
            confidence = min(confidence, 0.95)  # Cap confidence
            logger.debug(f"Adjusted confidence for {symbol}: {confidence:.2f}")

        # Check minimum confidence threshold
        if confidence < self.min_confidence:
            logger.info(f"Signal confidence too low for {symbol}: {confidence}")
            return None

        # Calculate risk management parameters
        stop_loss, take_profit = self._calculate_risk_params(symbol, prediction_type, latest_price, indicators)
        logger.debug(f"Risk params for {symbol}: SL={stop_loss}, TP={take_profit}")

        if not stop_loss or not take_profit:
            logger.warning(f"Could not calculate risk parameters for {symbol}")
            return None

        # Calculate risk-reward ratio
        if prediction_type == "LONG":
            risk = latest_price - stop_loss
            reward = take_profit - latest_price
        else:  # SHORT
            risk = stop_loss - latest_price
            reward = latest_price - take_profit

        risk_reward_ratio = reward / risk if risk > 0 else 0
        logger.debug(f"Risk-reward for {symbol}: {risk_reward_ratio:.2f}")

        # Minimum risk-reward ratio check
        if risk_reward_ratio < 1.5:
            logger.info(f"Risk-reward ratio too low for {symbol}: {risk_reward_ratio}")
            return None

        # Store signal in database
        conn = sqlite3.connect(os.path.join(self.db_path, 'signals.db'))
        cursor = conn.cursor()

        model_name = self.ai_model.get_best_model(symbol) if hasattr(self.ai_model, 'get_best_model') else 'default'
        logger.debug(f"Using model {model_name} for {symbol}")

        cursor.execute('''
        INSERT INTO signals
        (symbol, direction, entry_price, stop_loss, take_profit, risk_reward_ratio,
         confidence, model_name, feature_id, status, timestamp)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            symbol,
            prediction_type,
            latest_price,
            stop_loss,
            take_profit,
            risk_reward_ratio,
            confidence,
            model_name,
            feature_id,
            'PENDING',
            int(time.time())
        ))

        conn.commit()
        signal_id = cursor.lastrowid
        conn.close()

        self.signal_counter += 1

        logger.info(
            f"Generated {prediction_type} signal for {symbol} - ID: {signal_id}, Confidence: {confidence:.2f}, RR: {risk_reward_ratio:.2f}")

        signal = {
            'id': signal_id,
            'symbol': symbol,
            'direction': prediction_type,
            'entry_price': latest_price,
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'risk_reward': risk_reward_ratio,
            'confidence': confidence * 100,  # Convert to percentage for display
            'model_name': model_name
        }

        logger.info(f"Signal data: {json.dumps(signal)}")
        return signal

    def _validate_with_technicals(self, prediction_type, indicators):
        """Validate signal using technical indicators"""
        confidence_boost = 0
        validations_passed = 0
        total_validations = 0

        # RSI validation
        if 'rsi' in indicators:
            total_validations += 1
            if prediction_type == "LONG" and indicators['rsi'] < 70:
                # Not overbought
                validations_passed += 1
                if indicators['rsi'] < 30:
                    # Oversold - stronger signal for long
                    confidence_boost += 0.05
            elif prediction_type == "SHORT" and indicators['rsi'] > 30:
                # Not oversold
                validations_passed += 1
                if indicators['rsi'] > 70:
                    # Overbought - stronger signal for short
                    confidence_boost += 0.05

        # MACD validation
        if all(key in indicators for key in ['macd', 'macd_signal']):
            total_validations += 1
            if prediction_type == "LONG" and indicators['macd'] > indicators['macd_signal']:
                # MACD above signal line - bullish
                validations_passed += 1
                confidence_boost += 0.03
            elif prediction_type == "SHORT" and indicators['macd'] < indicators['macd_signal']:
                # MACD below signal line - bearish
                validations_passed += 1
                confidence_boost += 0.03

        # EMA crossover validation
        if 'ema_crossover_medium_long' in indicators:
            total_validations += 1
            if prediction_type == "LONG" and indicators['ema_crossover_medium_long'] == 1:
                # Bullish EMA crossover
                validations_passed += 1
                confidence_boost += 0.05
            elif prediction_type == "SHORT" and indicators['ema_crossover_medium_long'] == -1:
                # Bearish EMA crossover
                validations_passed += 1
                confidence_boost += 0.05

        # ADX validation - trend strength
        if 'adx' in indicators:
            total_validations += 1
            if indicators['adx'] > 25:  # Strong trend
                validations_passed += 1
                confidence_boost += 0.02

        # Require at least 50% of validations to pass
        valid = validations_passed / total_validations >= 0.5 if total_validations > 0 else False

        return valid, confidence_boost

    def _calculate_risk_params(self, symbol, direction, current_price, indicators=None):
        """Calculate stop loss and take profit levels"""
        # Get historical volatility to determine stop loss distance
        volatility_pct = self._get_volatility(symbol) or 2.0  # Default 2% if volatility data not available

        # Adjust based on indicators if available
        if indicators and 'adx' in indicators:
            # Higher ADX = stronger trend = tighter stop
            if indicators['adx'] > 30:
                volatility_pct *= 0.8
            elif indicators['adx'] < 20:
                volatility_pct *= 1.2

        # Calculate stop loss and take profit
        if direction == "LONG":
            stop_loss = current_price * (1 - volatility_pct / 100)
            take_profit = current_price * (1 + (volatility_pct * 2.5) / 100)  # 2.5x volatility for TP
        else:  # SHORT
            stop_loss = current_price * (1 + volatility_pct / 100)
            take_profit = current_price * (1 - (volatility_pct * 2.5) / 100)

        # Round to appropriate precision
        stop_loss = round(stop_loss, self._get_price_precision(symbol, current_price))
        take_profit = round(take_profit, self._get_price_precision(symbol, current_price))

        return stop_loss, take_profit

    def _get_volatility(self, symbol, days=1):
        """Calculate average volatility for a symbol"""
        conn = sqlite3.connect(os.path.join(self.db_path, 'market_data.db'))
        cursor = conn.cursor()

        # Get recent kline data
        cursor.execute('''
        SELECT high_price, low_price
        FROM kline_data
        WHERE symbol = ? AND timeframe = '1h'
        ORDER BY open_time DESC
        LIMIT ?
        ''', (symbol, 24 * days))  # 24 hours in a day

        rows = cursor.fetchall()
        conn.close()

        if not rows:
            return None

        # Calculate average volatility as (high-low)/low * 100
        volatility = [(high - low) / low * 100 for high, low in rows]

        return sum(volatility) / len(volatility)

    def _get_latest_price(self, symbol):
        """Get the latest price for a symbol"""
        conn = sqlite3.connect(os.path.join(self.db_path, 'market_data.db'))
        cursor = conn.cursor()

        cursor.execute('''
        SELECT price
        FROM market_data
        WHERE symbol = ?
        ORDER BY timestamp DESC
        LIMIT 1
        ''', (symbol,))

        row = cursor.fetchone()
        conn.close()

        return row[0] if row else None

    def _get_price_precision(self, symbol, price):
        """Determine appropriate price precision"""
        if price >= 1000:
            return 1  # 1 decimal place for high prices
        elif price >= 100:
            return 2  # 2 decimal places
        elif price >= 1:
            return 3  # 3 decimal places
        else:
            return 5  # 5 decimal places for low prices

    def update_signal_status(self, signal_id, status, price=None, profit_pct=None):
        """Update the status of a signal"""
        conn = sqlite3.connect(os.path.join(self.db_path, 'signals.db'))
        cursor = conn.cursor()

        update_fields = ['status']
        update_values = [status]

        if status in ['WIN', 'LOSS', 'CLOSED']:
            update_fields.append('closed_at')
            update_values.append(int(time.time() * 1000))

            if profit_pct is not None:
                update_fields.append('profit_pct')
                update_values.append(profit_pct)

        set_clause = ', '.join([f"{field} = ?" for field in update_fields])

        cursor.execute(f'''
        UPDATE signals
        SET {set_clause}
        WHERE id = ?
        ''', tuple(update_values + [signal_id]))

        conn.commit()
        conn.close()

        logger.info(f"Updated signal {signal_id} to status: {status}")

        # Update AI model with outcome for learning
        if status in ['WIN', 'LOSS'] and self.ai_model and hasattr(self.ai_model, 'update_prediction_outcome'):
            # Get signal details
            signal = self.get_signal(signal_id)
            if signal and 'feature_id' in signal:
                # Determine accuracy score
                accuracy = 0.9 if status == 'WIN' else 0.1

                # Update model learning
                self.ai_model.update_prediction_outcome(signal['feature_id'], status, accuracy)

    def confirm_signal(self, signal_id):
        """Mark signal as confirmed (entry executed)"""
        conn = sqlite3.connect(os.path.join(self.db_path, 'signals.db'))
        cursor = conn.cursor()

        cursor.execute('''
        UPDATE signals
        SET confirmed = 1, status = 'ACTIVE'
        WHERE id = ?
        ''', (signal_id,))

        conn.commit()
        conn.close()

        logger.info(f"Confirmed signal {signal_id}")

    def get_signal(self, signal_id):
        """Get signal details by ID"""
        conn = sqlite3.connect(os.path.join(self.db_path, 'signals.db'))
        cursor = conn.cursor()

        cursor.execute('''
        SELECT id, symbol, direction, entry_price, stop_loss, take_profit, 
               risk_reward_ratio, confidence, model_name, feature_id, status,
               confirmed, outcome, profit_pct, timestamp, closed_at
        FROM signals
        WHERE id = ?
        ''', (signal_id,))

        row = cursor.fetchone()
        conn.close()

        if not row:
            return None

        return {
            'id': row[0],
            'symbol': row[1],
            'direction': row[2],
            'entry_price': row[3],
            'stop_loss': row[4],
            'take_profit': row[5],
            'risk_reward_ratio': row[6],
            'confidence': row[7],
            'model_name': row[8],
            'feature_id': row[9],
            'status': row[10],
            'confirmed': bool(row[11]),
            'outcome': row[12],
            'profit_pct': row[13],
            'timestamp': row[14],
            'closed_at': row[15]
        }

    def get_active_signals(self):
        """Get all active signals"""
        conn = sqlite3.connect(os.path.join(self.db_path, 'signals.db'))
        cursor = conn.cursor()

        cursor.execute('''
        SELECT id, symbol, direction, entry_price, stop_loss, take_profit, 
               risk_reward_ratio, confidence, status, confirmed, timestamp
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
                'risk_reward_ratio': row[6],
                'confidence': row[7],
                'status': row[8],
                'confirmed': bool(row[9]),
                'timestamp': row[10]
            }
            for row in rows
        ]

    def get_signals_history(self, limit=100):
        """Get historical signals"""
        conn = sqlite3.connect(os.path.join(self.db_path, 'signals.db'))
        cursor = conn.cursor()

        cursor.execute('''
        SELECT id, symbol, direction, entry_price, stop_loss, take_profit,
               risk_reward_ratio, confidence, status, confirmed, outcome, profit_pct, timestamp, closed_at
        FROM signals
        WHERE status NOT IN ('PENDING', 'ACTIVE')
        ORDER BY timestamp DESC
        LIMIT ?
        ''', (limit,))

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
                'risk_reward_ratio': row[6],
                'confidence': row[7],
                'status': row[8],
                'confirmed': bool(row[9]),
                'outcome': row[10],
                'profit_pct': row[11],
                'timestamp': row[12],
                'closed_at': row[13]
            }
            for row in rows
        ]