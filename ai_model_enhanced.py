import numpy as np
import pandas as pd
import sqlite3
import logging
import os
import pickle
import time
import json
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVR, SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

logger = logging.getLogger(__name__)


class EnhancedAIModel:
    def __init__(self, db_path='data'):
        self.db_path = db_path
        self.model_registry = {}  # Dictionary to store different models per symbol
        self.scalers = {}
        self.min_data_points = 500  # Minimum data points for production training
        self.min_data_points_initial = 50  # Reduced threshold for initial training
        self.training_frequency = 100
        self.data_counter = {}
        self.model_dir = os.path.join(db_path, 'models')
        self.signal_throttle = {}  # Store last signal time per symbol
        self.throttle_settings = {}  # Custom throttle settings per symbol
        self.default_throttle_hours = 24  # Default 24h throttle

        os.makedirs(self.model_dir, exist_ok=True)

        # Initialize performance tracking
        self.performance_metrics = {}

        # Load existing models if available
        self.load_models()

        logger.info("Enhanced AI model initialized with multi-model support")

    def load_models(self):
        """Load all trained models from disk"""
        for model_file in os.listdir(self.model_dir):
            if model_file.endswith('.pkl'):
                symbol = model_file.split('_')[0]
                model_path = os.path.join(self.model_dir, model_file)

                try:
                    with open(model_path, 'rb') as f:
                        model_data = pickle.load(f)

                    # Initialize registry for this symbol if needed
                    if symbol not in self.model_registry:
                        self.model_registry[symbol] = {}

                    # Load all models for this symbol
                    for model_name, model_info in model_data['models'].items():
                        self.model_registry[symbol][model_name] = model_info['model']

                    self.scalers[symbol] = model_data['scaler']
                    self.performance_metrics[symbol] = model_data.get('metrics', {})

                    # Load throttle settings if available
                    if 'throttle_hours' in model_data:
                        self.throttle_settings[symbol] = model_data['throttle_hours']

                    logger.info(f"Loaded models for {symbol}")
                except Exception as e:
                    logger.error(f"Error loading model for {symbol}: {e}")

    def save_models(self, symbol):
        """Save all trained models for a symbol to disk"""
        if symbol in self.model_registry and symbol in self.scalers:
            model_data = {
                'models': {},
                'scaler': self.scalers[symbol],
                'metrics': self.performance_metrics.get(symbol, {}),
                'throttle_hours': self.throttle_settings.get(symbol, self.default_throttle_hours),
                'updated_at': datetime.now().isoformat()
            }

            # Store all models for this symbol
            for model_name, model in self.model_registry[symbol].items():
                # Get performance metrics for this specific model
                model_metrics = self.performance_metrics.get(symbol, {}).get(model_name, {})

                model_data['models'][model_name] = {
                    'model': model,
                    'metrics': model_metrics
                }

            model_path = os.path.join(self.model_dir, f"{symbol}_model.pkl")

            with open(model_path, 'wb') as f:
                pickle.dump(model_data, f)

            logger.info(f"Saved models for {symbol}")

    def set_throttle(self, symbol, hours):
        """Configure signal throttle settings for a specific symbol"""
        self.throttle_settings[symbol] = hours
        logger.info(f"Set signal throttle for {symbol} to {hours} hours")

        # Save the updated settings
        if symbol in self.model_registry:
            self.save_models(symbol)

    def generate_signals(self, min_confidence=None):
        """Generate trading signals based on model predictions"""
        if min_confidence is None:
            min_confidence = 0.5  # Default threshold

        logger.info(f"Generating signals with min_confidence={min_confidence}")
        signals = []
        min_price = 0.5  # Minimum price filter

        # Connect to database
        conn = sqlite3.connect(os.path.join(self.db_path, 'ai_model.db'))
        cursor = conn.cursor()

        # Get latest feature IDs for each symbol
        cursor.execute('''
        SELECT symbol, MAX(id) as latest_id FROM model_features
        GROUP BY symbol
        ''')

        latest_features = cursor.fetchall()

        for symbol, feature_id in latest_features:
            # Skip if no models available for this symbol
            if symbol not in self.model_registry:
                continue

            # Get prediction for this symbol
            prediction_type, probability = self.predict(symbol, feature_id)

            # Get current price
            current_price = self._get_current_price(symbol)
            if not current_price:
                continue

            logger.debug(
                f"{symbol} - prediction: {prediction_type}, confidence: {probability:.4f}, price: {current_price}")

            # Generate signal if confidence exceeds threshold and price above minimum
            if probability > min_confidence and current_price > min_price and prediction_type != "NEUTRAL" and prediction_type != "THROTTLED":
                # Calculate take profit and stop loss levels
                if prediction_type == "LONG":
                    take_profit = current_price * 1.03  # 3% profit target
                    stop_loss = current_price * 0.98  # 2% stop loss
                else:  # SHORT
                    take_profit = current_price * 0.97  # 3% profit target
                    stop_loss = current_price * 1.02  # 2% stop loss

                # Create signal dict
                signal = {
                    'symbol': symbol,
                    'direction': prediction_type,
                    'entry_price': current_price,
                    'take_profit': take_profit,
                    'stop_loss': stop_loss,
                    'confidence': probability,
                    'timestamp': int(time.time() * 1000)
                }

                signals.append(signal)
                logger.info(f"Generated {prediction_type} signal for {symbol} with confidence {probability:.4f}")

        conn.close()
        return signals

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

    def can_generate_signal(self, symbol, confidence):
        """Check if signal can be generated based on throttle settings"""
        current_time = time.time()
        throttle_hours = self.throttle_settings.get(symbol, self.default_throttle_hours)

        # Check last signal time
        if symbol in self.signal_throttle:
            hours_since_last = (current_time - self.signal_throttle[symbol]) / 3600

            # Allow high confidence signals to bypass throttle
            if confidence > 90:
                return True

            # Enforce throttle
            if hours_since_last < throttle_hours:
                return False

        # Update last signal time
        self.signal_throttle[symbol] = current_time
        return True

    def get_training_data(self, symbol, lookback=20):
        """Get enhanced training data with more features and lookback periods"""
        conn = sqlite3.connect(os.path.join(self.db_path, 'ai_model.db'))
        cursor = conn.cursor()

        # Get all feature data for this symbol
        cursor.execute('''
        SELECT id, price_change_1m, price_change_5m, price_change_15m, price_change_1h,
               volume_change_1m, volume_change_5m, rsi_value, macd_histogram,
               ema_crossover, timestamp
        FROM model_features
        WHERE symbol = ?
        ORDER BY timestamp
        ''', (symbol,))

        feature_rows = cursor.fetchall()

        # Determine minimum data points based on whether we're in initial training
        min_points = self.min_data_points_initial if symbol not in self.model_registry else self.min_data_points

        if not feature_rows or len(feature_rows) < min_points:
            conn.close()
            return None, None, None

        # Get market data for future price changes (for labeling)
        conn_market = sqlite3.connect(os.path.join(self.db_path, 'market_data.db'))
        cursor_market = conn_market.cursor()

        cursor_market.execute('''
        SELECT price, timestamp FROM market_data
        WHERE symbol = ?
        ORDER BY timestamp
        ''', (symbol,))

        market_data = cursor_market.fetchall()
        conn_market.close()

        if not market_data or len(market_data) < len(feature_rows):
            conn.close()
            return None, None, None

        # Create a mapping of timestamps to prices for easier lookup
        price_map = {ts: price for price, ts in market_data}

        # Prepare features and labels
        feature_ids = []
        features = []
        labels = []
        timestamps = []

        for i in range(lookback, len(feature_rows)):
            # Get current feature row
            feature_id, *feature_values, timestamp = feature_rows[i]

            # Create a window of features (lookback period)
            window_features = []
            for j in range(i - lookback, i + 1):
                _, *window_values, _ = feature_rows[j]
                window_features.extend(window_values)

            # Find future price for labeling (multi-timeframe)
            # Check 1h, 4h, and 24h future prices
            future_timestamps = [
                timestamp + (3600 * 1000),  # 1 hour
                timestamp + (4 * 3600 * 1000),  # 4 hours
                timestamp + (24 * 3600 * 1000)  # 24 hours
            ]

            future_prices = []
            for future_ts in future_timestamps:
                # Find the closest timestamp in our price data
                closest_timestamps = sorted(price_map.keys(), key=lambda x: abs(x - future_ts))
                if not closest_timestamps:
                    future_prices.append(None)
                    continue

                closest_timestamp = closest_timestamps[0]

                # Skip if the closest timestamp is too far from our target
                if abs(closest_timestamp - future_ts) > (15 * 60 * 1000):  # 15 minutes
                    future_prices.append(None)
                    continue

                future_prices.append(price_map[closest_timestamp])

            # Skip if any future price is missing
            if None in future_prices:
                continue

            current_price = price_map.get(timestamp, None)
            if not current_price:
                continue

            # Calculate price change percentages for each timeframe
            price_changes = [((fp - current_price) / current_price) * 100 for fp in future_prices]

            # Create label based on price change threshold (focus on 4h timeframe)
            threshold = 3.0  # Configurable

            # Primary label based on 4h timeframe (index 1)
            if price_changes[1] > threshold:
                label = 1  # Long signal
            elif price_changes[1] < -threshold:
                label = -1  # Short signal
            else:
                label = 0  # Neutral

            # Add additional features for price momentum across timeframes
            window_features.extend(price_changes)

            feature_ids.append(feature_id)
            features.append(window_features)
            labels.append(label)
            timestamps.append(timestamp)

        conn.close()

        if not features:
            return None, None, None

        return np.array(features), np.array(labels), feature_ids

    def train_multiple_models(self, symbol):
        """Train multiple model types for a symbol and evaluate each one"""
        features, labels, feature_ids = self.get_training_data(symbol)

        # Determine minimum data points based on whether we're in initial training
        min_points = self.min_data_points_initial if symbol not in self.model_registry else self.min_data_points

        if features is None or len(features) < min_points:
            logger.info(f"Not enough data to train models for {symbol}")
            return False

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            features, labels, test_size=0.2, random_state=42, stratify=labels
        )

        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Initialize model registry for this symbol if needed
        if symbol not in self.model_registry:
            self.model_registry[symbol] = {}

        if symbol not in self.performance_metrics:
            self.performance_metrics[symbol] = {}

        # Define the models to train
        models_to_train = {
            'gradient_boosting': GradientBoostingClassifier(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=5,
                random_state=42
            ),
            'random_forest': RandomForestClassifier(
                n_estimators=100,
                max_depth=5,
                random_state=42
            ),
            'svc': SVC(
                probability=True,
                random_state=42
            )
        }

        # Train and evaluate each model
        for model_name, model in models_to_train.items():
            # Train the model
            model.fit(X_train_scaled, y_train)

            # Evaluate on test set
            y_pred = model.predict(X_test_scaled)

            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
            recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
            f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
            conf_matrix = confusion_matrix(y_test, y_pred)

            # Calculate profit factor (custom metric)
            # This assumes 1 is LONG, -1 is SHORT, 0 is NEUTRAL
            profit_factor = self._calculate_profit_factor(y_test, y_pred)

            # Calculate composite score that prioritizes profitability
            composite_score = accuracy * 0.3 + precision * 0.2 + profit_factor * 0.5

            # Store model and metrics
            self.model_registry[symbol][model_name] = model
            self.performance_metrics[symbol][model_name] = {
                'accuracy': float(accuracy),
                'precision': float(precision),
                'recall': float(recall),
                'f1_score': float(f1),
                'profit_factor': float(profit_factor),
                'composite_score': float(composite_score),
                'training_samples': int(len(X_train)),
                'last_trained': datetime.now().isoformat()
            }

            logger.info(f"Trained {model_name} for {symbol} - Score: {composite_score:.4f}")

        # Save scaler
        self.scalers[symbol] = scaler

        # Save models to disk
        self.save_models(symbol)

        return True

    def _calculate_profit_factor(self, y_true, y_pred):
        """Calculate a simplified profit factor based on predictions vs actual"""
        total_gain = 0
        total_loss = 0

        for true_val, pred_val in zip(y_true, y_pred):
            # For long signals (pred=1)
            if pred_val == 1:
                if true_val == 1:  # Correct long prediction
                    total_gain += 1
                elif true_val == -1:  # Wrong direction
                    total_loss += 2  # Higher penalty for wrong direction
                else:  # Neutral (small loss)
                    total_loss += 0.5

            # For short signals (pred=-1)
            elif pred_val == -1:
                if true_val == -1:  # Correct short prediction
                    total_gain += 1
                elif true_val == 1:  # Wrong direction
                    total_loss += 2  # Higher penalty for wrong direction
                else:  # Neutral (small loss)
                    total_loss += 0.5

        # Avoid division by zero
        if total_loss == 0:
            return total_gain if total_gain > 0 else 1

        return total_gain / total_loss if total_loss > 0 else 0

    def get_best_model(self, symbol):
        """Get the best performing model for a symbol based on composite score"""
        if symbol not in self.performance_metrics:
            return None

        metrics = self.performance_metrics[symbol]
        best_model_name = max(metrics, key=lambda m: metrics[m]['composite_score'])

        return best_model_name

    def predict(self, symbol, features_id):
        """Make predictions using the best model for this symbol"""
        # Get feature data
        conn = sqlite3.connect(os.path.join(self.db_path, 'ai_model.db'))
        cursor = conn.cursor()

        cursor.execute('''
        SELECT symbol, price_change_1m, price_change_5m, price_change_15m, price_change_1h,
               volume_change_1m, volume_change_5m, rsi_value, macd_histogram,
               ema_crossover, timestamp
        FROM model_features
        WHERE id = ?
        ''', (features_id,))

        row = cursor.fetchone()

        if not row:
            conn.close()
            return None, 0.0

        symbol = row[0]
        feature_values = row[1:10]  # All features except timestamp
        timestamp = row[10]

        # Check if we need to train models
        if symbol not in self.data_counter:
            self.data_counter[symbol] = 0

        self.data_counter[symbol] += 1

        # Train if needed
        if (symbol not in self.model_registry or
                self.data_counter[symbol] % self.training_frequency == 0):
            self.train_multiple_models(symbol)

        # If we still don't have models, return neutral prediction
        if symbol not in self.model_registry or not self.model_registry[symbol]:
            prediction_type = "NEUTRAL"
            confidence = 0.0

            # Save prediction to database
            cursor.execute('''
            INSERT INTO model_predictions
            (symbol, prediction_type, confidence_score, features_id, timestamp)
            VALUES (?, ?, ?, ?, ?)
            ''', (symbol, prediction_type, confidence, features_id, timestamp))

            conn.commit()
            conn.close()

            return prediction_type, confidence

        # Get lookback features
        cursor.execute('''
        SELECT price_change_1m, price_change_5m, price_change_15m, price_change_1h,
               volume_change_1m, volume_change_5m, rsi_value, macd_histogram,
               ema_crossover
        FROM model_features
        WHERE symbol = ? AND timestamp < ?
        ORDER BY timestamp DESC
        LIMIT 20
        ''', (symbol, timestamp))

        lookback_rows = cursor.fetchall()

        # Prepare input features
        window_features = []

        if len(lookback_rows) >= 20:
            # Add lookback features (oldest to newest)
            for row in reversed(lookback_rows):
                window_features.extend(row)

            # Add current features
            window_features.extend(feature_values)
        else:
            # Use only current features if not enough history
            window_features = feature_values

        # Get best model for this symbol
        best_model_name = self.get_best_model(symbol)

        if not best_model_name:
            # Use first available model if no best model determined
            best_model_name = next(iter(self.model_registry[symbol]))

        best_model = self.model_registry[symbol][best_model_name]

        # Format input for prediction
        X = np.array(window_features).reshape(1, -1)
        X_scaled = self.scalers[symbol].transform(X)

        # Get prediction
        prediction = best_model.predict(X_scaled)[0]

        # Get probabilities
        if hasattr(best_model, 'predict_proba'):
            probabilities = best_model.predict_proba(X_scaled)[0]

            # Map prediction to type and get confidence
            if prediction == -1:  # Short
                prediction_type = "SHORT"
                confidence = probabilities[0] if len(probabilities) > 0 else 0.5
            elif prediction == 1:  # Long
                prediction_type = "LONG"
                confidence = probabilities[2] if len(probabilities) > 2 else 0.5
            else:  # Neutral
                prediction_type = "NEUTRAL"
                confidence = probabilities[1] if len(probabilities) > 1 else 0.5
        else:
            # For models without predict_proba, use a fixed confidence
            if prediction == -1:
                prediction_type = "SHORT"
                confidence = 0.7  # Default confidence
            elif prediction == 1:
                prediction_type = "LONG"
                confidence = 0.7  # Default confidence
            else:
                prediction_type = "NEUTRAL"
                confidence = 0.5  # Default confidence

        # Round confidence to 2 decimal places
        confidence = round(float(confidence), 2)

        # Save prediction to database with model name
        cursor.execute('''
        INSERT INTO model_predictions
        (symbol, prediction_type, confidence_score, features_id, model_name, timestamp)
        VALUES (?, ?, ?, ?, ?, ?)
        ''', (symbol, prediction_type, confidence, features_id, best_model_name, timestamp))

        conn.commit()
        conn.close()

        # Check if we can generate signal based on throttle
        can_signal = self.can_generate_signal(symbol, confidence * 100)

        if not can_signal:
            return "THROTTLED", 0.0

        return prediction_type, confidence

    def update_prediction_outcome(self, prediction_id, actual_outcome, accuracy):
        """Update the prediction with actual outcome and accuracy for model improvement"""
        conn = sqlite3.connect(os.path.join(self.db_path, 'ai_model.db'))
        cursor = conn.cursor()

        # Get the prediction details
        cursor.execute('''
        SELECT symbol, prediction_type, model_name, timestamp
        FROM model_predictions
        WHERE id = ?
        ''', (prediction_id,))

        row = cursor.fetchone()

        if not row:
            conn.close()
            return

        symbol, prediction_type, model_name, timestamp = row

        # Update prediction outcome
        cursor.execute('''
        UPDATE model_predictions
        SET actual_outcome = ?, accuracy = ?
        WHERE id = ?
        ''', (actual_outcome, accuracy, prediction_id))

        conn.commit()
        conn.close()

        # Check if we need to retrain based on outcome
        if symbol in self.model_registry and accuracy < 0.5:
            # Bad prediction - schedule faster retraining
            self.data_counter[symbol] = self.training_frequency - 5

        logger.info(f"Updated prediction {prediction_id} with outcome: {actual_outcome}, accuracy: {accuracy}")

    def get_model_performance(self, symbol):
        """Get performance metrics for a specific symbol"""
        if symbol in self.performance_metrics:
            return self.performance_metrics[symbol]
        return None

    def get_all_performance_metrics(self):
        """Get performance metrics for all models"""
        return self.performance_metrics

    def auto_label_outcomes(self):
        """Automatically label trade outcomes based on price movements"""
        conn = sqlite3.connect(os.path.join(self.db_path, 'ai_model.db'))
        cursor = conn.cursor()

        # Get predictions that don't have outcomes yet
        cursor.execute('''
        SELECT id, symbol, prediction_type, timestamp
        FROM model_predictions
        WHERE actual_outcome IS NULL AND timestamp < ?
        ''', (int(time.time() * 1000) - (24 * 3600 * 1000),))  # Older than 24 hours

        predictions = cursor.fetchall()

        # Get market data for outcome labeling
        conn_market = sqlite3.connect(os.path.join(self.db_path, 'market_data.db'))
        cursor_market = conn_market.cursor()

        for pred_id, symbol, prediction_type, timestamp in predictions:
            # Get future price data
            cursor_market.execute('''
            SELECT price, timestamp
            FROM market_data
            WHERE symbol = ? AND timestamp > ?
            ORDER BY timestamp
            LIMIT 24
            ''', (symbol, timestamp))

            future_prices = cursor_market.fetchall()

            if not future_prices:
                continue

            # Get entry price
            cursor_market.execute('''
            SELECT price
            FROM market_data
            WHERE symbol = ? AND timestamp <= ?
            ORDER BY timestamp DESC
            LIMIT 1
            ''', (symbol, timestamp))

            entry_row = cursor_market.fetchone()

            if not entry_row:
                continue

            entry_price = entry_row[0]

            # Calculate outcome based on price movements
            if prediction_type == "LONG":
                # For long signals, we want prices to go up
                best_price = max([price for price, _ in future_prices])
                worst_price = min([price for price, _ in future_prices])

                # Calculate percentages
                best_pct = ((best_price - entry_price) / entry_price) * 100
                worst_pct = ((worst_price - entry_price) / entry_price) * 100

                # Determine outcome
                if best_pct >= 3.0:  # Hit take profit
                    outcome = "WIN"
                    accuracy = min(1.0, best_pct / 5.0)  # Scale based on profit target
                elif worst_pct <= -2.0:  # Hit stop loss
                    outcome = "LOSS"
                    accuracy = 0.0
                else:  # Sideways movement
                    outcome = "NEUTRAL"
                    accuracy = 0.5

            elif prediction_type == "SHORT":
                # For short signals, we want prices to go down
                best_price = min([price for price, _ in future_prices])
                worst_price = max([price for price, _ in future_prices])

                # Calculate percentages (inverted for shorts)
                best_pct = ((entry_price - best_price) / entry_price) * 100
                worst_pct = ((entry_price - worst_price) / entry_price) * 100

                # Determine outcome
                if best_pct >= 3.0:  # Hit take profit
                    outcome = "WIN"
                    accuracy = min(1.0, best_pct / 5.0)  # Scale based on profit target
                elif worst_pct <= -2.0:  # Hit stop loss
                    outcome = "LOSS"
                    accuracy = 0.0
                else:  # Sideways movement
                    outcome = "NEUTRAL"
                    accuracy = 0.5
            else:
                # For neutral signals
                max_move_pct = max([abs(((price - entry_price) / entry_price) * 100) for price, _ in future_prices])

                if max_move_pct < 1.5:  # If price stayed relatively stable
                    outcome = "WIN"
                    accuracy = 0.8
                else:
                    outcome = "LOSS"  # Should have predicted a direction
                    accuracy = 0.3

            # Update prediction with outcome
            cursor.execute('''
            UPDATE model_predictions
            SET actual_outcome = ?, accuracy = ?
            WHERE id = ?
            ''', (outcome, accuracy, pred_id))

        conn.commit()
        conn.close()
        conn_market.close()

        logger.info(f"Auto-labeled {len(predictions)} prediction outcomes")

        # Trigger retraining if enough new labels
        if len(predictions) >= self.training_frequency:
            symbols = {symbol for _, symbol, _, _ in predictions}
            for symbol in symbols:
                if symbol in self.model_registry:
                    self.train_multiple_models(symbol)