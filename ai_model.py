import logging
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import joblib
import os
import time

# Configure logging
logger = logging.getLogger(__name__)

class AIModelWrapper:
    def __init__(self):
        self.model = None
        self.scaler = None
        self.features = [
            'price_change_1m', 'price_change_5m', 'price_change_15m',
            'volume_change_1m', 'volume_change_5m',
            'rsi_value', 'macd_histogram', 'ema_crossover'
        ]
        self.model_path = os.path.join('models', 'price_model.pkl')
        self.scaler_path = os.path.join('models', 'price_scaler.pkl')

        # Create models directory if it doesn't exist
        os.makedirs('models', exist_ok=True)

        # Try to load existing model or create a new one
        self.load_model()

        logger.info("AI model wrapper initialized")

    def load_model(self):
        """Load model from disk if exists, otherwise initialize a new one"""
        try:
            if os.path.exists(self.model_path) and os.path.exists(self.scaler_path):
                self.model = joblib.load(self.model_path)
                self.scaler = joblib.load(self.scaler_path)
                logger.info("Model loaded from disk")
            else:
                # Initialize a new model with default parameters
                self.model = RandomForestClassifier(
                    n_estimators=100,
                    max_depth=5,
                    random_state=42
                )
                self.scaler = StandardScaler()
                logger.info("No existing model found, using default parameters")
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            # Initialize a new model with default parameters
            self.model = RandomForestClassifier(
                n_estimators=100,
                max_depth=5,
                random_state=42
            )
            self.scaler = StandardScaler()

    def save_model(self):
        """Save model to disk"""
        try:
            joblib.dump(self.model, self.model_path)
            joblib.dump(self.scaler, self.scaler_path)
            logger.info("Model saved to disk")
            return True
        except Exception as e:
            logger.error(f"Error saving model: {e}")
            return False

    def preprocess_data(self, data):
        """Preprocess data for model input"""
        try:
            # Extract features
            X = data[self.features].values

            # Scale features
            if self.scaler is not None:
                X = self.scaler.transform(X)

            return X
        except Exception as e:
            logger.error(f"Error preprocessing data: {e}")
            return None

    def train(self, X, y):
        """Train the model on new data"""
        try:
            # Scale features if not already scaled
            X_scaled = self.scaler.fit_transform(X)

            # Train the model
            self.model.fit(X_scaled, y)

            # Save the model
            self.save_model()

            logger.info("Model trained successfully")
            return True
        except Exception as e:
            logger.error(f"Error training model: {e}")
            return False

    def predict(self, data):
        """Make predictions with the model"""
        try:
            # Preprocess data
            X = self.preprocess_data(data)
            if X is None:
                return None

            # Make predictions
            probabilities = self.model.predict_proba(X)

            # Get the class with highest probability
            predictions = self.model.classes_[np.argmax(probabilities, axis=1)]

            # Return predictions and confidence scores
            confidence_scores = np.max(probabilities, axis=1)

            prediction_results = []
            for i, pred in enumerate(predictions):
                prediction_results.append({
                    'prediction': pred,
                    'confidence': confidence_scores[i]
                })

            return prediction_results
        except Exception as e:
            logger.error(f"Error making predictions: {e}")
            return None

    def adjust_parameters(self, performance_metrics):
        """Adjust model parameters based on performance metrics"""
        try:
            # Simple parameter adjustment based on accuracy
            accuracy = performance_metrics.get('overall_accuracy', 0)

            # If accuracy is below threshold, increase model complexity
            if accuracy < 55:  # Less than 55% accuracy
                current_n_estimators = self.model.n_estimators
                new_n_estimators = min(current_n_estimators + 20, 200)  # Increase but cap at 200

                current_depth = self.model.max_depth
                new_depth = min(current_depth + 1, 10)  # Increase but cap at 10

                # Create new model with adjusted parameters
                self.model = RandomForestClassifier(
                    n_estimators=new_n_estimators,
                    max_depth=new_depth,
                    random_state=42
                )

                logger.info(f"Adjusted model parameters: n_estimators={new_n_estimators}, max_depth={new_depth}")
                return True

            return False
        except Exception as e:
            logger.error(f"Error adjusting model parameters: {e}")
            return False