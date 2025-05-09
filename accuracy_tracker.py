import logging
import time

# Configure logging
logger = logging.getLogger(__name__)

class AccuracyTracker:
    def __init__(self, database, ai_model):
        self.database = database
        self.ai_model = ai_model
        self.last_adjustment = int(time.time())
        self.min_predictions_for_adjustment = 20  # Minimum number of predictions before adjustment
        logger.info("Accuracy tracker initialized")

    def track_prediction_outcomes(self):
        """Track outcomes of previous predictions"""
        try:
            # Get open predictions without outcomes
            open_predictions = self.database.get_open_predictions()

            if not open_predictions or len(open_predictions) == 0:
                logger.info("No predictions to track outcomes for")
                return

            tracked_count = 0

            for prediction in open_predictions:
                # Get current price for the symbol
                current_price = self.database.get_latest_price(prediction['symbol'])

                if not current_price:
                    continue

                # Calculate time elapsed
                elapsed_time = int(time.time()) - prediction['timestamp'] // 1000

                # Only track outcome if sufficient time has passed
                if elapsed_time < 3600:  # At least 1 hour
                    continue

                # Determine outcome based on prediction type and price movement
                if prediction['prediction_type'] == 'LONG':
                    # For LONG predictions, price should go up
                    price_change_pct = (current_price - prediction['entry_price']) / prediction['entry_price'] * 100
                    if price_change_pct >= 1.0:  # Threshold: 1% increase
                        outcome = 'SUCCESS'
                        is_correct = True
                    elif price_change_pct <= -1.0:  # Threshold: 1% decrease
                        outcome = 'FAILED'
                        is_correct = False
                    else:
                        outcome = 'NEUTRAL'
                        is_correct = False

                elif prediction['prediction_type'] == 'SHORT':
                    # For SHORT predictions, price should go down
                    price_change_pct = (prediction['entry_price'] - current_price) / prediction['entry_price'] * 100
                    if price_change_pct >= 1.0:  # Threshold: 1% decrease
                        outcome = 'SUCCESS'
                        is_correct = True
                    elif price_change_pct <= -1.0:  # Threshold: 1% increase
                        outcome = 'FAILED'
                        is_correct = False
                    else:
                        outcome = 'NEUTRAL'
                        is_correct = False
                else:
                    continue  # Skip unknown prediction types

                # Update prediction with outcome
                self.database.update_prediction_outcome(
                    prediction['id'],
                    outcome,
                    is_correct,
                    price_change_pct
                )
                tracked_count += 1

            logger.info(f"Tracked outcomes for {tracked_count} predictions")

        except Exception as e:
            logger.error(f"Error tracking prediction outcomes: {e}")

    def adjust_model_parameters(self):
        """Adjust model parameters based on recent performance"""
        try:
            # Only adjust if enough time has passed since last adjustment
            current_time = int(time.time())
            if current_time - self.last_adjustment < 86400:  # 24 hours
                return

            # Get performance metrics
            performance = self.database.get_performance_metrics()

            if not performance:
                logger.info("Not enough data for performance analysis")
                return

            # Check if we have enough predictions
            if performance.get('total_predictions', 0) < self.min_predictions_for_adjustment:
                logger.info("Not enough predictions for model adjustment")
                return

            # Adjust model parameters based on performance
            adjusted = self.ai_model.adjust_parameters(performance)

            if adjusted:
                logger.info("Model parameters adjusted based on performance")
                self.last_adjustment = current_time

        except Exception as e:
            logger.error(f"Error adjusting model parameters: {e}")