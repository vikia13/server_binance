import logging
import time

# Configure logging
logger = logging.getLogger(__name__)

class PerformanceEvaluator:
    def __init__(self, database):
        self.database = database
        logger.info("Performance evaluator initialized")

    def evaluate_predictions(self):
        """Evaluate prediction performance"""
        try:
            # Get predictions with actual outcomes
            predictions = self.database.get_evaluated_predictions()

            if not predictions or len(predictions) == 0:
                logger.info("No predictions found for evaluation")
                return None

            # Count correct predictions
            total_predictions = len(predictions)
            correct_predictions = sum(1 for p in predictions if p['is_correct'])

            # Calculate overall accuracy
            overall_accuracy = (correct_predictions / total_predictions) * 100 if total_predictions > 0 else 0

            # Separate by prediction type
            long_predictions = [p for p in predictions if p['prediction_type'] == 'LONG']
            short_predictions = [p for p in predictions if p['prediction_type'] == 'SHORT']

            # Calculate accuracy by prediction type
            long_accuracy = (sum(1 for p in long_predictions if p['is_correct']) / len(long_predictions) * 100) if len(long_predictions) > 0 else 0
            short_accuracy = (sum(1 for p in short_predictions if p['is_correct']) / len(short_predictions) * 100) if len(short_predictions) > 0 else 0

            # Calculate confidence correlation
            high_confidence = [p for p in predictions if p['confidence_score'] >= 0.75]
            medium_confidence = [p for p in predictions if 0.5 <= p['confidence_score'] < 0.75]
            low_confidence = [p for p in predictions if p['confidence_score'] < 0.5]

            high_accuracy = (sum(1 for p in high_confidence if p['is_correct']) / len(high_confidence) * 100) if len(high_confidence) > 0 else 0
            medium_accuracy = (sum(1 for p in medium_confidence if p['is_correct']) / len(medium_confidence) * 100) if len(medium_confidence) > 0 else 0
            low_accuracy = (sum(1 for p in low_confidence if p['is_correct']) / len(low_confidence) * 100) if len(low_confidence) > 0 else 0

            # Return performance report
            return {
                'overall_accuracy': overall_accuracy,
                'total_predictions': total_predictions,
                'correct_predictions': correct_predictions,
                'long_accuracy': long_accuracy,
                'short_accuracy': short_accuracy,
                'high_confidence_accuracy': high_accuracy,
                'medium_confidence_accuracy': medium_accuracy,
                'low_confidence_accuracy': low_accuracy,
                'timestamp': int(time.time())
            }

        except Exception as e:
            logger.error(f"Error evaluating predictions: {e}")
            return None