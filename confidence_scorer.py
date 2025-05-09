import logging

# Configure logging
logger = logging.getLogger(__name__)

class AdvancedConfidenceScorer:
    def __init__(self):
        self.confidence_threshold = 0.7
        self.min_indicators_agreement = 3
        logger.info("Advanced confidence scorer initialized")

    def calculate_confidence(self, indicators):
        """Calculate confidence score based on multiple indicators"""
        try:
            # Extract indicator values
            rsi = indicators.get('rsi', 50)
            macd_diff = indicators.get('macd_diff', 0)
            adx = indicators.get('adx', 0)
            ema_crossover = indicators.get('ema_crossover', 0)
            price_change = indicators.get('price_change', 0)
            stoch_k = indicators.get('stoch_k', 50)

            # Count agreeing indicators for bullish signal
            bullish_count = 0
            if rsi > 50:
                bullish_count += 1
            if macd_diff > 0:
                bullish_count += 1
            if adx > 25:
                bullish_count += 1
            if ema_crossover > 0:
                bullish_count += 1
            if price_change > 1:
                bullish_count += 1
            if stoch_k > 50:
                bullish_count += 1

            # Count agreeing indicators for bearish signal
            bearish_count = 0
            if rsi < 50:
                bearish_count += 1
            if macd_diff < 0:
                bearish_count += 1
            if adx > 25:
                bearish_count += 1
            if ema_crossover < 0:
                bearish_count += 1
            if price_change < -1:
                bearish_count += 1
            if stoch_k < 50:
                bearish_count += 1

            # Calculate base confidence score
            if indicators.get('trend') == 'LONG':
                agreement_ratio = bullish_count / 6
            else:
                agreement_ratio = bearish_count / 6

            # Adjust based on ADX (trend strength)
            adx_factor = min(adx / 100, 1)  # Normalize to [0,1]

            # Calculate final confidence score
            confidence = agreement_ratio * (0.7 + 0.3 * adx_factor)

            # Ensure confidence is in [0,1] range
            confidence = max(0, min(1, confidence))

            return confidence

        except Exception as e:
            logger.error(f"Error calculating confidence: {e}")
            return 0.5  # Default to medium confidence on error

    def is_signal_reliable(self, confidence_score):
        """Determine if a signal is reliable based on confidence score"""
        return confidence_score >= self.confidence_threshold