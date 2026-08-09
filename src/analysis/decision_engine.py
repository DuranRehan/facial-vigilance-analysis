
class DecisionEngine:
    
    def __init__(self,vigilance_count=3,high_count=5,negative_trend_threshold=-0.02,strong_negative_trend_threshold=-0.05,intensity_threshold=-0.08,strong_intensity_threshold=-0.15):
        self.vigilance_count = vigilance_count
        self.high_count = high_count
        self.negative_trend_threshold = negative_trend_threshold
        self.strong_negative_trend_threshold = strong_negative_trend_threshold
        self.intensity_threshold = intensity_threshold
        self.strong_intensity_threshold = strong_intensity_threshold
    
    def evaluate(self, window_analysis):
        anomaly_count = window_analysis["anomaly_count"]
        intensity = window_analysis["intensity"]
        trend = window_analysis["trend"]

        if (anomaly_count >= self.high_count and intensity <= self.strong_intensity_threshold and trend <= self.strong_negative_trend_threshold):
            return {
                "risk_level": "high",
                "reason": "Frequent, strong anomalies with a degrading trend."
            }

        if (anomaly_count >= self.vigilance_count and intensity <= self.intensity_threshold and trend <= self.negative_trend_threshold):
            return {
                "risk_level": "vigilance",
                "reason": "Repeated anomalies with a degrading trend."
            }

        return {
            "risk_level": "normal",
            "reason": "No persistent degrading pattern."
        }