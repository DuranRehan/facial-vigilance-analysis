from src.utils.linear_slope import linear_slope


class TrendAnalyzer:

    @staticmethod
    def calculate_trend(anomaly_scores):
        return linear_slope(anomaly_scores)
    
    
