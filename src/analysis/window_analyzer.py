import pandas as pd

from .trend_analyzer import TrendAnalyzer


class WindowAnalyzer:
    def __init__(self, window_size=7):
        self.window_size = window_size

    def get_current_window(self, df: pd.DataFrame) -> pd.DataFrame:
        return df.tail(self.window_size)
    
    
    def count_anomalies(self, window: pd.DataFrame) -> int:
        return (window["is_anomaly"] == -1).sum()
    
    def calculate_intensity(self, window):
        anomaly_scores = window["anomaly_score"]
        intensity = anomaly_scores[window["is_anomaly"] == -1].mean() 
        return intensity
    
    
    def analyze_window(self, window: pd.DataFrame) -> dict:
        count_anomalies = self.count_anomalies(window)
        intensity = self.calculate_intensity(window)

        trend_analyzer = TrendAnalyzer()
        trend = trend_analyzer.calculate_trend(window["anomaly_score"].tolist())
        
        return {
            "anomaly_count": count_anomalies,
            "intensity": intensity,
            "trend": trend
        }