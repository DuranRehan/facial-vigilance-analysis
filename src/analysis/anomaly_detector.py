import pandas as pd
from sklearn.ensemble import IsolationForest

class AnomalyDetector:
    def __init__(self, threshold: float = 0.1, random_state: int = 42):
        self.fitted = False
        self.model = IsolationForest(
            contamination=threshold, random_state=random_state
        )


    def fit(self, df_scaled: pd.DataFrame) -> None:
        self.model.fit(df_scaled)
        self.fitted = True

    def predict_anomaly_score(self, df_scaled: pd.DataFrame) -> pd.DataFrame:
        if not self.fitted:
            raise ValueError("The model must be fitted before predicting anomaly scores.")
        
        results = df_scaled.copy()
        results["anomaly_score"] = self.model.decision_function(df_scaled)
        results["is_anomaly"] = self.model.predict(df_scaled)
        return results
    
    
