import pandas as pd
from sklearn.preprocessing import StandardScaler

class DataPreprocessor:
    
    FEATURES_COLS = ["average_eyes_ear","mouth_openness_mar","smile_score"]
    
    def __init__(self):
        self.scaler = StandardScaler()
    
    def fit_transform(self, df):
        features = df[self.FEATURES_COLS].copy()
        normalized= self.scaler.fit_transform(features)
        return pd.DataFrame(normalized, columns=self.FEATURES_COLS, index=df.index)
    
    def transform(self, df):
        features = df[self.FEATURES_COLS].copy()
        normalized= self.scaler.transform(features)
        return pd.DataFrame(normalized, columns=self.FEATURES_COLS, index=df.index)