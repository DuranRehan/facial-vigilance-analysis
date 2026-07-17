
from src.features.eyes import EyeFeatures
from src.features.mouth import MouthFeatures



class FeatureExtractor:
    
    def __init__(self):
        self.eyeFeatures = EyeFeatures()
        self.mouthFeatures = MouthFeatures()
        
    def extract(self,landmarks):
        features = {}
        features.update(self.eyeFeatures.extract(landmarks))
        features.update(self.mouthFeatures.extract(landmarks))
        return features





