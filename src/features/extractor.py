
from src.features.smile import SmileFeatures
from src.features.eyes import EyeFeatures
from src.features.mouth import MouthFeatures



class FeatureExtractor:
    
    def __init__(self):
        self.eyeFeatures = EyeFeatures()
        self.mouthFeatures = MouthFeatures()
        self.smileFeatures = SmileFeatures()
        
    def extract(self,landmarks,blendshapes):
        features = {}
        features.update(self.eyeFeatures.extract(landmarks))
        features.update(self.mouthFeatures.extract(landmarks))
        features.update(self.smileFeatures.extract(blendshapes))
        return features





