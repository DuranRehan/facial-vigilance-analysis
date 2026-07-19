
class SmileFeatures:
    
    def __init__(self):
        self.SMILE_LEFT_KEY = "mouthSmileLeft"
        self.SMILE_RIGHT_KEY = "mouthSmileRight"
    
    def extract(self,blendshapes):
        smile_left = blendshapes.get(self.SMILE_LEFT_KEY, 0.0)
        smile_right = blendshapes.get(self.SMILE_RIGHT_KEY, 0.0)
        average_smile = (smile_left + smile_right) / 2.0
        
        return {
            "smile_left": smile_left,
            "smile_right": smile_right,
            "smile_score": average_smile
        }