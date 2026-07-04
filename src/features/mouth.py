from src.utils.euclidean_distance import euclidean_distance

class MouthFeatures:
    
    # Based on MAR (Mouth Aspect Ratio) formula
    def mouth_openness_ratio(self,p1,p2,p3,p4,p5,p6):
        vertical_1 =  euclidean_distance(p2,p6)
        vertical_2 = euclidean_distance(p3,p5)
        horizontal = euclidean_distance(p1,p4)
        
        return (vertical_1 + vertical_2) / (2 * horizontal)
    