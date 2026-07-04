from src.utils.euclidean_distance import euclidean_distance

class EyeFeatures: 

    # Based on EAR (Eye Aspect Ratio) formula
    def __eye_openness_ratio(self,p1,p2,p3,p4,p5,p6):
        vertical_1 =  euclidean_distance(p2,p6)
        vertical_2 = euclidean_distance(p3,p5)
        horizontal = euclidean_distance(p1,p4)
        
        return (vertical_1 + vertical_2) / (2 * horizontal)
    
    def eye_openness_average(self,left_P1,left_P2,left_P3,left_P4,left_P5,left_P6,right_P1,right_P2,right_P3,right_P4,right_P5,right_P6):
        left_ratio = self.__eye_openness_ratio(left_P1,left_P2,left_P3,left_P4,left_P5,left_P6)
        right_ratio = self.__eye_openness_ratio(right_P1,right_P2,right_P3,right_P4,right_P5,right_P6)
        
        return (left_ratio + right_ratio) / 2