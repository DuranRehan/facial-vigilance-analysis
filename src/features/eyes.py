from src.utils.euclidean_distance import euclidean_distance

class EyeFeatures: 
    
    LEFT_EYE_INDICES = [33, 160, 158, 133, 153, 144]
    RIGHT_EYE_INDICES = [263, 385, 387, 362, 373, 380]
    
    def extract(self,landmarks):
        
        
        left_eye = self.__eye_openness_ratio(
           *[landmarks[k] for k in self.LEFT_EYE_INDICES]
        )

        right_eye = self.__eye_openness_ratio(
              *[landmarks[k] for k in self.RIGHT_EYE_INDICES]
        )
      
      
      
        average = (left_eye + right_eye) / 2

        return {
            "left_eye_ear": left_eye,
            "right_eye_ear": right_eye,
            "average_eyes_ear": average
        }
        

    # Based on EAR (Eye Aspect Ratio) formula
    def __eye_openness_ratio(self,p1,p2,p3,p4,p5,p6):
        vertical_1 =  euclidean_distance(p2,p6)
        vertical_2 = euclidean_distance(p3,p5)
        horizontal = euclidean_distance(p1,p4)
        
        return (vertical_1 + vertical_2) / (2 * horizontal)
    