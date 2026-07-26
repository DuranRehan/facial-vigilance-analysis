import numpy as np
import math

from src.models.types import Landmark

class LandmarkMapper:
    
    @staticmethod
    def map_to_pixel_matrix(detection_result, image_width: int, image_height: int):
       
        if not detection_result.face_landmarks:
            return None, None

        raw_landmarks = detection_result.face_landmarks[0]
        landmarks_objects = []
     
        for lm in raw_landmarks:
            x_pixel = min(math.floor(lm.x * image_width), image_width - 1)
            y_pixel = min(math.floor(lm.y * image_height), image_height - 1)
            z_pixel = min(math.floor(lm.z * image_width), image_width - 1)
            landmarks_objects.append(Landmark(x=x_pixel, y=y_pixel, z=z_pixel))
        
        blendshapes_dict = {}
        if detection_result.face_blendshapes:
            blendshapes_dict = {
                b.category_name: b.score 
                for b in detection_result.face_blendshapes[0]
            }

        return landmarks_objects, blendshapes_dict
    
    