import mediapipe as mp
import cv2
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

class FaceDetector:
    
    def __init__(self, model_path:str='./models/face_landmarker.task'):
        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.FaceLandmarkerOptions(base_options=base_options,
                                       output_face_blendshapes=True,
                                       output_facial_transformation_matrixes=True,
                                       num_faces=1)
        self.detector = vision.FaceLandmarker.create_from_options(options)
    
    def detect(self, image_bgr):
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
        
        return self.detector.detect(mp_image)