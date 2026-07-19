from collections import namedtuple

import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from src.utils.normalized_to_pixel import normalized_to_pixel_coordinates
from src.features.extractor import FeatureExtractor

SHOW_ALL_LANDMARKS = False
SELECTED_LANDMARKS = [33, 160, 158, 133, 153, 144, 263, 385, 387, 362, 373, 380]

base_options = python.BaseOptions(model_asset_path='./models/face_landmarker.task')
options = vision.FaceLandmarkerOptions(base_options=base_options,
                                       output_face_blendshapes=True,
                                       output_facial_transformation_matrixes=True,
                                       num_faces=1)
detector = vision.FaceLandmarker.create_from_options(options)


frame = cv2.imread('data/img/no_smile.jpg')
resized_img = cv2.resize(frame,(0,0),fx=2,fy=2)
image_rgb = cv2.cvtColor(resized_img, cv2.COLOR_BGR2RGB)
image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
detection_result = detector.detect(image)
image_copy = np.copy(image.numpy_view())
height, width,_ = image_copy.shape



for face_blendshapes in detection_result.face_blendshapes:
    blendshape_data = []
    for blendshape in face_blendshapes:
        blendshape_data.append({'category_name': blendshape.category_name, 'score': blendshape.score})
    df = pd.DataFrame(blendshape_data)
    df.to_csv('data/csv/blendshapes.csv', index=False, encoding='utf-8') 

    
for faces_landmarks in detection_result.face_landmarks:
    normalized_to_pixel_landmarks = []
    
    for idx, landmark in enumerate(faces_landmarks):
        coords = normalized_to_pixel_coordinates(landmark.x, landmark.y, landmark.z, width, height)
        if coords:
          
            normalized_to_pixel_landmarks.append({'index': idx,'x': coords[0],'y': coords[1],'z': coords[2]})
            
            if idx in SELECTED_LANDMARKS or SHOW_ALL_LANDMARKS:
              cv2.circle(image_copy, (coords[0], coords[1]), 2, (0, 255, 0), -1)
              text_coords = (coords[0] + 5, coords[1] + 5)
              cv2.putText(
                image_copy, 
                str(idx), 
                text_coords, 
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.5,          
                (0, 0, 255),  
                2)

    Landmark = namedtuple('Landmark', ['x', 'y', 'z'])
    landmarks_objects = [Landmark(pt['x'], pt['y'], pt['z']) for pt in normalized_to_pixel_landmarks]
    
    Blendshape = namedtuple('Blendshape', ['category_name', 'score'])
    blendshapes_dict = {item['category_name']: item['score'] for item in blendshape_data}
  
    features = FeatureExtractor().extract(landmarks_objects,blendshapes_dict)    
    print("Extracted Features:", features)
                
rgb_annotated_image = cv2.cvtColor(image_copy, cv2.COLOR_BGR2RGB)
cv2.imshow('frame',rgb_annotated_image)



cv2.waitKey(0)    
cv2.destroyAllWindows()