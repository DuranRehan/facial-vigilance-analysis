import cv2 
import mediapipe as mp

class ImageLoader: 
    
    @staticmethod
    def load_image(image_path,resize_factor=2):
        
        frame = cv2.imread(image_path)
        
        if frame is None:
            raise FileNotFoundError(f"Impossible de lire l'image à l'emplacement : {image_path}")
        
        resized_img = cv2.resize(frame,(0,0),fx=resize_factor,fy=resize_factor)
        return resized_img