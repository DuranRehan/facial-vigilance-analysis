
import cv2


class Visualizer:
    
    @staticmethod
    def draw_landmarks(image,landmarks,selected_landmarks=None,show_all=False):
        for idx,landmark in enumerate(landmarks):
            x, y, z = landmark
            if idx in selected_landmarks or show_all:
                    cv2.circle(image, (x, y), 2, (0, 255, 0), -1)
                    text_coords = (x + 5, y + 5)
                    cv2.putText(
                    image, 
                    str(idx), 
                    text_coords, 
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    0.5,          
                    (0, 0, 255),  
                    2)
    
    @staticmethod
    def show_image(image, window_name="Image"):
        cv2.imshow(window_name, image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    