import sys

from src.dataset.checkin_record import CheckInRecordBuilder
from src.dataset.csv_writer import CSVWriter
from src.features.extractor import FeatureExtractor
from src.vision.imageloader import ImageLoader
from src.vision.detector import FaceDetector
from src.vision.landmark_mapper import LandmarkMapper
from src.vision.visualizer import Visualizer

image = ImageLoader.load_image('data/img/no_smile.jpg')
height, width, _ = image.shape

detector = FaceDetector()
detection_result = detector.detect(image)

landmarks_matrix, raw_blendshapes = LandmarkMapper.map_to_pixel_matrix(
    detection_result, image_width=width, image_height=height
)

extractor = FeatureExtractor()
features = extractor.extract(landmarks_matrix, raw_blendshapes)    

user_mood = 3        
record = CheckInRecordBuilder.build_record(features, mood=user_mood)           
csv_path = "./data/csv/features.csv"

writer = CSVWriter(csv_path)
writer.append(record)


if __name__ == "__main__":

    if "--debug" in sys.argv or "--d" in sys.argv:
        if landmarks_matrix is None:
            print("Aucun visage détecté dans l'image.")
            exit()
            
        SHOW_ALL_LANDMARKS = False
        SELECTED_LANDMARKS = [33, 160, 158, 133, 153, 144, 263, 385, 387, 362, 373, 380]
        Visualizer.draw_landmarks(
            image,
            landmarks_matrix,
            selected_landmarks=SELECTED_LANDMARKS, 
            show_all=SHOW_ALL_LANDMARKS
        )
        Visualizer.show_image(image, window_name="Landmarks Visualization")