
import numpy as np
from typing import Optional, List, Union
from PIL.Image import Image
from pathlib import Path
import time


from services.face_detector import SCRFDFaceDetector
from services.face_matcher import DeepFaceFaceMatcher
from services.image_emedding import EfficientNetImageEmbedder
from services.faiss_manager import FAISSManager
from services.person_detector import YOLOPersonDetector
from services.face_matcher import DeepFaceFaceMatcher
from utils.utils import open_image, list_dir_image_groups, pilimage_2_ndarray, ndarray_2_pilimage
from schemas.services_schemas import NoAlarm, FaceMatchingAlarm, NoFaceMatchingAlarm, FoundBodyNoFaceAlarm



class DetectionManager:
    """
    Class for detecting alarms in images. 

    It uses YOLO for person detection, SCRFD for face detection, EfficientNet for image embeddings, 
    FAISS for efficient nearest neighbor search and DeepFace for face matching.
    """

    def __init__(self, 
                 face_detector: SCRFDFaceDetector, 
                 person_detector: YOLOPersonDetector,
                 embedder: EfficientNetImageEmbedder,
                 faiss_manager: FAISSManager,
                 face_matcher: DeepFaceFaceMatcher,
                 ) -> None:
        """
        Initializes the TheifDetector.

        Args:
        face_detector: The face detector.
        person_detector: The person detector.
        embedder: The image embedder.
        faiss_manager: The FAISS manager.
        face_matcher: The face matcher.
        """
        self.face_detector = face_detector
        self.person_detector = person_detector
        self.embedder = embedder
        self.faiss_manager = faiss_manager
        self.face_matcher = face_matcher
    
    def detect_alarm_in_image(
        self,
        image: Union[str, Path, np.ndarray, Image],
        save_path: Optional[str] = None,
        matching_top_k: int = 2,
        faiss_manager_top_k: int = 2,
        found_coverd_body_alarm_wait: int = 3
    ) -> List[Union[NoAlarm, FoundBodyNoFaceAlarm, FaceMatchingAlarm, NoFaceMatchingAlarm]]:
        """
        Analyze an image for person detection and face matching, returning appropriate alarm objects.

        Args:
        image: Input image to process. Can be:
            - str: Path to image file
            - Path: Path object to image file
            - np.ndarray: Image array (BGR or RGB)
            - PILImage.Image: PIL Image object
        matching_top_k: The number of top matches to consider for face matching.
        save_path: Optional path to save the annotated image
        faiss_manager_top_k: The number of top matches to consider for FAISS search
        found_coverd_body_alarm_wait: Time to wait before giving alarm of a body with coverd face

        Returns:
        A list of alarm objects, one for each detected person in the image.
        """

        face_alarms = []


        # Step 1: Detect persons
        persons_detected, annotated_img = self.person_detector.detect_person(image)
        if not persons_detected:
            print("No persons detected.")
            return [NoAlarm(False)]

        # Step 2: Extract faces
        extracted_faces = self.face_detector.cut_extracted_faces(image, save_path)

        if not extracted_faces:
            time.sleep(found_coverd_body_alarm_wait)
            print("Persons found, but no faces detected.")
            return [FoundBodyNoFaceAlarm(annotated_img)]


        # Step 3: Match extracted faces
        for extracted_face in extracted_faces:

            query_embedding = self.embedder.embed(extracted_face)
            matches = self.faiss_manager.search_with_paths(query_embedding, k=faiss_manager_top_k)

            # Convert the query face to ndarray once
            extracted_face_array = pilimage_2_ndarray(open_image(extracted_face))




            matched = False

            for matched_path, _ in matches[:matching_top_k]:
                known_face_array = pilimage_2_ndarray(open_image(matched_path))
                is_match = self.face_matcher.is_same_person(extracted_face_array, known_face_array)


                if is_match:
                    face_alarms.append(
                        FaceMatchingAlarm(known_face=known_face_array, extracted_face=extracted_face_array, verify=True)
                    )
                    print("Match found!")
                    matched = True
                    break

            if not matched:
                face_alarms.append(
                    NoFaceMatchingAlarm(extracted_face=extracted_face_array, verify=False)
                )
                print("No matching face found.")

        return face_alarms
    


    



        


        

