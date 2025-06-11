from scrfd import SCRFD, Threshold, Face
from PIL.Image import Image
import numpy as np
from typing import List, Union
from pathlib import Path
import os

from schemas.services_schemas import FaceMainPoints, FaceDetector
from utils.utils import open_image


class SCRFDFaceDetector(FaceDetector):
    """
    A face detector using the SCRFD model.
    """

    def __init__(self, model_path: str, threshold_probability: float, nms:float):
        """
        Initializes the face detector.

        Args:
        model_path: The path to the SCRFD model.
        threshold_probability: The probability threshold for face detection.
        nms: The NMS threshold for face detection.
        """
        self.model = SCRFD.from_path(model_path)
        self.threshold = Threshold(probability=threshold_probability, nms=nms)


    def convert_face_to_face_main_points(self, face: Face) -> FaceMainPoints:
        """
        Converts a Face object to a FaceMainPoints object.

        Args:
        face: The face to convert.

        Returns:
        A FaceMainPoints object.
        """
        return FaceMainPoints(
            box_start_point=(face.bbox.upper_left.x, face.bbox.upper_left.y,),
            box_end_point=(face.bbox.lower_right.x, face.bbox.lower_right.y),
            box_probabilty_score=face.probability,
            left_eye=(face.keypoints.left_eye.x, face.keypoints.left_eye.y),
            right_eye=(face.keypoints.right_eye.x, face.keypoints.right_eye.y),
            nose=(face.keypoints.nose.x, face.keypoints.nose.y),
            left_mouth=(face.keypoints.left_mouth.x, face.keypoints.left_mouth.y),
            right_mouth=(face.keypoints.right_mouth.x, face.keypoints.right_mouth.y)
        )


    def detect(self, image: Union[str, Path, np.ndarray, Image]) -> List[FaceMainPoints]:
        """
        Detect faces in an image and return their main points.

        Args:
            image: Input image that can be:
                - str: Path to image file
                - Path: Path object to image file
                - np.ndarray: Image array (BGR or RGB)
                - PILImage.Image: PIL Image object

        Returns:
            List[FaceMainPoints]: List of face detections, each containing:
                - Bounding box coordinates
                - Key facial landmarks (eyes, nose, mouth)
                - Confidence score

        Note:
            - Uses a threshold set during initialization for detection confidence
            - Returns an empty list if no faces are detected
            - Image is automatically converted to RGB format
        """
        image = open_image(image).convert("RGB")

        extracted_faces = self.model.detect(image, threshold=self.threshold)

        faces = [self.convert_face_to_face_main_points(extracted_face) for extracted_face in extracted_faces]

        return faces
    

    def extract_face_from_image(self, img, face_main_points:FaceMainPoints):
        """
        Extract a face from an image using FaceMainPoints coordinates.
        
        Args:
            img: Input PIL Image
            face_main_points: FaceMainPoints object containing face coordinates
            
        Returns:
            PIL.Image: Cropped face image
        """
        start_x, start_y = face_main_points.box_start_point
        end_x, end_y = face_main_points.box_end_point
        
        return img.crop((start_x, start_y, end_x, end_y))

    def cut_extracted_faces(self, image: Union[str, Path, np.ndarray, Image], 
                           save_path: str = None) -> List[Image]:
        """
        Detect and extract all faces from an image, optionally saving them.
        
        Args:
            image: Input image that can be:
                - str: Path to image file
                - Path: Path object to image file
                - np.ndarray: Image array (BGR or RGB)
                - PILImage.Image: PIL Image object
            save_path: Optional path to save extracted faces
                If provided, faces will be saved as 'face_{index}.jpg' in this directory
                
        Returns:
            List[PIL.Image]: List of cropped face images
        """
        # Convert input to PIL Image
        img = open_image(image)
        
        # Detect faces and get their main points
        faces_information = self.detect(img)
        
        if not faces_information:
            return []  # Return empty list if no faces detected
            
        # Extract and optionally save each face
        extracted_faces = []
        for i, face_info in enumerate(faces_information):
            face_image = self.extract_face_from_image(img, face_info)
            extracted_faces.append(face_image)
            
            # Save face if save_path is provided
            if save_path:
                os.makedirs(save_path, exist_ok=True)
                face_path = os.path.join(save_path, f'face_{i}.jpg')
                face_image.save(face_path)
                
        return extracted_faces
