from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Tuple, Union
from scrfd import Face
import numpy as np
from PIL import Image
import os
from pathlib import Path


@dataclass
class FaceMainPoints:
    """
    The main points of the face.

    Attributes:
    box_start_point (Tuple[int, int]): The start point of the bounding box.
    box_end_point (Tuple[int, int]): The end point of the bounding box.
    box_probabilty_score (float): The probability score of the bounding box.
    left_eye (Tuple[int, int], optional): The left eye coordinates. Defaults to (0, 0).
    right_eye (Tuple[int, int], optional): The right eye coordinates. Defaults to (0, 0).
    nose (Tuple[int, int], optional): The nose coordinates. Defaults to (0, 0).
    left_mouth (Tuple[int, int], optional): The left mouth coordinates. Defaults to (0, 0).
    right_mouth (Tuple[int, int], optional): The right mouth coordinates. Defaults to (0, 0).
    """
    box_start_point: Tuple[int, int]
    box_end_point: Tuple[int, int]
    box_probabilty_score: float
    left_eye: Tuple[int, int] = (0, 0)
    right_eye: Tuple[int, int] = (0, 0)
    nose: Tuple[int, int] = (0, 0)
    left_mouth: Tuple[int, int] = (0, 0)
    right_mouth: Tuple[int, int] = (0, 0)


class FaceDetector(ABC):
    """
    The face detector interface.
    """

    @abstractmethod
    def detect(self, image_path: str) -> List[FaceMainPoints]:
        """
        Detect the faces in an image.

        Args:
        image_path (str): The path to the image.

        Returns:
        A list of FaceMainPoints objects, one for each face in the image.
        """
        pass

    @abstractmethod
    def convert_face_to_face_main_points(self, face: Face) -> FaceMainPoints:
        """
        Convert a Face object to a FaceMainPoints object.

        Args:
        face (Face): The face to convert.

        Returns:
        A FaceMainPoints object.
        """
        pass


@dataclass
class PersonDetectionBox:
    """
    A bounding box detected by a detector, with an associated confidence and class ID.

    Attributes:
    x1 (float): The minimum x-coordinate of the bounding box.
    y1 (float): The minimum y-coordinate of the bounding box.
    x2 (float): The maximum x-coordinate of the bounding box.
    y2 (float): The maximum y-coordinate of the bounding box.
    confidence (float): The confidence score of the detection.
    class_id (int): The class ID of the detection.
    """
    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float
    class_id: int


class ImageEmbedder(ABC):
    """
    The image embedding interface.
    """

    @abstractmethod
    def embed(self, image: Union[Image.Image, np.ndarray, Path]) -> np.ndarray:
        """
        Embed the image.

        Args:
        image (Union[Image.Image, np.ndarray, Path]): The image to embed.

        Returns:
        A numpy array of the embedding.
        """
        pass


@dataclass
class FaceMatchingAlarm:
    """
    An alarm raised when a face is matched.

    Attributes:
    known_face (np.ndarray): The known face image.
    extracted_face (np.ndarray): The extracted face image.
    verify (bool): Whether the face is verified.
    """
    known_face: np.ndarray
    extracted_face: np.ndarray
    verify: bool


@dataclass
class NoFaceMatchingAlarm:
    """
    An alarm raised when a face is not matched.

    Attributes:
    extracted_face (np.ndarray): The extracted face image.
    verify (bool): Whether the face is verified.
    known_face (np.ndarray, optional): The known face image. Defaults to None.
    """
    extracted_face: np.ndarray
    verify: bool
    known_face: np.ndarray = None


@dataclass
class FoundBodyNoFaceAlarm:
    """
    An alarm raised when a body is detected without a face.

    Attributes:
    annotated_image (np.ndarray): The annotated image.
    """
    annotated_image: np.ndarray


@dataclass
class NoAlarm:
    """
    No alarm is raised.

    Attributes:
    alarm (bool): Whether an alarm is raised.
    """
    alarm: bool