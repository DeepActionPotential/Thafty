
from typing import List, Tuple, Optional
from ultralytics import YOLO
import cv2
import numpy as np
from PIL.Image import Image
from pathlib import Path
from typing import Union
from PIL import Image


from schemas.services_schemas import PersonDetectionBox
from utils.utils import open_image


class YOLOPersonDetector:
    """
    YOLO-based person detection system.

    This class provides a high-level interface for detecting people in images using
    YOLOv8 model. It includes methods for loading images, running detection,
    parsing results, and saving annotated images.

    Attributes:
        model: The YOLOv8 model instance
        confidence_threshold: Minimum confidence score for valid detections
    """

    def __init__(self, model_path: str, confidence_threshold: float = 0.75) -> None:
        """
        Initialize the YOLO person detector.

        Args:
            model_path: Path to the YOLOv8 model weights (.pt file)
            confidence_threshold: Minimum confidence score for valid detections (0.0-1.0)
        """
        self.model = YOLO(model_path)
        self.confidence_threshold = confidence_threshold

    def save_annotated_image(self, annotated_img: np.ndarray, save_path: str) -> None:
        """
        Save an annotated image to disk.

        Args:
            annotated_img: Image array with bounding boxes drawn
            save_path: Destination path for saving the image

        Note:
            The image is saved in BGR format (OpenCV default)
        """
        cv2.imwrite(save_path, annotated_img)
        print(f"Annotated image saved to {save_path}")

    def parse_detections(self, results) -> List[PersonDetectionBox]:
        """
        Parse YOLOv8 detection results into PersonDetectionBox objects.

        Args:
            results: YOLOv8 results object containing detection data

        Returns:
            List of PersonDetectionBox objects with:
                - Bounding box coordinates (x1, y1, x2, y2)
                - Confidence score
                - Class ID (0 for person)

        Note:
            This method assumes class ID 0 corresponds to person detections
        """
        boxes: List[PersonDetectionBox] = []
        for box in results[0].boxes:
            x1, y1, x2, y2 = box.xyxy.tolist()[0]
            conf_val = float(box.conf.tolist()[0])
            cls_id = int(box.cls.tolist()[0])
            boxes.append(PersonDetectionBox(x1, y1, x2, y2, conf_val, cls_id))
        return boxes
    
    

    def detect_person(
        self,
        image: Union[str, Path, np.ndarray, Image.Image],
        save_path: Optional[str] = None,
        show_boxes: bool = False
    ) -> Tuple[List[PersonDetectionBox], np.ndarray]:
        """
        Detect people in an image using YOLOv8.

        Args:
            image: Input image to process. Can be:
                - str: Path to image file
                - Path: Path object to image file
                - np.ndarray: Image array (BGR or RGB)
                - Image.Image: PIL Image object
            save_path: Optional path to save the annotated image
            show_boxes: If True, returns the annotated image with bounding boxes

        Returns:
            Tuple containing:
                - List[PersonDetectionBox]: List of detected person boxes with coordinates and confidence
                - np.ndarray: Annotated image with bounding boxes (if show_boxes=True)

        Note:
            - The function uses a confidence threshold set during initialization
            - Only detects person class (class ID 0)
            - Returns an empty list if no persons are detected
        """
        img = open_image(image)

        results = self.model(img, conf=self.confidence_threshold, classes=[0])

        detections = self.parse_detections(results)
        annotated_img = results[0].plot()

        if show_boxes or save_path:
            if save_path:
                self.save_annotated_image(annotated_img, save_path)

        return detections, annotated_img

