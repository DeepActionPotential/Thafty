

from deepface import DeepFace
import os
import numpy as np



class DeepFaceFaceMatcher:
    """
    A class for face matching using the DeepFace library.

    Attributes:
        model_name (str): The name of the model to use.
        models_path (str): The path to the directory containing the models.
        model (deepface.Facematching): The face matching model.
    """
    def __init__(self, model_name: str, models_path: str):
        """
        Initializes the face Matcher.

        Args:
            model_name (str): The name of the model to use.
            models_path (str): The path to the directory containing the models.
        """
        self.model_name = model_name
        self.models_path = models_path
        self.model = DeepFace.build_model(model_name)

    def set_deepface_models_path(self, models_path: str) -> None:
        """
        Sets the path to the directory containing the models.
        """
        os.environ["DEEPFACE_HOME"] = models_path
    
    def is_same_person(self, img1: np.ndarray, img2: np.ndarray) -> bool:
        """
        Checks if two images contain the same person.

        Args:
            img1_path (str): The path to the first image.
            img2_path (str): The path to the second image.

        Returns:
            bool: True if the images contain the same person, False otherwise.
        """


        result = DeepFace.verify(img1_path=img1, img2_path=img2, model_name=self.model_name, enforce_detection=False)
        return result['verified']