from pathlib import Path
import torch
from torchvision import models, transforms
from PIL.Image import Image
import numpy as np
from typing import Union
from deepface import DeepFace
import os

from utils.utils import open_image
from schemas.services_schemas import ImageEmbedder


class EfficientNetImageEmbedder(ImageEmbedder):
    """
    EfficientNet-based image embedding generator.

    This class generates fixed-size embeddings from images using EfficientNet B0
    architecture. It removes the classification head and uses the feature extractor
    part of the network.

    Attributes:
        embedding_model: PyTorch Sequential model containing EfficientNet backbone
        transform: Image preprocessing pipeline
    """

    def __init__(self, model_path: Path, device: str):
        """
        Initialize the EfficientNet embedding generator.

        Args:
            model_path: Path to the EfficientNet model weights file
            device: Device to run the model on ('cpu' or 'cuda')
        """
        # Load architecture
        model = models.efficientnet_b0()
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()

        # Remove classifier (keep only feature extractor)
        self.embedding_model = torch.nn.Sequential(
            model.features,
            model.avgpool,
            torch.nn.Flatten()
        ).to(device)

        # Image preprocessing pipeline
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
        ])



    def embed(self, image: Union[str, Path, np.ndarray, Image]) -> np.ndarray:
        """
        Generate an embedding vector from an input image.

        Args:
            image: Input image that can be:
                - str: Path to image file
                - Path: Path object to image file
                - np.ndarray: Image array (BGR or RGB)
                - PILImage.Image: PIL Image object

        Returns:
            np.ndarray: 1280-dimensional embedding vector

        Note:
            - The image is automatically preprocessed according to EfficientNet requirements
            - The embedding model runs in evaluation mode (no gradient tracking)
            - The output is a normalized vector suitable for similarity calculations
        """
        image = open_image(image)
        input_tensor = self.transform(image).unsqueeze(0)

        with torch.no_grad():
            embedding = self.embedding_model(input_tensor).squeeze().numpy()

        return embedding
    

