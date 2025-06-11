

from schemas.services_schemas import FaceMainPoints


from typing import List, Tuple, Optional, Union
import numpy as np
from PIL.Image import Image
from pathlib import Path
from typing import Union
from PIL import Image
import os
import cv2

from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QPushButton
from PyQt5.QtGui import QPixmap, QIcon, QPainter, QColor
from PyQt5.QtCore import Qt, QSize








def draw_faces_boxes(
    image_path: str,
    faces: List[FaceMainPoints],
    output_path: Optional[str] = None,
    scale_percent: int = 50
) -> None:
    """
    Draws bounding boxes and landmarks on the image for each face,
    then either saves or displays it scaled by a given percentage.

    Args:
        image_path (str): Path to source image.
        faces (List[FaceMainPoints]): List of face data to draw.
        output_path (Optional[str]): Path to save the annotated image. 
                                     If None, shows a window.
        scale_percent (int): Percent to scale displayed image (e.g. 50 = 50%).
    """
    # 1. Load
    img = cv2.imread(image_path)
    if img is None:
        print(f"Failed to load image: {image_path}")
        return

    # 2. Draw each face
    for face in faces:
        # a) box corners
        x1, y1 = map(int, face.box_start_point)
        x2, y2 = map(int, face.box_end_point)
        score = face.box_probabilty_score  # matches your dataclass

        # bounding box
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # confidence text
        cv2.putText(
            img,
            f"{score:.2f}",
            (x1, max(y1 - 10, 0)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            1
        )

        # landmarks
        for lx, ly in (
            face.left_eye,
            face.right_eye,
            face.nose,
            face.left_mouth,
            face.right_mouth
        ):
            if (lx, ly) != (0, 0):
                cv2.circle(img, (int(lx), int(ly)), 3, (0, 0, 255), -1)

    # 3. Save or display
    if output_path:
        cv2.imwrite(output_path, img)
        print(f"Saved annotated image to {output_path}")
    else:
        # compute new size
        h, w = img.shape[:2]
        new_w = int(w * scale_percent / 100)
        new_h = int(h * scale_percent / 100)
        disp = cv2.resize(img, (new_w, new_h))

        cv2.imshow("Annotated Faces", disp)
        cv2.waitKey(0)
        cv2.destroyAllWindows()




def open_image(image: Union[str, Path, Image.Image, np.ndarray], show: bool = False) -> Image.Image:
    """
    Load an image (from a path, OpenCV array, or PIL.Image) and return a PIL.Image in RGB.

    Args:
        image: 
        - Path or str: will be read via cv2.imread (fast C++ loader).
        - PIL.Image: will be converted to RGB.
        - np.ndarray: assumed to be an H×W×3 array (BGR or RGB—converted to RGB).
        show: If True, the image will be displayed using PIL's show() method.

    Returns:
        A PIL.Image in RGB mode.

    Raises:
        FileNotFoundError: if a path is given but the file doesn't exist or fails to load.
        ValueError: if the supplied object isn't a supported type.
    """
    # 1) Path or string → use OpenCV
    if isinstance(image, (str, Path)):
        path = str(image)
        arr = cv2.imread(path)
        if arr is None:
            raise FileNotFoundError(f"Image file not found or unreadable: '{path}'")
        # OpenCV gives BGR; convert to RGB
        arr = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(arr)

    # 2) PIL.Image → ensure RGB
    elif isinstance(image, Image.Image):
        image = image.convert("RGB")

    # 3) NumPy array → assume H×W×3, may be BGR or RGB
    elif isinstance(image, np.ndarray):
        arr = image
        # If dtype is uint8 and values seem in BGR order, we still convert
        if arr.ndim == 3 and arr.shape[2] == 3:
            # Convert BGR→RGB
            arr = arr[:, :, ::-1]
        image = Image.fromarray(arr.astype("uint8")).convert("RGB")

    else:
        raise ValueError(
            f"Unsupported image type: {type(image)}. "
            "Expected str/Path, PIL.Image, or np.ndarray."
        )

    # If show is True, display the image
    if show:
        image.show()
    
    return image





def list_dir_image_groups(groups_directory: str) -> List[Tuple[str, str]]:
    """
    List all images in subdirectories, returning paths and their parent directory names.

    Args:
        groups_directory: Path to the directory containing subdirectories of images

    Returns:
        List of tuples containing:
            - Full image path
            - Parent directory name (folder name)

    Note:
        - Each subdirectory represents a group/person
        - Supported image extensions: .jpg, .jpeg, .png, .bmp, .webp
    """
    # Supported image extensions (case-insensitive)
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
    
    all_images = []
    
    # Get all subdirectories
    for person_dir in os.listdir(groups_directory):
        person_dir_path = os.path.join(groups_directory, person_dir)
        
        # Skip if not a directory
        if not os.path.isdir(person_dir_path):
            continue
            
        # List images in this person's directory
        for file in os.listdir(person_dir_path):
            ext = os.path.splitext(file)[1].lower()
            if ext in image_extensions:
                full_path = os.path.join(person_dir_path, file)
                all_images.append((full_path, person_dir))
    
    return all_images



def pilimage_2_ndarray(pil_image: Image.Image) -> np.ndarray:
    """
    Convert a PIL Image to a NumPy array in RGB format.

    Args:
        pil_image (PIL.Image): Input PIL Image to be converted.

    Returns:
        np.ndarray: A NumPy array representing the image in RGB format with shape (H, W, 3).
    """
    if not isinstance(pil_image, Image.Image):
        raise ValueError("Input must be a PIL.Image object")
        
    # Ensure image is in RGB mode
    if pil_image.mode != 'RGB':
        pil_image = pil_image.convert('RGB')
    
    # Convert to NumPy array and ensure RGB order
    img_array = np.array(pil_image)
    
    # Verify and fix channel order if needed
    if img_array.shape[2] == 3:
        # Check if it looks like BGR by comparing channel means
        b_mean = img_array[:, :, 0].mean()
        r_mean = img_array[:, :, 2].mean()
        if b_mean > r_mean:  # If blue channel has higher mean than red, it's probably BGR
            img_array = img_array[:, :, ::-1]  # Convert BGR to RGB
    
    # If the image has an alpha channel, remove it
    if img_array.shape[-1] == 4:
        img_array = img_array[..., :3]
    
    return img_array


def ndarray_2_pilimage(ndarray_image: np.ndarray, save_path: str = None) -> Image.Image:
    """
    Convert a NumPy array to a PIL Image.
    
    Args:
        ndarray_image (np.ndarray): Input image as a NumPy array with shape (H, W, 3) for RGB or (H, W) for grayscale
        save_path (str, optional): Path to save the image. If None, image won't be saved. Defaults to None.
    
    Returns:
        Image.Image: A PIL Image object
    
    Raises:
        ValueError: If input is not a NumPy array or has incorrect shape
    """
    if not isinstance(ndarray_image, np.ndarray):
        raise ValueError("Input must be a NumPy array")
        
    # Convert to uint8 if not already
    if ndarray_image.dtype != np.uint8:
        ndarray_image = ndarray_image.astype(np.uint8)
        
    # Create PIL Image
    pil_image = Image.fromarray(ndarray_image, mode="RGB")
    
    # Save if save_path is provided
    if save_path is not None:
        pil_image.save(save_path)
    
    return pil_image





def numpy_to_pixmap(img: np.ndarray) -> QPixmap:
    """
    Convert a NumPy image (H×W or H×W×3 BGR) to a QPixmap of the same size.
    """
    h, w = img.shape[:2]
    if img.ndim == 3:
        # Convert BGR (OpenCV) → RGB
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # Create QImage with correct stride (bytes per line)
        qimg = QImage(rgb.data, w, h, rgb.strides[0], QImage.Format_RGB888)
    else:
        # Grayscale image
        qimg = QImage(img.data, w, h, img.strides[0], QImage.Format_Grayscale8)

    # Create QPixmap; it will be exactly w×h
    pix = QPixmap.fromImage(qimg)

    # If you ever need to force scaling to (w,h) explicitly:
    # pix = pix.scaled(w, h, Qt.IgnoreAspectRatio, Qt.SmoothTransformation)

    return pix








def brighten_button_icon(button: QPushButton, color: str = "#ffffff", size: int = 32):
    """
    Applies a bright color overlay to the QPushButton's icon.

    Args:
        button (QPushButton): The button whose icon should be brightened.
        color (str): Color name or hex code (default: '#00B8F4').
        size (int): Size of the icon in pixels.
    """
    icon = button.icon()
    if icon.isNull():
        return  # No icon to modify

    # Get pixmap from icon
    pixmap = icon.pixmap(size, size)

    # Create a new transparent pixmap to paint on
    colored_pixmap = QPixmap(pixmap.size())
    colored_pixmap.fill(Qt.transparent)

    # Paint the color onto the pixmap using composition
    painter = QPainter(colored_pixmap)
    painter.drawPixmap(0, 0, pixmap)
    painter.setCompositionMode(QPainter.CompositionMode_SourceIn)
    painter.fillRect(colored_pixmap.rect(), QColor(color))
    painter.end()

    # Set the new icon back on the button
    button.setIcon(QIcon(colored_pixmap))