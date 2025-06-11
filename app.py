

import sys


# Core Imports
from core import detection_manager
from core.detection_manager import DetectionManager
from config import DefaultCfg

# Services Imports
from services.face_detector import SCRFDFaceDetector
from services.person_detector import YOLOPersonDetector
from services.image_emedding import EfficientNetImageEmbedder
from services.faiss_manager import FAISSManager
from services.face_matcher import DeepFaceFaceMatcher
from services.telegram_manager import TelegramNotificationManager


# PyQt5 Imports
from PyQt5.QtWidgets import (
    QApplication)
from PyQt5.QtGui import QFontDatabase, QFont
from PyQt5.QtCore import Qt

# Local UI Imports
from ui.main_ui import CameraApp

# Initialize Services
embeddor = EfficientNetImageEmbedder(
    model_path=DefaultCfg.embedding_model_path, 
    device=DefaultCfg.embedding_device
)

faiss_manager = FAISSManager(dimension=1280)

face_matcher = DeepFaceFaceMatcher(
    model_name=DefaultCfg.deepface_model_name, 
    models_path=DefaultCfg.deepface_models_path
)

detection_manager = DetectionManager(
    face_detector=SCRFDFaceDetector(
        model_path=DefaultCfg.face_model_path, 
        threshold_probability=DefaultCfg.face_threshold_probability, 
        nms=DefaultCfg.face_nms
    ),
    person_detector=YOLOPersonDetector(
        model_path=DefaultCfg.person_model_path, 
        confidence_threshold=DefaultCfg.person_confidence_threshold
    ),
    embedder=embeddor,
    faiss_manager=faiss_manager,
    face_matcher=face_matcher
)

telegram_notification_manager = TelegramNotificationManager(
    DefaultCfg.telegram_bot_token, 
    DefaultCfg.notification_chat_id
)

# Font Loading Function
def load_fonts():
    """Load custom fonts for the application"""
    db = QFontDatabase()
    # Add the two core weights
    id_regular = db.addApplicationFont("./assets/fonts/RoadUI-Regular.ttf")
    id_medium  = db.addApplicationFont("./assets/fonts/RoadUI-Medium.ttf")

    families = db.applicationFontFamilies(id_regular)
    if families:
        return families[0]  # should be "Roboto"
    return None

# Main Application Entry Point
if __name__ == "__main__":
    # Initialize Qt Application
    app = QApplication(sys.argv)

    # Load and set application font
    font_family = load_fonts()
    if font_family:
        app.setFont(QFont(font_family))

    # Load and apply styles
    with open("./ui/styles.css") as f:
        app.setStyleSheet(f.read())
        
    # Initialize main window
    window = CameraApp(
        detection_manager=detection_manager,
        faiss_manager=faiss_manager,
        image_embedder=embeddor,
        telegram_notification_manager=telegram_notification_manager,
        default_config=DefaultCfg()
    )

    # Show window and start event loop
    window.show()
    sys.exit(app.exec_())
