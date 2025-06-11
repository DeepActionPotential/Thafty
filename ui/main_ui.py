import sys
import cv2
import threading
import time
import numpy as np
from typing import List, Union
from pathlib import Path
from PyQt5.QtWidgets import (
    QApplication, QWidget, QLabel, QVBoxLayout, QHBoxLayout,
    QSlider, QPushButton, QFrame, QSizePolicy, QStyle,
    QFileDialog, QScrollArea
)
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import QTimer, Qt, pyqtSignal

from core.detection_manager import DetectionManager
from config import DefaultCfg
from services.telegram_manager import TelegramNotificationManager
from services.faiss_manager import FAISSManager
from schemas.services_schemas import (
    ImageEmbedder, NoAlarm, FoundBodyNoFaceAlarm,
    FaceMatchingAlarm, NoFaceMatchingAlarm
)
from utils.utils import list_dir_image_groups



from .alarm_ui import AlarmBar
from .bars_ui import KnownPeopleBar, ToolsBar
from .camera_ui import CameraStreaming



class CameraApp(QWidget):
    def __init__(self, detection_manager, faiss_manager,
                 image_embedder, telegram_notification_manager, default_config):
        super().__init__()
        self.detection_manager = detection_manager
        self.faiss_manager = faiss_manager
        self.image_embedder = image_embedder
        self.telegram_notification_manager = telegram_notification_manager
        self.default_config = default_config

        self.setWindowFlags(Qt.Window | Qt.FramelessWindowHint | Qt.WindowSystemMenuHint | Qt.WindowMinimizeButtonHint | Qt.WindowCloseButtonHint)
        self.setWindowTitle("Camera UI – Dark Theme")
        # self.showMaximized()
        self.init_ui()

    def init_ui(self):
        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(0, 0, 0, 0)
        self.layout.setSpacing(0)

        # camera view
        self.camera_stream = CameraStreaming(
            self.detection_manager,
            self.default_config,
            parent=self
        )
        self.layout.addWidget(self.camera_stream, stretch=5)
        self.layout.addWidget(self._sep())

        # hook telegram send directly on alarms
        self.camera_stream.alarms_signal.connect(
            self.telegram_notification_manager.send_alarms
        )

        # known people bar (start hidden if you prefer)
        self.people_bar = KnownPeopleBar(self)
        self.layout.addWidget(self.people_bar, stretch=1)
        self.layout.addWidget(self._sep())
        self.people_bar.setVisible(False)

        # tools bar, pass toggle callback
        self.tools_bar = ToolsBar(
            load_cb=self._on_load,
            save_cb=self._on_save,
            add_cb=self.__on_adding_images,
            toggle_people_cb=self._toggle_people,
            parent=self
        )
        self.layout.addWidget(self.tools_bar)

    def _sep(self):
        sep = QFrame()
        sep.setFrameShape(QFrame.HLine)
        sep.setFrameShadow(QFrame.Sunken)
        sep.setFixedHeight(2)
        return sep

    def _toggle_people(self, show: bool):
        # Pause streaming if active
        was_streaming = self.camera_stream._streaming if hasattr(self.camera_stream, '_streaming') else False
        if was_streaming:
            self.camera_stream._stop_streaming()
        
        # Toggle people bar visibility
        self.people_bar.setVisible(show)
        
        # Adjust layout stretches based on visibility
        if show:
            self.layout.setStretch(0, 3)  # Camera takes 3/4 of space
            self.layout.setStretch(2, 1)  # People bar takes 1/4
        else:
            self.layout.setStretch(0, 1)  # Camera takes all space
            self.layout.setStretch(2, 0)  # People bar takes no space
            
        # Restart streaming if it was active
        if was_streaming:
            self.camera_stream._start_streaming()

    def _on_load(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Load People", "", "Pickle (*.pkl);;All Files (*)"
        )
        if not path:
            return
        imgs = self.faiss_manager.load(path)
        self.people_bar.set_images(imgs, True)

    def _on_save(self):
        path, _ = QFileDialog.getSaveFileName(
            self, "Save People", "", "Pickle (*.pkl);;All Files (*)"
        )
        if not path:
            return
        self.faiss_manager.save(path)

    def __on_adding_images(self):
        path = QFileDialog.getExistingDirectory(
            self, "Add People Images Folder", ""
        )
        if not path:
            return
        imgs_info = list_dir_image_groups(path)
        imgs = self.faiss_manager.add_images(imgs_info, self.image_embedder)
        self.people_bar.set_images(imgs, False)

    def closeEvent(self, event):
        self.camera_stream.close()
        super().closeEvent(event)

