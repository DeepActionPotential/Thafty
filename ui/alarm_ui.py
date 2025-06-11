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

from schemas.services_schemas import (
    ImageEmbedder, NoAlarm, FoundBodyNoFaceAlarm,
    FaceMatchingAlarm, NoFaceMatchingAlarm
)
from utils.utils import numpy_to_pixmap
from services.telegram_manager import TelegramNotificationManager


class AlarmBar(QWidget):
    def __init__(self, telegram_notification_manager:TelegramNotificationManager, parent=None):
        super().__init__(parent)
        self.init_ui()
        self.telegram_notification_manager = telegram_notification_manager

    def init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 6, 10, 6)
        layout.setSpacing(5)

        header = QLabel("Alarm")
        layout.addWidget(header)

        # Horizontal scroll area for alarms
        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOn)
        self.scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.scroll.setFixedHeight(120)

        container = QWidget()
        self.alarms_layout = QHBoxLayout(container)
        self.alarms_layout.setContentsMargins(0, 0, 0, 0)
        self.alarms_layout.setSpacing(10)
        self.scroll.setWidget(container)
        layout.addWidget(self.scroll)

        self.run_test_alarms()

    def clear_alarms(self):
        while self.alarms_layout.count():
            item = self.alarms_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

    def _create_alarm_widget(self, alarm: Union[NoAlarm, FoundBodyNoFaceAlarm,
                                           FaceMatchingAlarm, NoFaceMatchingAlarm]) -> QWidget:
        """
        Returns a custom widget based on the alarm type with specific visual representation.
        """
        if isinstance(alarm, NoAlarm):
            return 
        elif isinstance(alarm, FaceMatchingAlarm):
            return self._create_face_matching_widget(alarm)
        elif isinstance(alarm, NoFaceMatchingAlarm):
            return self._create_no_face_matching_widget(alarm)
        elif isinstance(alarm, FoundBodyNoFaceAlarm):
            return self._create_found_body_widget(alarm)
        else:
            # Fallback for unknown alarm types
            lbl = QLabel(str(type(alarm).__name__))
            lbl.setFrameStyle(QFrame.Box | QFrame.Plain)
            lbl.setLineWidth(1)
            lbl.setMargin(5)
            return lbl

    def _create_no_alarm_widget(self, alarm: NoAlarm) -> QWidget:
        frame = QFrame()
        frame.setFixedSize(180, 140)
        frame.setStyleSheet("""
            QFrame {
                border: 2px dashed #4CAF50;
                border-radius: 8px;
                background-color: #1B1B1B;
            }
        """)
        
        layout = QVBoxLayout(frame)
        icon = QLabel("✓")
        icon.setStyleSheet("QLabel { color: #4CAF50; font-size: 32px; }")
        icon.setAlignment(Qt.AlignCenter)
        
        text = QLabel("No Alarms")
        text.setStyleSheet("QLabel { color: #AAAAAA; font-size: 12px; }")
        text.setAlignment(Qt.AlignCenter)
        
        layout.addWidget(icon)
        layout.addWidget(text)
        return frame

    def _create_face_matching_widget(self, alarm: FaceMatchingAlarm) -> QWidget:
        frame = QFrame()
        frame.setFixedSize(360, 160)
        border_color = "#4CAF50" if alarm.verify else "#F44336"
        frame.setStyleSheet(f"""
            QFrame {{
                border: 2px solid {border_color};
                border-radius: 8px;
                background-color: #2D2D2D;
            }}
        """)
        
        main_layout = QHBoxLayout(frame)
        main_layout.setContentsMargins(8, 8, 8, 8)
        
        # Known Face
        known_pixmap = numpy_to_pixmap(alarm.known_face)
        known_label = QLabel()
        known_label.setPixmap(known_pixmap)
        known_label.setStyleSheet("QLabel { background-color: #1B1B1B; border-radius: 4px; }")
        known_label.setAlignment(Qt.AlignCenter)
        
        # Matched Face
        matched_pixmap = numpy_to_pixmap(alarm.matched_face)
        matched_label = QLabel()
        matched_label.setPixmap(matched_pixmap)
        matched_label.setStyleSheet("QLabel { background-color: #1B1B1B; border-radius: 4px; }")
        matched_label.setAlignment(Qt.AlignCenter)
        
        # Text Info
        text_layout = QVBoxLayout()
        status_text = "Verified Match" if alarm.verify else "Verification Failed"
        status = QLabel(status_text)
        status.setStyleSheet(f"QLabel {{ color: {border_color}; font-size: 14px; font-weight: bold; }}")
        
        info = QLabel("Face Match")
        info.setStyleSheet("QLabel { color: #AAAAAA; font-size: 12px; }")
        
        text_layout.addWidget(status)
        text_layout.addWidget(info)
        text_layout.addStretch()
        
        main_layout.addWidget(known_label)
        main_layout.addWidget(matched_label)
        main_layout.addLayout(text_layout)
        return frame

    def _create_no_face_matching_widget(self, alarm: NoFaceMatchingAlarm) -> QWidget:
        frame = QFrame()
        frame.setFixedSize(360, 160)
        frame.setStyleSheet("""
            QFrame {
                border: 2px solid #F44336;
                border-radius: 8px;
                background-color: #2D2D2D;
            }
        """)
        
        main_layout = QHBoxLayout(frame)
        main_layout.setContentsMargins(8, 8, 8, 8)
        
        # Unmatched Face
        unmatched_pixmap = numpy_to_pixmap(alarm.unmatched_face)
        unmatched_label = QLabel()
        unmatched_label.setPixmap(unmatched_pixmap)
        unmatched_label.setStyleSheet("QLabel { background-color: #1B1B1B; border-radius: 4px; }")
        unmatched_label.setAlignment(Qt.AlignCenter)
        
        # Known Face (if available)
        known_label = QLabel()
        if alarm.known_face is not None:
            known_pixmap = numpy_to_pixmap(alarm.known_face)
            known_label.setPixmap(known_pixmap)
        known_label.setStyleSheet("QLabel { background-color: #1B1B1B; border-radius: 4px; }")
        known_label.setAlignment(Qt.AlignCenter)
        
        # Text Info
        text_layout = QVBoxLayout()
        status = QLabel("No Match Found")
        status.setStyleSheet("QLabel { color: #F44336; font-size: 14px; font-weight: bold; }")
        
        info = QLabel("Face not in database" + (" (Reference Available)" if alarm.known_face is not None else ""))
        info.setStyleSheet("QLabel { color: #AAAAAA; font-size: 12px; }")
        
        text_layout.addWidget(status)
        text_layout.addWidget(info)
        text_layout.addStretch()
        
        main_layout.addWidget(unmatched_label)
        if alarm.known_face is not None:
            main_layout.addWidget(known_label)
        main_layout.addLayout(text_layout)
        return frame

    def _create_found_body_widget(self, alarm: FoundBodyNoFaceAlarm) -> QWidget:

        frame = QFrame()
        frame.setFixedSize(240, 160)
        frame.setStyleSheet("""
            QFrame {
                border: 2px solid #FF9800;
                border-radius: 8px;
                background-color: #2D2D2D;
            }
        """)
        
        layout = QVBoxLayout(frame)
        layout.setContentsMargins(8, 8, 8, 8)
        
        # Body Image
        body_pixmap = numpy_to_pixmap(alarm.annotated_image)
        image_label = QLabel()
        image_label.setPixmap(body_pixmap)
        image_label.setStyleSheet("QLabel { background-color: #1B1B1B; border-radius: 4px; }")
        image_label.setAlignment(Qt.AlignCenter)
        
        # Text Info
        status = QLabel("Person Detected")
        status.setStyleSheet("QLabel { color: #FF9800; font-size: 14px; font-weight: bold; }")
        
        info = QLabel("No face detected")
        info.setStyleSheet("QLabel { color: #AAAAAA; font-size: 12px; }")
        
        layout.addWidget(image_label)
        layout.addWidget(status)
        layout.addWidget(info)
        return frame

    def set_alarms(self, alarms: List[Union[NoAlarm, FoundBodyNoFaceAlarm,
                                           FaceMatchingAlarm, NoFaceMatchingAlarm]]):

        # Add alarms in reverse order to maintain proper layout
        for alarm in reversed(alarms):
            widget = self._create_alarm_widget(alarm)
            if widget:
                self.alarms_layout.addWidget(widget, 0)
    

    def manage_alarms_notifications(self, alarms: List[Union[NoAlarm, FoundBodyNoFaceAlarm,
                                           FaceMatchingAlarm, NoFaceMatchingAlarm]]):

        self.telegram_notification_manager.send_alarms(alarms)
    


        
