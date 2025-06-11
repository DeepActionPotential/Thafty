import sys
import cv2
import threading
import traceback
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
from PyQt5.QtCore import QTimer, Qt, pyqtSignal, QThread

from core.detection_manager import DetectionManager
from config import DefaultCfg
from services.faiss_manager import FAISSManager


from utils.utils import list_dir_image_groups, brighten_button_icon

class CameraStreaming(QWidget):
    # Signal to emit alarms list to GUI thread
    alarms_signal = pyqtSignal(list)

    def __init__(self, detection_manager: DetectionManager, default_config: DefaultCfg,
                 parent=None):
        super().__init__(parent)
        self.detection_manager = detection_manager
        self.default_config = default_config
        self._streaming = False
        self.cap = None
        self.latest_frame = None
        self.time_label = None  # Initialize time_label
        self.seek_slider = None  # Initialize seek_slider
        self.camera_label = None  # Initialize camera_label
        self.toggle_btn = None  # Initialize toggle_btn
        self._periodic_thread = None  # Initialize periodic thread
        self._run_periodic = False  # Control flag for periodic thread
        self.init_ui()  # Initialize UI first
        self.init_camera()  # Then initialize camera

    def init_camera(self):
        if self.cap is not None:
            self.cap.release()
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            print("Error: Could not open camera")
            return
            
        # Set camera to use maximum resolution
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)  # Max width
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)  # Max height
        
        # Initialize timers
        self.frame_timer = QTimer(self)
        self.frame_timer.timeout.connect(self._update_frame)
        self.seconds_timer = QTimer(self)
        self.seconds_timer.timeout.connect(self._update_seconds)
        
        # Start streaming automatically
        self._start_streaming()

    def init_ui(self):
        # Set minimum size for the widget
        self.setMinimumSize(640, 480)
        
        # Set size policy to be expanding
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 0, 10, 0)
        layout.setSpacing(0)
        layout.setStretch(0, 1)  # Camera label takes all available space

        # Camera feed display
        self.camera_label = QLabel()
        self.camera_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.camera_label.setAlignment(Qt.AlignCenter)
        self.camera_label.setMinimumSize(640, 480)  # Set minimum size for the label
        self.camera_label.setStyleSheet("background-color: black;")  # Black background when no frame
        layout.addWidget(self.camera_label)

        # Play/Stop controls
        btn_row = QHBoxLayout()
        btn_row.addStretch()
        self.toggle_btn = QPushButton()
        self.toggle_btn.setIcon(self.style().standardIcon(QStyle.SP_MediaPlay))
        brighten_button_icon(self.toggle_btn)
        self.toggle_btn.clicked.connect(self._toggle_stream)
        btn_row.addWidget(self.toggle_btn)
        btn_row.addStretch()
        layout.addLayout(btn_row)

        # Separator
        layout.addWidget(self._sep())

        # Elapsed time slider and label
        slider_layout = QHBoxLayout()
        slider_layout.setContentsMargins(10, 6, 10, 6)
        slider_layout.setSpacing(10)

        self.time_label = QLabel("0s")
        self.time_label.setFixedWidth(60)
        self.time_label.setAlignment(Qt.AlignCenter)
        slider_layout.addWidget(self.time_label)

        self.seek_slider = QSlider(Qt.Horizontal)
        self.seek_slider.setRange(0, 100)
        self.seek_slider.setValue(0)
        self.seek_slider.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        slider_layout.addWidget(self.seek_slider)

        layout.addLayout(slider_layout, stretch=1)

    def _sep(self):
        sep = QFrame()
        sep.setFrameShape(QFrame.HLine)
        sep.setFrameShadow(QFrame.Sunken)
        sep.setFixedHeight(2)
        return sep

    def _toggle_stream(self):
        if self._streaming:
            self._stop_streaming()
            self.toggle_btn.setIcon(self.style().standardIcon(QStyle.SP_MediaPlay))
            brighten_button_icon(self.toggle_btn)

        else:
            self._start_streaming()
            self.toggle_btn.setIcon(self.style().standardIcon(QStyle.SP_MediaStop))
            brighten_button_icon(self.toggle_btn)

    def _start_streaming(self):
        if not hasattr(self, 'cap') or not self.cap or not self.cap.isOpened():
            print("Cannot start streaming: Camera not available")
            return
            
        self.seconds = 0
        if self.time_label:
            self.time_label.setText("00:00:00s")
        if self.seek_slider:
            self.seek_slider.setValue(0)
            
        try:
            self.frame_timer.start(30)  # ~33 FPS
            self.seconds_timer.start(1000)  # 1 second
            self._streaming = True
            self._start_periodic_thread()
            if self.toggle_btn:
                self.toggle_btn.setIcon(self.style().standardIcon(QStyle.SP_MediaStop))
                brighten_button_icon(self.toggle_btn)
        except Exception as e:
            print(f"Error starting stream: {e}")

    def _stop_streaming(self):
        self._streaming = False
        
        if hasattr(self, 'frame_timer') and self.frame_timer.isActive():
            self.frame_timer.stop()
        if hasattr(self, 'seconds_timer') and self.seconds_timer.isActive():
            self.seconds_timer.stop()
            
        if hasattr(self, 'camera_label') and self.camera_label:
            self.camera_label.clear()
            
        self._stop_periodic_thread()
        
        if hasattr(self, 'toggle_btn') and self.toggle_btn:
            self.toggle_btn.setIcon(self.style().standardIcon(QStyle.SP_MediaPlay))
            brighten_button_icon(self.toggle_btn)

    def _update_seconds(self):

        # increment total elapsed seconds
        self.seconds += 1
        # compute h:m:s
        hrs   = self.seconds // 3600
        mins  = (self.seconds % 3600) // 60
        secs  = self.seconds % 60
        # format as HH:MM:SS
        self.time_label.setText(f"{hrs:02d}:{mins:02d}:{secs:02d}")

    def _update_frame(self):
        if not self._streaming or self.cap is None or not self.cap.isOpened():
            return
            
        ret, frame = self.cap.read()
        if not ret:
            return
            
        # Store the original frame for processing
        self.latest_frame = frame.copy()
        
        # Convert the image from BGR (OpenCV) to RGB for display
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Get the size of the label
        label_size = self.camera_label.size()
        if label_size.width() <= 0 or label_size.height() <= 0:
            return
            
        # Calculate aspect ratio
        h, w = frame_rgb.shape[:2]
        label_aspect = label_size.width() / label_size.height()
        frame_aspect = w / h
        
        # Resize frame to fit the label while maintaining aspect ratio
        if frame_aspect > label_aspect:
            # Frame is wider than label
            new_w = label_size.width()
            new_h = int(new_w / frame_aspect)
        else:
            # Frame is taller than label
            new_h = label_size.height()
            new_w = int(new_h * frame_aspect)
            
        # Resize the frame
        resized_frame = cv2.resize(frame_rgb, (new_w, new_h), interpolation=cv2.INTER_AREA)
        
        # Convert to QImage and then to QPixmap
        height, width, channel = resized_frame.shape
        bytes_per_line = 3 * width
        q_img = QImage(resized_frame.data, width, height, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(q_img)
        
        # Set the pixmap on the label
        self.camera_label.setPixmap(pixmap)

    def resizeEvent(self, event):
        """Handle window resize events to maintain aspect ratio"""
        if self._streaming and self.latest_frame is not None:
            self._update_frame()
        super().resizeEvent(event)

    def _periodic_task(self):
        if self.latest_frame is None:
            self.alarms_signal.emit([])
            return
        
        try:
            alarms = self.detection_manager.detect_alarm_in_image(
                image=self.latest_frame,
                matching_top_k=self.default_config.matching_top_k,
                faiss_manager_top_k=self.default_config.faiss_manager_top_k,
                found_coverd_body_alarm_wait=self.default_config.found_coverd_body_alarm_wait
            )
            self.alarms_signal.emit(alarms)
        
        except Exception:
            traceback.print_exc()
            self.alarms_signal.emit([])

    def _periodic_worker(self):


        interval_time_in_seconds = self.default_config.frame_analysis_interval_wait_in_seconds

        interval_ms = 100
        ticks = int((interval_time_in_seconds * 1000) / interval_ms)

        while self._run_periodic:
            self._periodic_task()
            for _ in range(ticks):
                if not self._run_periodic:
                    return
                QThread.msleep(interval_ms)

    def _start_periodic_thread(self):
        if self._periodic_thread and self._periodic_thread.is_alive():
            return
        self._run_periodic = True
        self._periodic_thread = threading.Thread(
            target=self._periodic_worker, daemon=True
        )
        self._periodic_thread.start()

    def _stop_periodic_thread(self):
        if not hasattr(self, '_run_periodic') or not hasattr(self, '_periodic_thread'):
            return
            
        self._run_periodic = False
        
        if self._periodic_thread is not None:
            try:
                self._periodic_thread.join(timeout=1)
            except RuntimeError:
                pass  # Thread was not started
            self._periodic_thread = None

    def close(self):
        if self.cap.isOpened():
            self.cap.release()