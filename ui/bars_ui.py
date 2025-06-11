
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



class KnownPeopleBar(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.init_ui()

    def init_ui(self):
        self.setVisible(False)
        # Reduced margins and spacing for a tighter layout
        outer_layout = QVBoxLayout(self)
        outer_layout.setContentsMargins(10, 0, 10, 0)
        outer_layout.setSpacing(1)


        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOn)
        self.scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        # Keep enough height for thumbnails but tighten if needed
        self.scroll.setFixedHeight(120)

        container = QWidget()
        self.images_layout = QHBoxLayout(container)
        self.images_layout.setContentsMargins(10, -100, 10, -100)
        self.images_layout.setSpacing(2)
        self.scroll.setWidget(container)

        outer_layout.addWidget(self.scroll)

    def set_images(self, image_list: List[np.ndarray], clear: bool):
        if clear:
            while self.images_layout.count():
                itm = self.images_layout.takeAt(0)
                if itm.widget():
                    itm.widget().deleteLater()
        for img in image_list:
            h, w, ch = img.shape
            qt = QImage(img.data, w, h, ch * w, QImage.Format_RGB888)
            pix = QPixmap.fromImage(qt).scaled(
                100, 100, Qt.KeepAspectRatio, Qt.SmoothTransformation
            )
            lbl = QLabel()
            lbl.setPixmap(pix)
            self.images_layout.addWidget(lbl)


class DragButton(QPushButton):
    """A little handle you can click+drag to move the window."""
    def __init__(self, parent=None):
        super().__init__("⠿", parent)
        self.setFixedSize(36, 36)
        # Make sure it doesn’t steal focus or show a hover highlight
        self.setFlat(True)
        self._drag_pos = None

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            # record where in the window we clicked
            self._drag_pos = event.globalPos()
            event.accept()
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self._drag_pos:
            # how far the mouse has moved since press
            delta = event.globalPos() - self._drag_pos
            # move the entire window by that delta
            self.window().move(self.window().pos() + delta)
            # update origin for smooth continuous dragging
            self._drag_pos = event.globalPos()
            event.accept()
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        self._drag_pos = None
        super().mouseReleaseEvent(event)


class ToolsBar(QWidget):
    def __init__(self, load_cb, save_cb, add_cb, toggle_people_cb, parent=None):
        super().__init__(parent)
        self.load_cb = load_cb
        self.save_cb = save_cb
        self.add_cb = add_cb
        self.toggle_people_cb = toggle_people_cb
        self.parent = parent
        self.init_ui()

    def init_ui(self):
        layout = QHBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(10)

        

        # File operations
        btn_load = QPushButton("Load")
        btn_load.clicked.connect(self.load_cb)
        layout.addWidget(btn_load)

        btn_save = QPushButton("Save")
        btn_save.clicked.connect(self.save_cb)
        layout.addWidget(btn_save)

        btn_add = QPushButton("Add People")
        btn_add.clicked.connect(self.add_cb)
        layout.addWidget(btn_add)

        # Toggle People Bar
        self.toggle_btn = QPushButton("Show People")
        self.toggle_btn.setCheckable(True)
        self.toggle_btn.toggled.connect(self._toggle_people)
        layout.addWidget(self.toggle_btn)

        layout.addStretch()

        # Window controls
        btn_min = QPushButton("🗕")
        btn_min.setFixedSize(36, 36)
        btn_min.clicked.connect(self.parent.showMinimized)
        layout.addWidget(btn_min)

        btn_max = QPushButton("🗖")
        btn_max.setFixedSize(36, 36)
        btn_max.clicked.connect(self._toggle_max_restore)
        layout.addWidget(btn_max)

        btn_close = QPushButton("✕")
        btn_close.setFixedSize(36, 36)
        btn_close.clicked.connect(self.parent.close)
        layout.addWidget(btn_close)

        # Drag handle
        drag_btn = DragButton(self)
        layout.addWidget(drag_btn)

    def _toggle_people(self, checked):
        self.toggle_btn.setText("Hide People" if checked else "Show People")
        self.toggle_people_cb(checked)

    def _toggle_max_restore(self):
        if self.parent.isMaximized():
            self.parent.showNormal()
        else:
            self.parent.showMaximized()