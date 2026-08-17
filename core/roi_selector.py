from __future__ import annotations

from PIL.ImageChops import screen
from PySide6.QtGui import QGuiApplication
from dataclasses import dataclass

import cv2
import numpy as np

from PySide6.QtCore import QPointF, Qt, Signal
from PySide6.QtGui import (
    QColor,
    QImage,
    QPainter,
    QPen,
    QPixmap,
)
from PySide6.QtWidgets import (
    QApplication,
    QDialog,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from core.models import ImageData, ROI

class ROISelector(QDialog):
    """
    Interactive ROI selector for a high-resolution image.

    The displayed image may be scaled to fit the window, but all
    ROI coordinates are stored in original-image coordinates.
    """

    roi_confirmed = Signal(object)

    MIN_ROI_WIDTH = 50
    MIN_ROI_HEIGHT = 50

    GRID_DIVISIONS = 10

    def __init__(
        self,
        image_data: ImageData,
        parent: QWidget | None = None,
    ) -> None:

        super().__init__(parent)

        self.image_data = image_data
        self.image = image_data.image

        self.image_height, self.image_width = self.image.shape[:2]

        # -----------------------------------------------------
        # ROI state
        # -----------------------------------------------------

        self.roi: ROI | None = None

        # Mouse positions are temporarily stored in
        # display coordinates while dragging.
        self._drag_start_display: QPointF | None = None
        self._drag_current_display: QPointF | None = None

        # -----------------------------------------------------
        # Display transform
        # -----------------------------------------------------

        self._display_scale = 1.0
        self._display_offset_x = 0.0
        self._display_offset_y = 0.0

        self._setup_ui()

    def _setup_ui(self) -> None:

        self.setWindowTitle(
            "Blueprint Author — Select Target ROI"
        )

        self.setMinimumSize(
            600,
            450,
        )

        screen = QGuiApplication.primaryScreen()

        if screen is not None:
            available = screen.availableGeometry()

            self.resize(
                int(available.width() * 0.80),
                int(available.height() * 0.80),
            )
            
        else:
            self.resize(
                1200,
                900,
            )

        # -----------------------------------------------------
        # Image canvas
        # -----------------------------------------------------

        self.canvas = ROICanvas(
            self,
        )

        self.canvas.set_image_data(
            self.image
        )

        self.canvas.roi_changed.connect(
            self._on_roi_changed
        )

        # -----------------------------------------------------
        # Status label
        # -----------------------------------------------------

        self.status_label = QLabel(
            "Drag a rectangle around the target."
        )

        self.status_label.setAlignment(
            Qt.AlignmentFlag.AlignCenter
        )

        # -----------------------------------------------------
        # Buttons
        # -----------------------------------------------------

        self.reset_button = QPushButton(
            "Reset"
        )

        self.reset_button.clicked.connect(
            self.reset_roi
        )

        self.confirm_button = QPushButton(
            "Confirm"
        )

        self.confirm_button.clicked.connect(
            self.confirm_roi
        )

        self.cancel_button = QPushButton(
            "Cancel"
        )

        self.cancel_button.clicked.connect(
            self.reject
        )

        button_layout = QHBoxLayout()

        button_layout.addStretch()

        button_layout.addWidget(
            self.reset_button
        )

        button_layout.addWidget(
            self.cancel_button
        )

        button_layout.addWidget(
            self.confirm_button
        )

        # -----------------------------------------------------
        # Main layout
        # -----------------------------------------------------

        layout = QVBoxLayout(self)

        layout.addWidget(
            self.canvas,
            stretch=1,
        )

        layout.addWidget(
            self.status_label
        )

        layout.addLayout(
            button_layout
        )

        self.setLayout(layout)

    def _on_roi_changed(
        self,
        roi: ROI | None,
    ) -> None:

        self.roi = roi

        if roi is None:

            self.status_label.setText(
                "Drag a rectangle around the target."
            )

            return

        self.status_label.setText(
            f"ROI: "
            f"x={roi.x}, "
            f"y={roi.y}, "
            f"w={roi.width}, "
            f"h={roi.height}"
        )

    def reset_roi(self) -> None:

        self.roi = None

        self.canvas.clear_roi()

        self.status_label.setText(
            "Drag a rectangle around the target."
        )

    def confirm_roi(self) -> None:

        if self.roi is None:

            self.status_label.setText(
                "Please draw an ROI first."
            )

            return

        if (
            self.roi.width < self.MIN_ROI_WIDTH
            or
            self.roi.height < self.MIN_ROI_HEIGHT
        ):

            self.status_label.setText(
                "ROI is too small."
            )

            return

        self.roi_confirmed.emit(
            self.roi
        )

        self.accept()

class ROICanvas(QWidget):
    """
    Canvas responsible for displaying the image and handling
    ROI interaction.

    Mouse coordinates are display coordinates.
    ROI coordinates are converted immediately into
    original-image coordinates.
    """

    roi_changed = Signal(object)

    def __init__(
        self,
        parent: QWidget | None = None,
    ) -> None:

        super().__init__(parent)

        self.setMouseTracking(
            True
        )

        self.setMinimumSize(
            200,
            150,
        )

        self._image: np.ndarray | None = None
        self._pixmap: QPixmap | None = None

        self._scale = 1.0
        self._offset_x = 0.0
        self._offset_y = 0.0

        self._drag_start: QPointF | None = None
        self._drag_current: QPointF | None = None

        self._roi: ROI | None = None

    def set_image_data(
        self,
        image: np.ndarray,
    ) -> None:

        self._image = image

        rgb = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2RGB,
        )

        height, width = rgb.shape[:2]

        bytes_per_line = (
            width * 3
        )

        qimage = QImage(
            rgb.data,
            width,
            height,
            bytes_per_line,
            QImage.Format.Format_RGB888,
        )

        # Make a deep copy because the NumPy array's memory
        # should not be relied upon by the QImage.
        qimage = qimage.copy()

        self._pixmap = QPixmap.fromImage(
            qimage
        )

        self.update()

    def _update_transform(self) -> None:

        if self._pixmap is None:
            return

        image_width = self._pixmap.width()
        image_height = self._pixmap.height()

        canvas_width = self.width()
        canvas_height = self.height()

        if (
            image_width <= 0
            or image_height <= 0
            or canvas_width <= 0
            or canvas_height <= 0
        ):
            return

        scale_x = (
            canvas_width / image_width
        )

        scale_y = (
            canvas_height / image_height
        )

        # IMPORTANT:
        # Use the smaller scale so the entire image fits.
        self._scale = min(
            scale_x,
            scale_y,
        )

        display_width = (
            image_width * self._scale
        )

        display_height = (
            image_height * self._scale
        )

        # Center the image inside the canvas.
        self._offset_x = (
            canvas_width - display_width
        ) / 2.0

        self._offset_y = (
            canvas_height - display_height
        ) / 2.0

    def display_to_image(
        self,
        point: QPointF,
    ) -> tuple[int, int]:

        x = (
            point.x()
            - self._offset_x
        ) / self._scale

        y = (
            point.y()
            - self._offset_y
        ) / self._scale

        x = int(
            round(x)
        )

        y = int(
            round(y)
        )

        if self._image is not None:

            height, width = (
                self._image.shape[:2]
            )

            x = max(
                0,
                min(
                    x,
                    width - 1,
                ),
            )

            y = max(
                0,
                min(
                    y,
                    height - 1,
                ),
            )

        return x, y

    def image_to_display(
        self,
        x: float,
        y: float,
    ) -> QPointF:

        display_x = (
            x * self._scale
            + self._offset_x
        )

        display_y = (
            y * self._scale
            + self._offset_y
        )

        return QPointF(
            display_x,
            display_y,
        )


    def mousePressEvent(
        self,
        event,
    ) -> None:

        if (
            event.button()
            != Qt.MouseButton.LeftButton
        ):
            return

        self._update_transform()

        self._drag_start = (
            event.position()
        )

        self._drag_current = (
            event.position()
        )

        self.update()

    def mouseMoveEvent(
        self,
        event,
    ) -> None:

        if self._drag_start is None:
            return

        self._drag_current = (
            event.position()
        )

        self._update_roi_from_drag()

        self.update()

    def mouseReleaseEvent(
        self,
        event,
    ) -> None:

        if (
            event.button()
            != Qt.MouseButton.LeftButton
        ):
            return

        if self._drag_start is None:
            return

        self._drag_current = (
            event.position()
        )

        self._update_roi_from_drag()

        self._drag_start = None
        self._drag_current = None

        self.update()

    def _update_roi_from_drag(self) -> None:

        if (
            self._drag_start is None
            or
            self._drag_current is None
        ):
            return

        x1, y1 = (
            self.display_to_image(
                self._drag_start
            )
        )

        x2, y2 = (
            self.display_to_image(
                self._drag_current
            )
        )

        x = min(x1, x2)
        y = min(y1, y2)

        width = abs(
            x2 - x1
        )

        height = abs(
            y2 - y1
        )

        if width == 0 or height == 0:

            self._roi = None

        else:

            self._roi = ROI(
                x=x,
                y=y,
                width=width,
                height=height,
            )

        self.roi_changed.emit(
            self._roi
        )

    def paintEvent(
        self,
        event,
    ) -> None:

        painter = QPainter(
            self
        )

        painter.fillRect(
            self.rect(),
            QColor(30, 30, 30),
        )

        if self._pixmap is None:
            painter.end()
            return

        self._update_transform()

        display_width = int(
            self._pixmap.width()
            * self._scale
        )

        display_height = int(
            self._pixmap.height()
            * self._scale
        )

        target_rect = self._pixmap.rect()

        scaled_pixmap = self._pixmap.scaled(
            display_width,
            display_height,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )

        painter.drawPixmap(
            int(self._offset_x),
            int(self._offset_y),
            scaled_pixmap,
        )

        self._draw_grid(
            painter,
            display_width,
            display_height,
        )

        self._draw_roi(
            painter,
        )

        painter.end()

    def _draw_grid(
        self,
        painter: QPainter,
        display_width: int,
        display_height: int,
    ) -> None:

        pen = QPen(
            QColor(255, 255, 255, 70),
        )

        pen.setWidth(1)

        painter.setPen(
            pen
        )

        for i in range(
            1,
            10,
        ):

            x = (
                self._offset_x
                + display_width
                * i
                / 10.0
            )

            painter.drawLine(
                int(x),
                int(self._offset_y),
                int(x),
                int(
                    self._offset_y
                    + display_height
                ),
            )

            y = (
                self._offset_y
                + display_height
                * i
                / 10.0
            )

            painter.drawLine(
                int(self._offset_x),
                int(y),
                int(
                    self._offset_x
                    + display_width
                ),
                int(y),
            )

    def _draw_roi(
        self,
        painter: QPainter,
    ) -> None:

        roi = self._roi

        if roi is None:
            return

        top_left = (
            self.image_to_display(
                roi.x,
                roi.y,
            )
        )

        bottom_right = (
            self.image_to_display(
                roi.x2,
                roi.y2,
            )
        )

        pen = QPen(
            QColor(0, 255, 0),
        )

        pen.setWidth(3)

        painter.setPen(
            pen
        )

        painter.drawRect(
            top_left.x(),
            top_left.y(),
            bottom_right.x()
            - top_left.x(),
            bottom_right.y()
            - top_left.y(),
        )

    def clear_roi(self) -> None:

        self._roi = None

        self._drag_start = None
        self._drag_current = None

        self.roi_changed.emit(
            None
        )

        self.update()

    def resizeEvent(
        self,
        event,
    ) -> None:

        self._update_transform()

        self.update()

        super().resizeEvent(
            event
        )