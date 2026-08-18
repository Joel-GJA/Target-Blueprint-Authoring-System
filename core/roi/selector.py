from __future__ import annotations

import cv2
import numpy as np

from PySide6.QtCore import QPointF, Qt, Signal
from PySide6.QtGui import (
    QColor,
    QGuiApplication,
    QImage,
    QPainter,
    QPen,
    QPixmap,
)
from PySide6.QtWidgets import (
    QDialog,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from core.models import ImageData, ROI
from core.roi.coordinate_transform import ImageDisplayTransform


class ROICanvas(QWidget):
    """
    Canvas responsible for displaying the image and handling ROI interaction.

    Mouse coordinates are display coordinates.
    ROI coordinates are converted immediately into original-image coordinates.
    """

    roi_changed = Signal(object)

    def __init__(
        self,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)

        self.setMouseTracking(True)
        self.setMinimumSize(200, 150)

        self._image: np.ndarray | None = None
        self._pixmap: QPixmap | None = None
        self._transform: ImageDisplayTransform | None = None

        self._drag_start: QPointF | None = None
        self._drag_current: QPointF | None = None
        self._roi: ROI | None = None

    def set_image_data(
        self,
        image: np.ndarray,
    ) -> None:
        """Set the source image to display and interact with."""
        self._image = image

        rgb = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2RGB,
        )

        height, width = rgb.shape[:2]
        bytes_per_line = width * 3

        qimage = QImage(
            rgb.data,
            width,
            height,
            bytes_per_line,
            QImage.Format.Format_RGB888,
        )

        # Make a deep copy to ensure the NumPy array's memory
        # is not relied upon by the QImage.
        qimage = qimage.copy()

        self._pixmap = QPixmap.fromImage(qimage)

        # Instantiate the coordinate transformation utility
        self._transform = ImageDisplayTransform(
            image_width=width,
            image_height=height,
        )

        self._update_transform()
        self.update()

    def _update_transform(self) -> None:
        if self._transform is None:
            return

        self._transform.update(
            self.width(),
            self.height(),
        )

    def display_to_image(
        self,
        point: QPointF,
    ) -> tuple[int, int]:
        if self._transform is None:
            return 0, 0
        return self._transform.display_to_image(point)

    def image_to_display(
        self,
        x: float,
        y: float,
    ) -> QPointF:
        if self._transform is None:
            return QPointF(0, 0)
        return self._transform.image_to_display(x, y)

    def mousePressEvent(
        self,
        event,
    ) -> None:
        if event.button() != Qt.MouseButton.LeftButton:
            return

        self._update_transform()

        self._drag_start = event.position()
        self._drag_current = event.position()

        self.update()

    def mouseMoveEvent(
        self,
        event,
    ) -> None:
        if self._drag_start is None:
            return

        self._drag_current = event.position()
        self._update_roi_from_drag()
        self.update()

    def mouseReleaseEvent(
        self,
        event,
    ) -> None:
        if event.button() != Qt.MouseButton.LeftButton:
            return

        if self._drag_start is None:
            return

        self._drag_current = event.position()
        self._update_roi_from_drag()

        self._drag_start = None
        self._drag_current = None

        self.update()

    def _update_roi_from_drag(self) -> None:
        if (
            self._drag_start is None
            or self._drag_current is None
        ):
            return

        x1, y1 = self.display_to_image(self._drag_start)
        x2, y2 = self.display_to_image(self._drag_current)

        x = min(x1, x2)
        y = min(y1, y2)

        width = abs(x2 - x1)
        height = abs(y2 - y1)

        if width == 0 or height == 0:
            self._roi = None
        else:
            self._roi = ROI(
                x=x,
                y=y,
                width=width,
                height=height,
            )

        self.roi_changed.emit(self._roi)

    def paintEvent(
        self,
        event,
    ) -> None:
        painter = QPainter(self)

        painter.fillRect(
            self.rect(),
            QColor(30, 30, 30),
        )

        if self._pixmap is None or self._transform is None:
            painter.end()
            return

        self._update_transform()

        display_width = int(
            self._pixmap.width()
            * self._transform.scale
        )

        display_height = int(
            self._pixmap.height()
            * self._transform.scale
        )

        scaled_pixmap = self._pixmap.scaled(
            display_width,
            display_height,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )

        painter.drawPixmap(
            int(self._transform.offset_x),
            int(self._transform.offset_y),
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
        if self._transform is None:
            return

        pen = QPen(
            QColor(255, 255, 255, 70),
        )
        pen.setWidth(1)
        painter.setPen(pen)

        for i in range(1, 10):
            x = (
                self._transform.offset_x
                + display_width * i / 10.0
            )

            painter.drawLine(
                int(x),
                int(self._transform.offset_y),
                int(x),
                int(self._transform.offset_y + display_height),
            )

            y = (
                self._transform.offset_y
                + display_height * i / 10.0
            )

            painter.drawLine(
                int(self._transform.offset_x),
                int(y),
                int(self._transform.offset_x + display_width),
                int(y),
            )

    def _draw_roi(
        self,
        painter: QPainter,
    ) -> None:
        roi = self._roi

        if roi is None:
            return

        top_left = self.image_to_display(roi.x, roi.y)
        bottom_right = self.image_to_display(roi.x2, roi.y2)

        pen = QPen(
            QColor(0, 255, 0),
        )
        pen.setWidth(3)
        painter.setPen(pen)

        painter.drawRect(
            int(top_left.x()),
            int(top_left.y()),
            int(bottom_right.x() - top_left.x()),
            int(bottom_right.y() - top_left.y()),
        )

    def clear_roi(self) -> None:
        self._roi = None
        self._drag_start = None
        self._drag_current = None
        self.roi_changed.emit(None)
        self.update()

    def resizeEvent(
        self,
        event,
    ) -> None:
        self._update_transform()
        self.update()
        super().resizeEvent(event)


class ROISelector(QDialog):
    """
    Interactive ROI selector dialog for a high-resolution image.

    This dialog wraps the ROICanvas and offers Reset, Confirm, and Cancel buttons.
    """

    roi_confirmed = Signal(object)

    MIN_ROI_WIDTH = 50
    MIN_ROI_HEIGHT = 50

    def __init__(
        self,
        image_data: ImageData,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)

        self.image_data = image_data
        self.image = image_data.image

        self.image_height, self.image_width = self.image.shape[:2]

        self.roi: ROI | None = None

        # Physical specs parameters and background detection
        self.tag_size_mm: float = 24.0
        self.target_width_mm: float = 210.0
        self.target_height_mm: float = 297.0
        self.detected_tags = None

        self._tag_thread_result = None
        import threading
        self._tag_thread = threading.Thread(target=self._run_bg_apriltag_detection)
        self._tag_thread.daemon = True
        self._tag_thread.start()

        self._setup_ui()

    def _run_bg_apriltag_detection(self) -> None:
        try:
            from core.detection.apriltag_detector import AprilTagDetector
            detector = AprilTagDetector(families="tag36h11", nthreads=4)
            self._tag_thread_result = detector.detect(self.image_data)
        except Exception as e:
            print(f"Background AprilTag detection error: {e}")

    def _setup_ui(self) -> None:
        self.setWindowTitle("Blueprint Author — Select Target ROI")

        self.setMinimumSize(600, 450)

        screen = QGuiApplication.primaryScreen()
        if screen is not None:
            available = screen.availableGeometry()
            self.resize(
                int(available.width() * 0.80),
                int(available.height() * 0.80),
            )
        else:
            self.resize(1200, 900)

        # Image canvas
        self.canvas = ROICanvas(self)
        self.canvas.set_image_data(self.image)
        self.canvas.roi_changed.connect(self._on_roi_changed)

        # Status label
        self.status_label = QLabel("Drag a rectangle around the target.")
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        # Buttons
        self.reset_button = QPushButton("Reset")
        self.reset_button.clicked.connect(self.reset_roi)

        self.confirm_button = QPushButton("Confirm")
        self.confirm_button.clicked.connect(self.confirm_roi)

        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.clicked.connect(self.reject)

        button_layout = QHBoxLayout()
        button_layout.addStretch()
        button_layout.addWidget(self.reset_button)
        button_layout.addWidget(self.cancel_button)
        button_layout.addWidget(self.confirm_button)

        # Main layout
        layout = QVBoxLayout(self)
        layout.addWidget(
            self.canvas,
            stretch=1,
        )
        layout.addWidget(self.status_label)
        layout.addLayout(button_layout)
        self.setLayout(layout)

    def _on_roi_changed(
        self,
        roi: ROI | None,
    ) -> None:
        self.roi = roi

        if roi is None:
            self.status_label.setText("Drag a rectangle around the target.")
            return

        self.status_label.setText(
            f"ROI: x={roi.x}, y={roi.y}, w={roi.width}, h={roi.height}"
        )

    def reset_roi(self) -> None:
        self.roi = None
        self.canvas.clear_roi()
        self.status_label.setText("Drag a rectangle around the target.")

    def confirm_roi(self) -> None:
        if self.roi is None:
            self.status_label.setText("Please draw an ROI first.")
            return

        if (
            self.roi.width < self.MIN_ROI_WIDTH
            or self.roi.height < self.MIN_ROI_HEIGHT
        ):
            self.status_label.setText("ROI is too small.")
            return

        # 1. Wait for background AprilTag detection thread to complete
        self.status_label.setText("Waiting for background AprilTag detection...")
        QGuiApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        if self._tag_thread and self._tag_thread.is_alive():
            self._tag_thread.join(timeout=4.0)
        QGuiApplication.restoreOverrideCursor()

        # 2. Prompt for target physical specifications
        specs_dialog = PhysicalSpecsDialog(self)
        if specs_dialog.exec() == QDialog.DialogCode.Accepted:
            tag_size, width, height = specs_dialog.get_specs()
            self.tag_size_mm = tag_size
            self.target_width_mm = width
            self.target_height_mm = height
            self.detected_tags = self._tag_thread_result
            
            self.roi_confirmed.emit(self.roi)
            self.accept()
        else:
            self.status_label.setText("Physical specs entry cancelled.")


class PhysicalSpecsDialog(QDialog):
    """
    Dialog asking for physical dimensions: AprilTag size, target page width, and target page height.
    """
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Target Physical Specifications")
        self.resize(350, 180)
        
        from PySide6.QtWidgets import QFormLayout, QLineEdit, QDialogButtonBox
        layout = QFormLayout(self)
        
        self.tag_size_edit = QLineEdit(self)
        self.tag_size_edit.setText("24.0")
        layout.addRow("AprilTag Size (mm):", self.tag_size_edit)
        
        self.width_edit = QLineEdit(self)
        self.width_edit.setText("210.0")
        layout.addRow("Target Page Width (mm):", self.width_edit)
        
        self.height_edit = QLineEdit(self)
        self.height_edit.setText("297.0")
        layout.addRow("Target Page Height (mm):", self.height_edit)
        
        self.buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel,
            Qt.Orientation.Horizontal,
            self
        )
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)
        layout.addRow(self.buttons)
        
    def get_specs(self) -> tuple[float, float, float]:
        try:
            tag_size = float(self.tag_size_edit.text())
        except ValueError:
            tag_size = 24.0
            
        try:
            width = float(self.width_edit.text())
        except ValueError:
            width = 210.0
            
        try:
            height = float(self.height_edit.text())
        except ValueError:
            height = 297.0
            
        return tag_size, width, height
