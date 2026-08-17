from __future__ import annotations

import cv2
import numpy as np
from PySide6.QtCore import Qt, QPointF, Signal, QRectF
from PySide6.QtGui import QPainter, QWheelEvent, QMouseEvent, QImage, QPixmap, QCursor
from PySide6.QtWidgets import QGraphicsView, QGraphicsScene, QFrame, QWidget

from core.roi.coordinate_transform import ImageDisplayTransform


class ImageCanvas(QGraphicsView):
    """
    A reusable graphics view for displaying images with zoom, pan, and coordinate conversion.
    Supports middle-mouse-button panning and wheel zoom.
    """
    # Emits the current cursor position in image coordinates (x, y)
    mouse_coords_changed = Signal(int, int)
    # Emits scene coordinate of left-click
    canvas_clicked = Signal(QPointF)
    
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.scene = QGraphicsScene(self)
        self.setScene(self.scene)

        # Rendering options
        self.setRenderHint(QPainter.RenderHint.Antialiasing)
        self.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setFrameShape(QFrame.Shape.NoFrame)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)

        # Image state
        self._pixmap_item = None
        self._image_width = 0
        self._image_height = 0
        self._pan_active = False
        self._last_pan_pos = None
        
        self.zoom_factor = 1.15

    def set_image(self, bgr_image: np.ndarray) -> None:
        """
        Loads a BGR numpy image, converts to RGB, and adds to scene.
        """
        self.scene.clear()
        self._pixmap_item = None

        if bgr_image is None or bgr_image.size == 0:
            self._image_width = 0
            self._image_height = 0
            return

        self._image_height, self._image_width = bgr_image.shape[:2]

        # Convert BGR to RGB
        rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
        bytes_per_line = self._image_width * 3
        
        qimage = QImage(
            rgb_image.data,
            self._image_width,
            self._image_height,
            bytes_per_line,
            QImage.Format.Format_RGB888
        ).copy()

        pixmap = QPixmap.fromImage(qimage)
        self._pixmap_item = self.scene.addPixmap(pixmap)
        
        # Set scene bounds to fit the image
        self.scene.setSceneRect(QRectF(0, 0, self._image_width, self._image_height))
        self.zoom_to_fit()

    def zoom_to_fit(self) -> None:
        """
        Fits the entire image inside the viewport.
        """
        if self._image_width > 0:
            self.fitInView(self.scene.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)

    def zoom_to_rect(self, rect: QRectF) -> None:
        """
        Zooms to a specific region inside the scene.
        """
        self.fitInView(rect, Qt.AspectRatioMode.KeepAspectRatio)

    def wheelEvent(self, event: QWheelEvent) -> None:
        """
        Handles mouse wheel zoom centered under mouse.
        """
        if self._image_width == 0:
            return

        # Calculate zoom scaling factor
        factor = self.zoom_factor if event.angleDelta().y() > 0 else (1.0 / self.zoom_factor)
        
        # Apply transformation
        self.scale(factor, factor)

    def mousePressEvent(self, event: QMouseEvent) -> None:
        """
        Initiates panning on middle mouse click, and emits canvas_clicked on left click.
        """
        if event.button() == Qt.MouseButton.MiddleButton:
            self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
            # Create a simulated left mouse event to start dragging
            fake_event = QMouseEvent(
                event.type(),
                event.position(),
                Qt.MouseButton.LeftButton,
                event.buttons() | Qt.MouseButton.LeftButton,
                event.modifiers()
            )
            super().mousePressEvent(fake_event)
            self._pan_active = True
        elif event.button() == Qt.MouseButton.LeftButton:
            scene_pos = self.mapToScene(event.position().toPoint())
            self.canvas_clicked.emit(scene_pos)
            super().mousePressEvent(event)
        else:
            super().mousePressEvent(event)

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        """
        Ends panning.
        """
        if event.button() == Qt.MouseButton.MiddleButton and self._pan_active:
            fake_event = QMouseEvent(
                event.type(),
                event.position(),
                Qt.MouseButton.LeftButton,
                event.buttons() & ~Qt.MouseButton.LeftButton,
                event.modifiers()
            )
            super().mouseReleaseEvent(fake_event)
            self.setDragMode(QGraphicsView.DragMode.NoDrag)
            self._pan_active = False
        else:
            super().mouseReleaseEvent(event)

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        """
        Emits coordinates under cursor in image space.
        """
        super().mouseMoveEvent(event)
        
        scene_pos = self.mapToScene(event.position().toPoint())
        x = int(scene_pos.x())
        y = int(scene_pos.y())
        
        # Constrain to image boundaries
        x = max(0, min(x, self._image_width - 1))
        y = max(0, min(y, self._image_height - 1))
        
        self.mouse_coords_changed.emit(x, y)
