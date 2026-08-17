from __future__ import annotations

import cv2
import numpy as np
from PySide6.QtCore import Qt, QPointF, QRectF, QTimer
from PySide6.QtGui import QColor, QPen, QPolygonF, QPainterPath
from PySide6.QtWidgets import (
    QDialog,
    QHBoxLayout,
    QVBoxLayout,
    QPushButton,
    QLabel,
    QMessageBox,
    QWidget,
    QGraphicsItem,
    QGraphicsEllipseItem,
    QGraphicsPolygonItem,
    QGraphicsPathItem,
    QButtonGroup
)

from core.models import ImageData, ROI
from core.ui.image_canvas import ImageCanvas


class VertexHandle(QGraphicsEllipseItem):
    """
    A draggable handle representing a vertex of the editable contour.
    Coordinates are in scene (original image) space.
    """
    def __init__(self, idx: int, pos: QPointF, parent_editor: ContourEditor) -> None:
        # 6 pixel radius handle circle
        r = 6.0
        super().__init__(-r, -r, 2 * r, 2 * r)
        self.idx = idx
        self.parent_editor = parent_editor

        self.setPos(pos)
        self.setFlags(
            QGraphicsItem.GraphicsItemFlag.ItemIsMovable |
            QGraphicsItem.GraphicsItemFlag.ItemSendsGeometryChanges
        )

        # Style
        self.setBrush(QColor(0, 255, 0)) # Green fill
        self.setPen(QPen(QColor(255, 255, 255), 1.5)) # White outline
        self.setAcceptHoverEvents(True)

    def hoverEnterEvent(self, event) -> None:
        self.setBrush(QColor(255, 165, 0)) # Orange highlight
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        super().hoverEnterEvent(event)

    def hoverLeaveEvent(self, event) -> None:
        self.setBrush(QColor(0, 255, 0))
        self.unsetCursor()
        super().hoverLeaveEvent(event)

    def mousePressEvent(self, event) -> None:
        if self.parent_editor.mode == "delete":
            # Direct delete action in Delete Mode
            self.parent_editor.delete_vertex(self.idx)
            event.accept()
        else:
            super().mousePressEvent(event)

    def itemChange(self, change, value):
        if change == QGraphicsItem.GraphicsItemChange.ItemPositionChange and self.parent_editor:
            new_pos = value
            img_h, img_w = self.parent_editor.image_data.image.shape[:2]
            
            # Constrain vertex handle within image boundaries
            cx = max(0.0, min(new_pos.x(), float(img_w - 1)))
            cy = max(0.0, min(new_pos.y(), float(img_h - 1)))
            clamped = QPointF(cx, cy)
            
            self.parent_editor.update_vertex(self.idx, clamped)
            return clamped

        return super().itemChange(change, value)


class ContourEditor(QDialog):
    """
    Dialog for manual editing of the target silhouette.
    Supports Select/Move, Add Point, and Delete Point modes with zoom and pan.
    """
    def __init__(
        self,
        image_data: ImageData,
        roi: ROI,
        contour: np.ndarray,
        parent: QWidget | None = None
    ) -> None:
        super().__init__(parent)
        self.image_data = image_data
        self.roi = roi
        
        # Simplify contour for editing while keeping features ( Douglas-Peucker epsilon=1.0 )
        simplified_roi_local = self._simplify_contour(contour)
        
        # Convert contour coordinates from ROI-local to whole-image space
        self.initial_contour = simplified_roi_local + np.array([[[roi.x, roi.y]]], dtype=np.int32)
        self.current_contour = self.initial_contour.copy()
        self.final_contour: np.ndarray | None = None

        # Mode state: "select", "add", "delete"
        self.mode = "select"

        # Item tracking
        self._handles: list[VertexHandle] = []
        self._line_item: QGraphicsPolygonItem | None = None
        self._ref_contour_item: QGraphicsPolygonItem | None = None

        self._setup_ui()
        self._setup_overlays()
        self._init_contours()
        self._update_status()

    def _simplify_contour(self, contour: np.ndarray) -> np.ndarray:
        """
        Reduces number of vertices for easy editing while preserving shape accuracy.
        """
        if len(contour) <= 4:
            return contour
        # Use cv2.approxPolyDP with a 1.0 pixel threshold
        simplified = cv2.approxPolyDP(contour, 1.0, closed=True)
        return simplified

    def _setup_ui(self) -> None:
        self.setWindowTitle("Blueprint Author — Edit Silhouette Contour")
        self.resize(1100, 750)
        self.setMinimumSize(800, 600)

        # Main Layout
        layout = QVBoxLayout(self)

        # Toolbar
        toolbar_layout = QHBoxLayout()
        
        self.select_btn = QPushButton("Select / Move")
        self.select_btn.setCheckable(True)
        self.select_btn.setChecked(True)
        self.select_btn.clicked.connect(lambda: self._set_mode("select"))

        self.add_btn = QPushButton("Add Point")
        self.add_btn.setCheckable(True)
        self.add_btn.clicked.connect(lambda: self._set_mode("add"))

        self.add_multiple_btn = QPushButton("Add Multiple Points")
        self.add_multiple_btn.setCheckable(True)
        self.add_multiple_btn.clicked.connect(lambda: self._set_mode("add_multiple"))

        self.delete_btn = QPushButton("Delete Point")
        self.delete_btn.setCheckable(True)
        self.delete_btn.clicked.connect(lambda: self._set_mode("delete"))

        # Make mode buttons exclusive
        self.mode_group = QButtonGroup(self)
        self.mode_group.addButton(self.select_btn)
        self.mode_group.addButton(self.add_btn)
        self.mode_group.addButton(self.add_multiple_btn)
        self.mode_group.addButton(self.delete_btn)
        self.mode_group.setExclusive(True)

        self.reset_btn = QPushButton("Reset Contour")
        self.reset_btn.clicked.connect(self.reset_contour)

        toolbar_layout.addWidget(self.select_btn)
        toolbar_layout.addWidget(self.add_btn)
        toolbar_layout.addWidget(self.add_multiple_btn)
        toolbar_layout.addWidget(self.delete_btn)
        toolbar_layout.addSpacing(20)
        toolbar_layout.addWidget(self.reset_btn)
        toolbar_layout.addStretch()
        layout.addLayout(toolbar_layout)

        # Image Canvas
        self.canvas = ImageCanvas(self)
        self.canvas.set_image(self.image_data.image)
        self.canvas.mouse_coords_changed.connect(self._on_mouse_coords_changed)
        self.canvas.canvas_clicked.connect(self._on_canvas_clicked)
        layout.addWidget(self.canvas, stretch=1)

        # Bottom Bar (Status + OK/Cancel)
        bottom_layout = QHBoxLayout()
        
        self.status_label = QLabel()
        bottom_layout.addWidget(self.status_label, stretch=1)

        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.clicked.connect(self.reject)
        
        self.confirm_button = QPushButton("Confirm")
        self.confirm_button.setDefault(True)
        self.confirm_button.clicked.connect(self.confirm_contour)

        bottom_layout.addWidget(self.cancel_button)
        bottom_layout.addWidget(self.confirm_button)
        layout.addLayout(bottom_layout)

        self.setLayout(layout)

    def _setup_overlays(self) -> None:
        """
        Highlights the ROI by dimming the surrounding background.
        """
        scene = self.canvas.scene
        img_h, img_w = self.image_data.image.shape[:2]

        # 1. Dim outside ROI
        full_path = QPainterPath()
        full_path.addRect(0, 0, img_w, img_h)
        
        roi_path = QPainterPath()
        roi_path.addRect(self.roi.x, self.roi.y, self.roi.width, self.roi.height)
        
        outside_path = full_path.subtracted(roi_path)
        
        outside_item = QGraphicsPathItem(outside_path)
        outside_item.setBrush(QColor(0, 0, 0, 140))
        outside_item.setPen(Qt.PenStyle.NoPen)
        scene.addItem(outside_item)



    def zoom_to_roi(self) -> None:
        """
        Centering/zooming of the canvas viewport on the ROI with margin.
        """
        img_h, img_w = self.image_data.image.shape[:2]
        margin = 80
        zoom_rect = QRectF(
            self.roi.x - margin,
            self.roi.y - margin,
            self.roi.width + 2 * margin,
            self.roi.height + 2 * margin
        )
        zoom_rect = zoom_rect.intersected(QRectF(0, 0, img_w, img_h))
        self.canvas.zoom_to_rect(zoom_rect)

    def showEvent(self, event) -> None:
        super().showEvent(event)
        # Use single-shot timer to ensure the window geometry settles before fitInView calculations
        QTimer.singleShot(50, self.zoom_to_roi)

    def _init_contours(self) -> None:
        scene = self.canvas.scene

        # 1. Draw the static blue reference contour (whole-image space)
        ref_poly = QPolygonF()
        for pt in self.initial_contour:
            pt_data = pt.reshape(-1)
            ref_poly.append(QPointF(pt_data[0], pt_data[1]))
        
        self._ref_contour_item = QGraphicsPolygonItem(ref_poly)
        self._ref_contour_item.setPen(QPen(QColor(0, 120, 255), 1.5, Qt.PenStyle.DashLine))
        scene.addItem(self._ref_contour_item)

        # 2. Draw the interactive green user contour (whole-image space)
        self._line_item = QGraphicsPolygonItem()
        self._line_item.setPen(QPen(QColor(0, 255, 0), 2.5))
        self._line_item.setBrush(QColor(0, 255, 0, 25)) # Light fill
        scene.addItem(self._line_item)

        self.recreate_handles()
        self.redraw_contour_line()

    def recreate_handles(self) -> None:
        """
        Clears and rebuilds draggable vertex handles.
        """
        scene = self.canvas.scene

        # Remove old handles
        for handle in self._handles:
            scene.removeItem(handle)
        self._handles.clear()

        # Add new handles for current contour coordinates (whole-image space)
        for idx, (x, y) in enumerate(self.current_contour.reshape(-1, 2)):
            scene_pos = QPointF(x, y)
            handle = VertexHandle(idx, scene_pos, self)
            scene.addItem(handle)
            self._handles.append(handle)

    def redraw_contour_line(self) -> None:
        """
        Updates the green editable line representing the current contour.
        """
        if self._line_item is None:
            return
        
        qpoly = QPolygonF()
        for (x, y) in self.current_contour.reshape(-1, 2):
            qpoly.append(QPointF(x, y))
        
        self._line_item.setPolygon(qpoly)

    def update_vertex(self, idx: int, scene_pos: QPointF) -> None:
        """
        Updates a specific vertex coordinate based on handle drag.
        """
        # Coordinates are already in scene (whole-image) space
        wx = int(round(scene_pos.x()))
        wy = int(round(scene_pos.y()))

        # Clamp inside ROI bounds (in whole-image space)
        wx = max(self.roi.x, min(wx, self.roi.x2 - 1))
        wy = max(self.roi.y, min(wy, self.roi.y2 - 1))

        self.current_contour[idx] = [wx, wy]
        self.redraw_contour_line()
        self._update_status()

    def add_vertex(self, scene_pos: QPointF) -> None:
        """
        Calculates projection point and inserts a vertex on the closest segment.
        """
        wx = scene_pos.x()
        wy = scene_pos.y()

        # Only allow additions within the ROI margins (in whole-image coordinates)
        if not (self.roi.x - 20 <= wx < self.roi.x2 + 20 and self.roi.y - 20 <= wy < self.roi.y2 + 20):
            return

        pts = self.current_contour.reshape(-1, 2)
        best_idx, best_proj = self._find_nearest_segment_insertion(pts, (wx, wy))

        # Clamp projected coordinate to ROI bounds (in whole-image space)
        new_x = int(round(max(float(self.roi.x), min(best_proj[0], float(self.roi.x2 - 1)))))
        new_y = int(round(max(float(self.roi.y), min(best_proj[1], float(self.roi.y2 - 1)))))
        new_pt = np.array([new_x, new_y])

        # Insert new point
        pts = np.insert(pts, best_idx, new_pt, axis=0)
        self.current_contour = pts.reshape(-1, 1, 2)

        self.recreate_handles()
        self.redraw_contour_line()
        self._update_status()

    def delete_vertex(self, idx: int) -> None:
        """
        Deletes the vertex at index idx.
        """
        pts = self.current_contour.reshape(-1, 2)
        if len(pts) <= 3:
            QMessageBox.warning(self, "Invalid Action", "Contours must have at least 3 vertices.")
            return

        pts = np.delete(pts, idx, axis=0)
        self.current_contour = pts.reshape(-1, 1, 2)

        self.recreate_handles()
        self.redraw_contour_line()
        self._update_status()

    def _find_nearest_segment_insertion(
        self, pts: np.ndarray, click_pt: tuple[float, float]
    ) -> tuple[int, tuple[float, float]]:
        cx, cy = click_pt
        N = len(pts)
        
        min_dist = float('inf')
        best_idx = 0
        best_proj = (cx, cy)
        
        for i in range(N):
            p1 = pts[i]
            p2 = pts[(i + 1) % N]
            
            vx = p2[0] - p1[0]
            vy = p2[1] - p1[1]
            
            wx = cx - p1[0]
            wy = cy - p1[1]
            
            seg_len_sq = vx*vx + vy*vy
            if seg_len_sq == 0:
                t = 0.0
            else:
                t = (wx*vx + wy*vy) / seg_len_sq
                t = max(0.0, min(1.0, t))
                
            px = p1[0] + t * vx
            py = p1[1] + t * vy
            
            dist_sq = (cx - px)**2 + (cy - py)**2
            
            if dist_sq < min_dist:
                min_dist = dist_sq
                best_idx = i + 1
                best_proj = (px, py)
                
        return best_idx, best_proj

    def _set_mode(self, mode: str) -> None:
        self.mode = mode
        self._update_status()

    def reset_contour(self) -> None:
        reply = QMessageBox.question(
            self,
            "Reset Contour",
            "Are you sure you want to discard your edits and reset the contour?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        if reply == QMessageBox.StandardButton.Yes:
            self.current_contour = self.initial_contour.copy()
            self.recreate_handles()
            self.redraw_contour_line()
            self._update_status()

    def _on_mouse_coords_changed(self, x: int, y: int) -> None:
        self._update_status(mouse_pos=(x, y))

    def _on_canvas_clicked(self, scene_pos: QPointF) -> None:
        if self.mode == "add":
            self.add_vertex(scene_pos)
            # Auto-revert to Select mode
            self.select_btn.setChecked(True)
            self._set_mode("select")
        elif self.mode == "add_multiple":
            self.add_vertex(scene_pos)

    def _update_status(self, mouse_pos: tuple[int, int] | None = None) -> None:
        if self.mode == "add_multiple":
            mode_str = "Add Multiple Points"
        else:
            mode_str = self.mode.capitalize()
            
        pt_count = len(self.current_contour)
        
        pos_str = ""
        if mouse_pos is not None:
            # Check if mouse is inside ROI
            rx = mouse_pos[0] - self.roi.x
            ry = mouse_pos[1] - self.roi.y
            if 0 <= rx < self.roi.width and 0 <= ry < self.roi.height:
                pos_str = f" | Cursor (ROI-local): x={rx}, y={ry}"
            else:
                pos_str = f" | Cursor: x={mouse_pos[0]}, y={mouse_pos[1]}"
                
        self.status_label.setText(
            f"<b>Mode:</b> {mode_str} | <b>Vertices:</b> {pt_count}{pos_str}"
        )

    def confirm_contour(self) -> None:
        """
        Validates the edited contour shape before closing.
        """
        pts = self.current_contour.reshape(-1, 2)
        if len(pts) < 3:
            QMessageBox.critical(self, "Validation Error", "Contour must have at least 3 vertices.")
            return

        # Calculate area using cv2.contourArea
        area = cv2.contourArea(self.current_contour)
        if area <= 0.0:
            QMessageBox.critical(self, "Validation Error", "Contour has zero or invalid area.")
            return

        # Success: store the contour in ROI-local coordinates (matching input)
        self.final_contour = self.current_contour
        self.accept()
