from __future__ import annotations

import cv2
import numpy as np
from PySide6.QtCore import Qt, QPointF, QRectF, QTimer
from PySide6.QtGui import QColor, QPen, QPolygonF, QPainterPath
from PySide6.QtWidgets import (
    QDialog,
    QHBoxLayout,
    QVBoxLayout,
    QListWidget,
    QListWidgetItem,
    QLabel,
    QPushButton,
    QTextBrowser,
    QWidget,
    QGraphicsPathItem,
    QGraphicsPolygonItem,
    QGraphicsRectItem
)

from core.models import ImageData, ROI
from core.ui.image_canvas import ImageCanvas


class SilhouetteCandidateSelector(QDialog):
    """
    Dialog for choosing the initial silhouette contour from either the 
    ranked Hessian+HSV candidates or the white-mask contour.
    """
    def __init__(
        self,
        image_data: ImageData,
        roi: ROI,
        candidates: list,
        white_contour: np.ndarray | None,
        parent: QWidget | None = None
    ) -> None:
        super().__init__(parent)
        self.image_data = image_data
        self.roi = roi
        self.candidates = candidates
        self.white_contour = white_contour

        self.selected_contour: np.ndarray | None = None
        self._active_contour_item: QGraphicsPolygonItem | None = None

        self._setup_ui()
        self._load_candidates()

    def _setup_ui(self) -> None:
        self.setWindowTitle("Blueprint Author — Select Target Silhouette")
        self.resize(1100, 750)
        self.setMinimumSize(800, 600)

        # Main horizontal layout
        main_layout = QHBoxLayout(self)

        # Left: Interactive Image Canvas
        self.canvas = ImageCanvas(self)
        self.canvas.set_image(self.image_data.image)
        main_layout.addWidget(self.canvas, stretch=3)

        # Right: Candidate Selection Sidebar
        sidebar = QWidget(self)
        sidebar_layout = QVBoxLayout(sidebar)
        sidebar_layout.setContentsMargins(10, 0, 10, 0)
        sidebar.setMaximumWidth(320)

        sidebar_layout.addWidget(QLabel("<b>Available Candidate Interpretations:</b>"))
        
        self.list_widget = QListWidget(self)
        self.list_widget.currentItemChanged.connect(self._on_candidate_changed)
        sidebar_layout.addWidget(self.list_widget)

        # Details Panel
        sidebar_layout.addWidget(QLabel("<b>Candidate Statistics:</b>"))
        self.details_box = QTextBrowser(self)
        self.details_box.setMaximumHeight(180)
        sidebar_layout.addWidget(self.details_box)

        # Action Buttons
        button_layout = QHBoxLayout()
        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.clicked.connect(self.reject)
        
        self.edit_button = QPushButton("Edit Contour")
        self.edit_button.setDefault(True)
        self.edit_button.clicked.connect(self.accept_selection)
        
        button_layout.addWidget(self.cancel_button)
        button_layout.addWidget(self.edit_button)
        sidebar_layout.addLayout(button_layout)

        main_layout.addWidget(sidebar, stretch=1)
        self.setLayout(main_layout)

        # Set up ROI overlay outside region dimming and ROI borders
        self._setup_roi_overlays()

    def _setup_roi_overlays(self) -> None:
        scene = self.canvas.scene
        img_h, img_w = self.image_data.image.shape[:2]

        # 1. Dim the region outside the ROI using a QGraphicsPathItem
        full_path = QPainterPath()
        full_path.addRect(0, 0, img_w, img_h)
        
        roi_path = QPainterPath()
        roi_path.addRect(self.roi.x, self.roi.y, self.roi.width, self.roi.height)
        
        outside_path = full_path.subtracted(roi_path)
        
        outside_item = QGraphicsPathItem(outside_path)
        outside_item.setBrush(QColor(0, 0, 0, 140)) # Dimmed background
        outside_item.setPen(Qt.PenStyle.NoPen)
        scene.addItem(outside_item)



    def zoom_to_roi(self) -> None:
        """
        Centering/zooming of the canvas viewport on the ROI with margin.
        """
        img_h, img_w = self.image_data.image.shape[:2]
        margin = 100
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

    def _load_candidates(self) -> None:
        """
        Populate the list widget with the available candidate options.
        """
        # Load White Mask Contour first so it is the default choice
        if self.white_contour is not None:
            item = QListWidgetItem("White Mask Contour")
            item.setData(Qt.ItemDataRole.UserRole, {"type": "white_mask"})
            self.list_widget.addItem(item)

        # Load ranked Hessian+HSV candidates next
        for idx, candidate in enumerate(self.candidates[:5]):
            item = QListWidgetItem(f"Candidate #{idx + 1} (Hessian+HSV)")
            # Store the index in user data role
            item.setData(Qt.ItemDataRole.UserRole, {"type": "candidate", "index": idx})
            self.list_widget.addItem(item)

        # Default select the first item if available
        if self.list_widget.count() > 0:
            self.list_widget.setCurrentRow(0)

    def _on_candidate_changed(self, current: QListWidgetItem, previous: QListWidgetItem) -> None:
        if current is None:
            return

        data = current.data(Qt.ItemDataRole.UserRole)
        scene = self.canvas.scene

        # Clear existing active contour
        if self._active_contour_item is not None:
            scene.removeItem(self._active_contour_item)
            self._active_contour_item = None

        contour = None
        details = ""

        if data["type"] == "candidate":
            idx = data["index"]
            cand = self.candidates[idx]
            contour = cand.contour
            color = QColor(0, 255, 0) # Green for candidates
            
            # Populate details
            details = (
                f"<b>Type:</b> Hessian + HSV Candidate #{idx + 1}<br>"
                f"<b>Overall Score:</b> {cand.score:.4f}<br>"
                f"<b>Hessian Support:</b> {cand.hessian_support:.4f}<br>"
                f"<b>Inside Target Support:</b> {cand.inside_target_support:.4f}<br>"
                f"<b>Outside White Support:</b> {cand.outside_white_support:.4f}<br>"
                f"<b>Solidity:</b> {cand.solidity:.4f}<br>"
                f"<b>Perimeter:</b> {cand.perimeter:.1f} px<br>"
                f"<b>Area Ratio:</b> {cand.area_ratio:.4f}"
            )
        elif data["type"] == "white_mask":
            contour = self.white_contour
            color = QColor(0, 120, 255) # Blue for white mask
            
            area = cv2.contourArea(contour)
            perimeter = cv2.arcLength(contour, closed=True)
            details = (
                f"<b>Type:</b> Inverted White Mask Contour<br>"
                f"<b>Description:</b> Outer boundary extracted from the white border's inner black region.<br>"
                f"<b>Area:</b> {area:.1f} px²<br>"
                f"<b>Perimeter:</b> {perimeter:.1f} px"
            )

        self.selected_contour = contour
        self.details_box.setHtml(details)

        # Draw the selected contour on the image (offset by ROI coords)
        if contour is not None:
            qpoly = QPolygonF()
            for pt in contour:
                pt_data = pt.reshape(-1)
                # Offset by ROI coordinate to display correctly on the full image
                qpoly.append(QPointF(pt_data[0] + self.roi.x, pt_data[1] + self.roi.y))

            self._active_contour_item = QGraphicsPolygonItem(qpoly)
            # Semi-transparent fill and bold outline
            fill_color = QColor(color.red(), color.green(), color.blue(), 30)
            self._active_contour_item.setBrush(fill_color)
            self._active_contour_item.setPen(QPen(color, 3, Qt.PenStyle.SolidLine))
            scene.addItem(self._active_contour_item)

    def accept_selection(self) -> None:
        if self.selected_contour is None:
            self.details_box.setHtml("<font color='red'><b>Please select a contour first.</b></font>")
            return
        self.accept()
