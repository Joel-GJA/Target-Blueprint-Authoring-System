from __future__ import annotations

import cv2
import numpy as np
from PySide6.QtCore import Qt, QPointF, QRectF, QTimer
from PySide6.QtGui import QColor, QPen, QPolygonF, QPainterPath, QShortcut, QKeySequence, QImage, QPixmap
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
    QButtonGroup,
    QListWidget,
    QListWidgetItem,
    QLineEdit,
    QSpinBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QGridLayout,
    QComboBox
)

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT
from matplotlib.figure import Figure

from core.models import ImageData, ROI, Zone
from core.ui.image_canvas import ImageCanvas
from core.geometry.zone_snapper import snap_zone_polygon


class SnappingDebugDialog(QDialog):
    """
    A resizable modal dialog displaying the gradient-profile edge refinement debugging plots.
    Reconstructs the 5 experimental views for any of the 4 edges of the scoring zone.
    """
    def __init__(self, debug_data: dict, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Refinement Snapping Debugger")
        self.resize(1100, 850)
        
        self.debug_data = debug_data
        
        layout = QVBoxLayout(self)
        
        # Edge selector layout
        sel_layout = QHBoxLayout()
        sel_layout.addWidget(QLabel("<b>Select Edge to Inspect:</b>"))
        self.edge_combo = QComboBox(self)
        self.edge_combo.addItems([
            "Edge 1 (P1 -> P2)",
            "Edge 2 (P2 -> P3)",
            "Edge 3 (P3 -> P4)",
            "Edge 4 (P4 -> P1)"
        ])
        self.edge_combo.currentIndexChanged.connect(self.update_plots)
        sel_layout.addWidget(self.edge_combo)
        sel_layout.addStretch()
        layout.addLayout(sel_layout)
        
        # Embed Matplotlib Figure & Canvas
        self.figure = Figure(figsize=(15, 9), dpi=100)
        self.canvas = FigureCanvasQTAgg(self.figure)
        layout.addWidget(self.canvas)
        
        # Add Matplotlib interactive toolbar
        self.toolbar = NavigationToolbar2QT(self.canvas, self)
        layout.addWidget(self.toolbar)
        
        # Close button at the bottom
        btn_layout = QHBoxLayout()
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.accept)
        btn_layout.addStretch()
        btn_layout.addWidget(close_btn)
        layout.addLayout(btn_layout)
        
        # Draw initial plots
        self.update_plots()
        
    def update_plots(self) -> None:
        edge_idx = self.edge_combo.currentIndex()
        crop_rgb = self.debug_data.get("cropped_image")
        edges_data = self.debug_data.get("edges_data", [])
        pts = self.debug_data.get("pts")
        search_margin = self.debug_data.get("search_margin", 15.0)
        
        if crop_rgb is None or len(edges_data) != 4 or pts is None:
            return
            
        p1 = pts[edge_idx]
        p2 = pts[(edge_idx + 1) % 4]
        res = edges_data[edge_idx]
        
        v_edge = p2.astype(np.float32) - p1.astype(np.float32)
        length = np.linalg.norm(v_edge)
        tangent = v_edge / (length + 1e-5)
        normal = np.array([-tangent[1], tangent[0]], dtype=np.float32)
        
        self.figure.clear()
        
        # View 1 — Original ROI with overlays
        ax1 = self.figure.add_subplot(2, 3, 1)
        ax1.imshow(crop_rgb)
        ax1.plot([p1[0], p2[0]], [p1[1], p2[1]], color="red", linewidth=1.2, label="Original Rough Edge")
        p1_ref, p2_ref = res.refined_edge
        ax1.plot([p1_ref[0], p2_ref[0]], [p1_ref[1], p2_ref[1]], color="lime", linewidth=1.2, label="Refined Edge")
        ax1.set_title("View 1: Original Crop & Comparison")
        ax1.legend()
        
        # View 2 — Gradient Magnitude (Scharr Filter)
        crop = cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        gx = cv2.Scharr(gray, cv2.CV_32F, 1, 0)
        gy = cv2.Scharr(gray, cv2.CV_32F, 0, 1)
        grad_mag = cv2.magnitude(gx, gy)
        
        ax2 = self.figure.add_subplot(2, 3, 2)
        ax2.imshow(grad_mag, cmap="gray")
        ax2.set_title("View 2: Grayscale Gradient Magnitude")
        
        # View 3 — Search Corridor Mask
        mask = np.zeros(grad_mag.shape, dtype=np.uint8)
        cv2.line(
            mask,
            (int(round(p1[0])), int(round(p1[1]))),
            (int(round(p2[0])), int(round(p2[1]))),
            255,
            thickness=int(round(2 * search_margin))
        )
        ax3 = self.figure.add_subplot(2, 3, 3)
        ax3.imshow(mask, cmap="gray")
        ax3.set_title("View 3: Search Corridor Mask")
        
        # View 4 — Profile Samples & normal vectors
        ax4 = self.figure.add_subplot(2, 3, 4)
        ax4.imshow(crop_rgb)
        ax4.plot([p1[0], p2[0]], [p1[1], p2[1]], color="red", alpha=0.5, linewidth=1.5)
        num_samples = max(2, int(np.ceil(length / 5.0)))
        sample_factors = np.linspace(0.0, 1.0, num_samples)
        for t in sample_factors:
            pt = p1 + t * v_edge
            ax4.plot(pt[0], pt[1], "go", markersize=4)
            cp1 = pt - normal * search_margin
            cp2 = pt + normal * search_margin
            ax4.plot([cp1[0], cp2[0]], [cp1[1], cp2[1]], color="cyan", alpha=0.6, linewidth=1.0)
        ax4.set_title("View 4: Perpendicular Sampling Lines")
        
        # View 5 — Aggregated 1D Evidence Curve
        ax5 = self.figure.add_subplot(2, 3, 5)
        ax5.plot(res.offsets, res.aggregate_response, color="blue", linewidth=2.0)
        ax5.axvline(0, color="red", linestyle="--", label="User Edge (0)")
        ax5.axvline(res.offset_pixels, color="lime", linestyle="--", label=f"Refined Peak ({res.offset_pixels:+.1f} px)")
        ax5.set_xlabel("Corridor Offset (px)")
        ax5.set_ylabel("Gradient Sum")
        ax5.set_title("View 5: 1D Aggregated Evidence")
        ax5.legend()
        ax5.grid(True)
        
        # View 6 — Metrics
        ax6 = self.figure.add_subplot(2, 3, 6)
        ax6.axis("off")
        stats_text = (
            f"EXPERIMENTAL METRICS:\n"
            f"----------------------------------------\n"
            f"Edge Length: {length:.1f} px\n"
            f"Num Profiles: {num_samples}\n"
            f"Search Margin: {search_margin} px\n"
            f"Aggregation: Median\n\n"
            f"Detected Offset: {res.offset_pixels:+.1f} px\n"
            f"Peak Strength: {res.peak_strength:.1f}\n"
            f"Confidence Ratio: {res.confidence:.3f}\n"
            f"Localization Success: {res.success}\n"
        )
        ax6.text(
            0.05, 0.85,
            stats_text,
            fontsize=10,
            fontfamily="monospace",
            verticalalignment="top",
            bbox=dict(boxstyle="round,pad=0.5", fc="wheat", alpha=0.3)
        )
        ax6.set_title("Edge Localization Statistics")
        
        self.figure.tight_layout()
        self.canvas.draw()


class ZoneVertexHandle(QGraphicsEllipseItem):
    """
    Draggable visual handle representing a single vertex of the active scoring zone.
    """
    def __init__(self, idx: int, pos: QPointF, parent_editor: ZonePolygonEditor) -> None:
        # 8x8 pixel circle handle centered at position
        radius = 5.0
        super().__init__(-radius, -radius, 2 * radius, 2 * radius)
        
        self.idx = idx
        self.parent_editor = parent_editor
        
        self.setPos(pos)
        self.setPen(QPen(QColor(0, 0, 0), 1))
        self.setBrush(QColor(0, 255, 0)) # Green handle
        
        self.setFlags(
            QGraphicsItem.GraphicsItemFlag.ItemIsMovable |
            QGraphicsItem.GraphicsItemFlag.ItemSendsGeometryChanges
        )
        self.setAcceptHoverEvents(True)

    def hoverEnterEvent(self, event) -> None:
        self.setBrush(QColor(255, 165, 0)) # Orange highlight on hover
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        super().hoverEnterEvent(event)

    def hoverLeaveEvent(self, event) -> None:
        self.setBrush(QColor(0, 255, 0))
        self.unsetCursor()
        super().hoverLeaveEvent(event)

    def mousePressEvent(self, event) -> None:
        if self.parent_editor.mode == "delete_node":
            self.parent_editor.delete_vertex(self.idx)
            event.accept()
        else:
            self.parent_editor.save_undo_state()
            super().mousePressEvent(event)

    def itemChange(self, change, value):
        if change == QGraphicsItem.GraphicsItemChange.ItemPositionChange and self.parent_editor:
            new_pos = value
            img_h, img_w = self.parent_editor.image_data.image.shape[:2]
            
            # Constrain handle position to the ROI boundaries (in whole-image space)
            roi = self.parent_editor.roi
            cx = max(float(roi.x), min(new_pos.x(), float(roi.x2 - 1)))
            cy = max(float(roi.y), min(new_pos.y(), float(roi.y2 - 1)))
            clamped = QPointF(cx, cy)
            
            self.parent_editor.update_vertex(self.idx, clamped)
            return clamped

        return super().itemChange(change, value)


class ZonePolygonEditor(QDialog):
    """
    Dialog for creating, snapping, and editing multiple scoring zones.
    """
    def __init__(
        self,
        image_data: ImageData,
        roi: ROI,
        initial_zones: list[Zone] | None = None,
        silhouette_contour: np.ndarray | None = None,
        parent: QWidget | None = None
    ) -> None:
        super().__init__(parent)
        self.image_data = image_data
        self.roi = roi
        self.silhouette_contour = silhouette_contour
        self._silhouette_item: QGraphicsPolygonItem | None = None
        
        # Deep copy initial zones and convert their polygon coordinates from ROI-local to whole-image
        self.zones: list[Zone] = []
        if initial_zones:
            for z in initial_zones:
                # Add roi.x, roi.y offset to polygon points to get whole-image coordinates
                poly_offset = z.polygon + np.array([[[roi.x, roi.y]]], dtype=np.int32)
                self.zones.append(Zone(
                    zone_id=z.zone_id,
                    polygon=poly_offset,
                    score=z.score,
                    name=z.name
                ))

        self.active_zone_index: int | None = None
        
        # History state for the selected zone's snapping undo/redo
        # Maps zone_index -> list of polygon state numpy arrays
        self.undo_stacks: dict[int, list[np.ndarray]] = {}
        self.redo_stacks: dict[int, list[np.ndarray]] = {}

        # Interaction mode: "select", "draw", "add_node", "add_multiple_nodes", "delete_node"
        self.mode = "select"
        
        # Drawing state for a new zone
        self._draw_points: list[QPointF] = []
        self._draw_handles: list[QGraphicsEllipseItem] = []
        self._temp_draw_line: QGraphicsPolygonItem | None = None

        # Scene items
        self._handles: list[ZoneVertexHandle] = []
        self._active_line_item: QGraphicsPolygonItem | None = None
        self._ref_line_item: QGraphicsPolygonItem | None = None # Shows pre-snap rough reference
        self._inactive_line_items: list[QGraphicsPolygonItem] = []

        self._setup_ui()
        self._setup_overlays()
        self._update_zone_list()
        
        # Select first zone by default if available
        if len(self.zones) > 0:
            self._select_zone(0)
            
        self._update_status()

    def _setup_ui(self) -> None:
        self.setWindowTitle("Blueprint Author — Scoring Zone Editor")
        self.resize(1200, 800)
        self.setMinimumSize(900, 650)

        # Horizontal Main Splitter Layout
        main_layout = QHBoxLayout(self)

        # Left Container (Canvas & Toolbar)
        left_widget = QWidget(self)
        left_layout = QVBoxLayout(left_widget)
        left_layout.setContentsMargins(0, 0, 0, 0)

        # Toolbar layout
        toolbar_layout = QHBoxLayout()

        # Zone list modifications
        self.add_zone_btn = QPushButton("+ Add Zone")
        self.add_zone_btn.setCheckable(True)
        self.add_zone_btn.setShortcut(QKeySequence("N"))
        self.add_zone_btn.setToolTip("Start drawing a new scoring zone (Shortcut: N)")
        self.add_zone_btn.clicked.connect(self.start_drawing_zone)

        self.delete_zone_btn = QPushButton("Delete Zone")
        self.delete_zone_btn.setShortcut(QKeySequence("Delete"))
        self.delete_zone_btn.setToolTip("Delete the currently selected scoring zone (Shortcut: Del)")
        self.delete_zone_btn.clicked.connect(self.delete_active_zone)
        self.delete_zone_btn.setEnabled(False)

        # Node editing mode buttons
        self.select_btn = QPushButton("Select / Move")
        self.select_btn.setCheckable(True)
        self.select_btn.setChecked(True)
        self.select_btn.setShortcut(QKeySequence("S"))
        self.select_btn.setToolTip("Select and drag active zone vertices (Shortcut: S)")
        self.select_btn.clicked.connect(lambda: self._set_mode("select"))

        self.add_node_btn = QPushButton("Add Point")
        self.add_node_btn.setCheckable(True)
        self.add_node_btn.setShortcut(QKeySequence("A"))
        self.add_node_btn.setToolTip("Insert a single point on segment, then auto-select (Shortcut: A)")
        self.add_node_btn.clicked.connect(lambda: self._set_mode("add_node"))

        self.add_mult_node_btn = QPushButton("Add Multiple Points")
        self.add_mult_node_btn.setCheckable(True)
        self.add_mult_node_btn.setShortcut(QKeySequence("M"))
        self.add_mult_node_btn.setToolTip("Click to insert multiple vertices continuously (Shortcut: M)")
        self.add_mult_node_btn.clicked.connect(lambda: self._set_mode("add_multiple_nodes"))

        self.delete_node_btn = QPushButton("Delete Point")
        self.delete_node_btn.setCheckable(True)
        self.delete_node_btn.setShortcut(QKeySequence("D"))
        self.delete_node_btn.setToolTip("Click a vertex to remove it from active zone (Shortcut: D)")
        self.delete_node_btn.clicked.connect(lambda: self._set_mode("delete_node"))

        # Button group for edit modes (Add Zone is excluded)
        self.mode_group = QButtonGroup(self)
        self.mode_group.addButton(self.select_btn)
        self.mode_group.addButton(self.add_node_btn)
        self.mode_group.addButton(self.add_mult_node_btn)
        self.mode_group.addButton(self.delete_node_btn)
        self.mode_group.setExclusive(True)

        # Undo / Redo
        self.undo_btn = QPushButton("Undo")
        self.undo_btn.setEnabled(False)
        self.undo_btn.setToolTip("Undo last change to selected zone (Shortcut: Ctrl+Z)")
        self.undo_btn.clicked.connect(self.undo)

        self.redo_btn = QPushButton("Redo")
        self.redo_btn.setEnabled(False)
        self.redo_btn.setToolTip("Redo last change to selected zone (Shortcut: Ctrl+Y)")
        self.redo_btn.clicked.connect(self.redo)

        # Shortcuts
        self.undo_shortcut = QShortcut(QKeySequence.StandardKey.Undo, self)
        self.undo_shortcut.activated.connect(self.undo)
        self.redo_shortcut = QShortcut(QKeySequence.StandardKey.Redo, self)
        self.redo_shortcut.activated.connect(self.redo)

        # Add to toolbar layout
        toolbar_layout.addWidget(self.add_zone_btn)
        toolbar_layout.addWidget(self.delete_zone_btn)
        toolbar_layout.addSpacing(15)
        toolbar_layout.addWidget(self.select_btn)
        toolbar_layout.addWidget(self.add_node_btn)
        toolbar_layout.addWidget(self.add_mult_node_btn)
        toolbar_layout.addWidget(self.delete_node_btn)
        toolbar_layout.addSpacing(15)
        toolbar_layout.addWidget(self.undo_btn)
        toolbar_layout.addWidget(self.redo_btn)
        toolbar_layout.addStretch()
        left_layout.addLayout(toolbar_layout)

        # Image Canvas
        self.canvas = ImageCanvas(self)
        self.canvas.set_image(self.image_data.image)
        self.canvas.mouse_coords_changed.connect(self._on_mouse_coords_changed)
        self.canvas.canvas_clicked.connect(self._on_canvas_clicked)
        left_layout.addWidget(self.canvas, stretch=1)

        # Status Label
        self.status_label = QLabel()
        left_layout.addWidget(self.status_label)

        # Right Container (Sidebar List + Metadata)
        sidebar_widget = QWidget(self)
        sidebar_layout = QVBoxLayout(sidebar_widget)
        sidebar_layout.setContentsMargins(0, 0, 0, 0)
        
        # Zones List
        sidebar_layout.addWidget(QLabel("<b>Scoring Zones:</b>"))
        self.zone_list_widget = QListWidget(self)
        self.zone_list_widget.currentRowChanged.connect(self._on_zone_selection_changed)
        sidebar_layout.addWidget(self.zone_list_widget, stretch=1)

        # Zone refinement group
        self.refinement_group = QGroupBox("Refinement Snapping")
        ref_layout = QVBoxLayout(self.refinement_group)
        
        self.margin_spin = QSpinBox()
        self.margin_spin.setRange(5, 50)
        self.margin_spin.setValue(15)
        self.margin_spin.setSuffix(" px")
        
        margin_form = QFormLayout()
        margin_form.addRow("Search Corridor Margin:", self.margin_spin)
        ref_layout.addLayout(margin_form)
        
        self.snap_btn = QPushButton("Snap Selected Zone")
        self.snap_btn.setShortcut(QKeySequence("Shift+S"))
        self.snap_btn.setToolTip("Refine active zone coordinates by snapping to nearby printed boundaries (Shortcut: Shift+S)")
        self.snap_btn.clicked.connect(self.snap_active_zone)
        self.snap_btn.setEnabled(False)
        ref_layout.addWidget(self.snap_btn)

        self.debug_snap_btn = QPushButton("Debug Snapping")
        self.debug_snap_btn.setShortcut(QKeySequence("Shift+D"))
        self.debug_snap_btn.setToolTip("Show intermediate CV filters and candidate line matches (Shortcut: Shift+D)")
        self.debug_snap_btn.clicked.connect(self.debug_snapping)
        self.debug_snap_btn.setEnabled(False)
        ref_layout.addWidget(self.debug_snap_btn)
        
        sidebar_layout.addWidget(self.refinement_group)

        # Metadata Panel
        self.metadata_group = QGroupBox("Zone Properties")
        self.metadata_group.setEnabled(False)
        meta_layout = QFormLayout(self.metadata_group)

        self.zone_id_edit = QLineEdit()
        self.zone_id_edit.textEdited.connect(self._on_metadata_edited)
        meta_layout.addRow("Zone ID / Label:", self.zone_id_edit)

        self.zone_name_edit = QLineEdit()
        self.zone_name_edit.textEdited.connect(self._on_metadata_edited)
        meta_layout.addRow("Zone Name:", self.zone_name_edit)

        self.score_spin = QDoubleSpinBox()
        self.score_spin.setRange(-1000.0, 1000.0)
        self.score_spin.setValue(10.0)
        self.score_spin.setSingleStep(1.0)
        self.score_spin.valueChanged.connect(self._on_metadata_edited)
        meta_layout.addRow("Points Score:", self.score_spin)

        sidebar_layout.addWidget(self.metadata_group)

        # Dialog Buttons
        dialog_btns_layout = QHBoxLayout()
        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.setShortcut(QKeySequence("Escape"))
        self.cancel_btn.clicked.connect(self.reject)
        
        self.confirm_btn = QPushButton("Confirm")
        self.confirm_btn.setDefault(True)
        self.confirm_btn.setShortcut(QKeySequence("Ctrl+Return"))
        self.confirm_btn.clicked.connect(self.confirm_zones)
        
        dialog_btns_layout.addWidget(self.cancel_btn)
        dialog_btns_layout.addWidget(self.confirm_btn)
        sidebar_layout.addLayout(dialog_btns_layout)

        main_layout.addWidget(left_widget, stretch=3)
        main_layout.addWidget(sidebar_widget, stretch=1)
        self.setLayout(main_layout)

    def _setup_overlays(self) -> None:
        """
        Dims the region outside the ROI to focus attention.
        """
        scene = self.canvas.scene
        img_h, img_w = self.image_data.image.shape[:2]

        full_path = QPainterPath()
        full_path.addRect(0, 0, img_w, img_h)
        
        roi_path = QPainterPath()
        roi_path.addRect(self.roi.x, self.roi.y, self.roi.width, self.roi.height)
        
        outside_path = full_path.subtracted(roi_path)
        
        outside_item = QGraphicsPathItem(outside_path)
        outside_item.setBrush(QColor(0, 0, 0, 140))
        outside_item.setPen(Qt.PenStyle.NoPen)
        scene.addItem(outside_item)

        # Draw the target silhouette contour as a dashed blue reference boundary if available
        if self.silhouette_contour is not None:
            sil_poly = QPolygonF()
            for pt in self.silhouette_contour:
                pt_data = pt.reshape(-1)
                sil_poly.append(QPointF(pt_data[0], pt_data[1]))
            
            self._silhouette_item = QGraphicsPolygonItem(sil_poly)
            # Draw as deep blue reference line
            self._silhouette_item.setPen(QPen(QColor(0, 120, 255), 2.0, Qt.PenStyle.DashLine))
            self._silhouette_item.setBrush(QColor(0, 120, 255, 10))
            self._silhouette_item.setToolTip("Target Silhouette Boundary (Reference)")
            scene.addItem(self._silhouette_item)

    def zoom_to_roi(self) -> None:
        """
        Zooms viewport to ROI with margin.
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
        QTimer.singleShot(50, self.zoom_to_roi)

    # ----------------------------------------------------------------------
    # Selection and List management
    # ----------------------------------------------------------------------

    def _update_zone_list(self) -> None:
        self.zone_list_widget.blockSignals(True)
        self.zone_list_widget.clear()
        
        for idx, zone in enumerate(self.zones):
            label = f"{zone.zone_id} (Score: {zone.score:.1f})"
            if zone.name:
                label += f" - {zone.name}"
            item = QListWidgetItem(label)
            item.setData(Qt.ItemDataRole.UserRole, idx)
            self.zone_list_widget.addItem(item)
            
        if self.active_zone_index is not None:
            self.zone_list_widget.setCurrentRow(self.active_zone_index)
            
        self.zone_list_widget.blockSignals(False)

    def _on_zone_selection_changed(self, row: int) -> None:
        if row < 0 or row >= len(self.zones):
            self._select_zone(None)
        else:
            self._select_zone(row)

    def _select_zone(self, idx: int | None) -> None:
        # Exit drawing mode if active
        if self.mode == "draw":
            self.cancel_drawing()

        self.active_zone_index = idx
        
        # Enable/disable controls
        has_selection = idx is not None
        self.delete_zone_btn.setEnabled(has_selection)
        self.snap_btn.setEnabled(has_selection)
        self.debug_snap_btn.setEnabled(has_selection)
        self.metadata_group.setEnabled(has_selection)
        
        # Toggle tool buttons
        self.select_btn.setEnabled(has_selection)
        self.add_node_btn.setEnabled(has_selection)
        self.add_mult_node_btn.setEnabled(has_selection)
        self.delete_node_btn.setEnabled(has_selection)

        # Clear state
        if not has_selection:
            self.zone_id_edit.clear()
            self.zone_name_edit.clear()
            self.score_spin.setValue(0.0)
            self._clear_active_scene_elements()
            self._update_undo_redo_buttons()
        else:
            zone = self.zones[idx]
            self.zone_id_edit.blockSignals(True)
            self.zone_id_edit.setText(zone.zone_id)
            self.zone_id_edit.blockSignals(False)

            self.zone_name_edit.blockSignals(True)
            self.zone_name_edit.setText(zone.name)
            self.zone_name_edit.blockSignals(False)

            self.score_spin.blockSignals(True)
            self.score_spin.setValue(zone.score)
            self.score_spin.blockSignals(False)

            # Re-draw scene geometry
            self.recreate_handles()
            self.redraw_active_lines()
            self._update_undo_redo_buttons()

        self._draw_inactive_zones()
        self._update_status()

    def _on_metadata_edited(self) -> None:
        if self.active_zone_index is None:
            return
            
        zone = self.zones[self.active_zone_index]
        zone.zone_id = self.zone_id_edit.text().strip() or f"Zone {self.active_zone_index + 1}"
        zone.name = self.zone_name_edit.text().strip()
        zone.score = self.score_spin.value()
        
        self._update_zone_list()

    # ----------------------------------------------------------------------
    # Drawing a new zone
    # ----------------------------------------------------------------------

    def start_drawing_zone(self, checked: bool) -> None:
        if not checked:
            self.cancel_drawing()
            return
            
        self._select_zone(None) # Deselect current zone
        self._set_mode("draw")
        self._draw_points = []
        
        # Clear temporary drawing line and handles
        if self._temp_draw_line:
            self.canvas.scene.removeItem(self._temp_draw_line)
            self._temp_draw_line = None
            
        for handle in self._draw_handles:
            self.canvas.scene.removeItem(handle)
        self._draw_handles.clear()
            
        self._update_status()

    def cancel_drawing(self) -> None:
        self.add_zone_btn.setChecked(False)
        if self._temp_draw_line:
            self.canvas.scene.removeItem(self._temp_draw_line)
            self._temp_draw_line = None
            
        for handle in self._draw_handles:
            self.canvas.scene.removeItem(handle)
        self._draw_handles.clear()
        
        self._draw_points = []
        self._set_mode("select")
        self.select_btn.setChecked(True)
        self._update_status()

    def _add_draw_point(self, scene_pos: QPointF) -> None:
        # Constrain to ROI bounds
        wx = max(float(self.roi.x), min(scene_pos.x(), float(self.roi.x2 - 1)))
        wy = max(float(self.roi.y), min(scene_pos.y(), float(self.roi.y2 - 1)))
        pt = QPointF(wx, wy)
        
        self._draw_points.append(pt)
        
        # Add visual point handle for instant feedback
        radius = 4.0
        handle = QGraphicsEllipseItem(wx - radius, wy - radius, 2 * radius, 2 * radius)
        handle.setBrush(QColor(0, 255, 0)) # Green dot
        handle.setPen(QPen(QColor(0, 0, 0), 1))
        self.canvas.scene.addItem(handle)
        self._draw_handles.append(handle)
        
        # Redraw temporary polygon line
        if self._temp_draw_line is None:
            self._temp_draw_line = QGraphicsPolygonItem()
            self._temp_draw_line.setPen(QPen(QColor(255, 0, 0), 1.5, Qt.PenStyle.DashLine))
            self._temp_draw_line.setBrush(QColor(255, 0, 0, 30))
            self.canvas.scene.addItem(self._temp_draw_line)
            
        poly = QPolygonF(self._draw_points)
        self._temp_draw_line.setPolygon(poly)
        
        self._update_status()
        
        # Auto-lock the zone as soon as 4 vertices are placed
        if len(self._draw_points) == 4:
            self._finish_drawing_zone()

    def _finish_drawing_zone(self) -> None:
        if len(self._draw_points) < 3:
            QMessageBox.warning(self, "Invalid Zone", "A zone must have at least 3 points.")
            self.cancel_drawing()
            return
            
        # Clean up temporary drawing handles
        for handle in self._draw_handles:
            self.canvas.scene.removeItem(handle)
        self._draw_handles.clear()
        
        # Convert QPointF drawing list back to ROI-local OpenCV contour numpy array
        # Input coords are whole-image, we save as whole-image first (will subtract offset in output)
        pts_list = []
        for pt in self._draw_points:
            pts_list.append([[int(round(pt.x())), int(round(pt.y()))]])
            
        polygon = np.array(pts_list, dtype=np.int32)
        
        # Create zone with points score starting from 5.0 and decrementing from previously created zone score
        zone_num = len(self.zones) + 1
        if len(self.zones) == 0:
            default_score = 5.0
        else:
            default_score = self.zones[-1].score - 1.0
            
        new_zone = Zone(
            zone_id=f"Zone {zone_num}",
            polygon=polygon,
            score=default_score,
            name=""
        )
        
        self.zones.append(new_zone)
        
        # Initialize undo/redo history for this new zone index
        idx = len(self.zones) - 1
        self.undo_stacks[idx] = []
        self.redo_stacks[idx] = []

        self.cancel_drawing()
        self._update_zone_list()
        self._select_zone(idx)
        self.snap_active_zone()

    def delete_active_zone(self) -> None:
        if self.active_zone_index is None:
            return
            
        reply = QMessageBox.question(
            self,
            "Delete Zone",
            f"Are you sure you want to delete {self.zones[self.active_zone_index].zone_id}?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        if reply != QMessageBox.StandardButton.Yes:
            return
            
        # Remove and fix history stacks mapping index offsets
        idx = self.active_zone_index
        self.zones.pop(idx)
        
        # Re-index history keys
        new_undos = {}
        new_redos = {}
        for key in sorted(self.undo_stacks.keys()):
            if key < idx:
                new_undos[key] = self.undo_stacks[key]
                new_redos[key] = self.redo_stacks[key]
            elif key > idx:
                new_undos[key - 1] = self.undo_stacks[key]
                new_redos[key - 1] = self.redo_stacks[key]
        self.undo_stacks = new_undos
        self.redo_stacks = new_redos

        self._select_zone(None)
        self._update_zone_list()

    # ----------------------------------------------------------------------
    # Vertex Editing on Active Zone (Similar to ContourEditor)
    # ----------------------------------------------------------------------

    def recreate_handles(self) -> None:
        scene = self.canvas.scene
        
        # Remove old handles
        for handle in self._handles:
            scene.removeItem(handle)
        self._handles.clear()
        
        if self.active_zone_index is None:
            return
            
        zone = self.zones[self.active_zone_index]
        for idx, (x, y) in enumerate(zone.polygon.reshape(-1, 2)):
            scene_pos = QPointF(x, y)
            handle = ZoneVertexHandle(idx, scene_pos, self)
            scene.addItem(handle)
            self._handles.append(handle)

    def redraw_active_lines(self) -> None:
        scene = self.canvas.scene
        
        # Remove old items
        if self._active_line_item:
            scene.removeItem(self._active_line_item)
            self._active_line_item = None
            
        if self._ref_line_item:
            scene.removeItem(self._ref_line_item)
            self._ref_line_item = None

        if self.active_zone_index is None:
            return
            
        zone = self.zones[self.active_zone_index]
        
        # Draw active solid green line
        qpoly = QPolygonF()
        for (x, y) in zone.polygon.reshape(-1, 2):
            qpoly.append(QPointF(x, y))
            
        self._active_line_item = QGraphicsPolygonItem(qpoly)
        self._active_line_item.setPen(QPen(QColor(0, 255, 0), 2.5))
        self._active_line_item.setBrush(QColor(0, 255, 0, 30))
        scene.addItem(self._active_line_item)

    def _draw_inactive_zones(self) -> None:
        scene = self.canvas.scene
        
        # Clear old inactive lines
        for item in self._inactive_line_items:
            scene.removeItem(item)
        self._inactive_line_items.clear()
        
        # Draw all zones except the active one
        for idx, zone in enumerate(self.zones):
            if idx == self.active_zone_index:
                continue
                
            qpoly = QPolygonF()
            for (x, y) in zone.polygon.reshape(-1, 2):
                qpoly.append(QPointF(x, y))
                
            item = QGraphicsPolygonItem(qpoly)
            # Draw inactive zones in Orange with light fill
            item.setPen(QPen(QColor(255, 120, 0), 1.5, Qt.PenStyle.DashLine))
            item.setBrush(QColor(255, 120, 0, 15))
            scene.addItem(item)
            self._inactive_line_items.append(item)

    def _clear_active_scene_elements(self) -> None:
        scene = self.canvas.scene
        for handle in self._handles:
            scene.removeItem(handle)
        self._handles.clear()
        
        if self._active_line_item:
            scene.removeItem(self._active_line_item)
            self._active_line_item = None
            
        if self._ref_line_item:
            scene.removeItem(self._ref_line_item)
            self._ref_line_item = None

    def update_vertex(self, idx: int, scene_pos: QPointF) -> None:
        if self.active_zone_index is None:
            return
            
        zone = self.zones[self.active_zone_index]
        wx = int(round(scene_pos.x()))
        wy = int(round(scene_pos.y()))
        
        # Clamp to ROI bounds
        wx = max(self.roi.x, min(wx, self.roi.x2 - 1))
        wy = max(self.roi.y, min(wy, self.roi.y2 - 1))
        
        zone.polygon[idx] = [wx, wy]
        self.redraw_active_lines()
        self._update_status()

    def add_vertex(self, scene_pos: QPointF) -> None:
        if self.active_zone_index is None:
            return
            
        zone = self.zones[self.active_zone_index]
        wx = scene_pos.x()
        wy = scene_pos.y()
        
        # Verify click is near the ROI
        if not (self.roi.x - 20 <= wx < self.roi.x2 + 20 and self.roi.y - 20 <= wy < self.roi.y2 + 20):
            return

        self.save_undo_state()

        pts = zone.polygon.reshape(-1, 2)
        best_idx, best_proj = self._find_nearest_segment_insertion(pts, (wx, wy))
        
        new_x = int(round(max(float(self.roi.x), min(best_proj[0], float(self.roi.x2 - 1)))))
        new_y = int(round(max(float(self.roi.y), min(best_proj[1], float(self.roi.y2 - 1)))))
        new_pt = np.array([new_x, new_y])
        
        pts = np.insert(pts, best_idx, new_pt, axis=0)
        zone.polygon = pts.reshape(-1, 1, 2)
        
        self.recreate_handles()
        self.redraw_active_lines()
        self._update_status()

    def delete_vertex(self, idx: int) -> None:
        if self.active_zone_index is None:
            return
            
        zone = self.zones[self.active_zone_index]
        pts = zone.polygon.reshape(-1, 2)
        if len(pts) <= 3:
            QMessageBox.warning(self, "Invalid Action", "Scoring zone polygon must have at least 3 vertices.")
            return

        self.save_undo_state()

        pts = np.delete(pts, idx, axis=0)
        zone.polygon = pts.reshape(-1, 1, 2)
        
        self.recreate_handles()
        self.redraw_active_lines()
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

    # ----------------------------------------------------------------------
    # Undo / Redo History Management
    # ----------------------------------------------------------------------

    def save_undo_state(self) -> None:
        idx = self.active_zone_index
        if idx is None:
            return
            
        zone = self.zones[idx]
        self.undo_stacks[idx].append(zone.polygon.copy())
        self.redo_stacks[idx].clear()
        
        if len(self.undo_stacks[idx]) > 5:
            self.undo_stacks[idx].pop(0)
            
        self._update_undo_redo_buttons()

    def undo(self) -> None:
        idx = self.active_zone_index
        if idx is None or not self.undo_stacks.get(idx):
            return
            
        zone = self.zones[idx]
        self.redo_stacks[idx].append(zone.polygon.copy())
        if len(self.redo_stacks[idx]) > 5:
            self.redo_stacks[idx].pop(0)
            
        zone.polygon = self.undo_stacks[idx].pop()
        
        self.recreate_handles()
        self.redraw_active_lines()
        self._update_status()
        self._update_undo_redo_buttons()

    def redo(self) -> None:
        idx = self.active_zone_index
        if idx is None or not self.redo_stacks.get(idx):
            return
            
        zone = self.zones[idx]
        self.undo_stacks[idx].append(zone.polygon.copy())
        if len(self.undo_stacks[idx]) > 5:
            self.undo_stacks[idx].pop(0)
            
        zone.polygon = self.redo_stacks[idx].pop()
        
        self.recreate_handles()
        self.redraw_active_lines()
        self._update_status()
        self._update_undo_redo_buttons()

    def _update_undo_redo_buttons(self) -> None:
        idx = self.active_zone_index
        if idx is None:
            self.undo_btn.setEnabled(False)
            self.redo_btn.setEnabled(False)
            return
            
        undos = self.undo_stacks.get(idx, [])
        redos = self.redo_stacks.get(idx, [])
        self.undo_btn.setEnabled(len(undos) > 0)
        self.redo_btn.setEnabled(len(redos) > 0)

    # ----------------------------------------------------------------------
    # Local Snapping Algorithm Integration
    # ----------------------------------------------------------------------

    def snap_active_zone(self) -> None:
        idx = self.active_zone_index
        if idx is None:
            return
            
        zone = self.zones[idx]
        
        # Save undo state before snapping
        self.save_undo_state()
        
        # The snap algorithm runs on ROI-local coordinates.
        # Convert our whole-image coordinates back to ROI-local for processing.
        roi_local_poly = zone.polygon - np.array([[[self.roi.x, self.roi.y]]], dtype=np.int32)
        
        search_margin = float(self.margin_spin.value())
        
        print(f"Snapping {zone.zone_id} with margin={search_margin} px...")
        
        # Execute local snap
        snapped_roi_local = snap_zone_polygon(
            image_data=self.image_data,
            roi=self.roi,
            polygon=roi_local_poly,
            search_margin=search_margin,
            return_debug=False
        )
        
        # Convert result back to whole-image coordinates
        snapped_whole_image = snapped_roi_local + np.array([[[self.roi.x, self.roi.y]]], dtype=np.int32)
        
        # Apply the snapped contour
        zone.polygon = snapped_whole_image
        
        self.recreate_handles()
        self.redraw_active_lines()
        self._update_status()
        self._update_undo_redo_buttons()
        
        print("Snapping complete.")

    def debug_snapping(self) -> None:
        """
        Executes a dry-run snapping refinement pass on the active zone
        specifically to collect and display the snapping debug details dialog.
        """
        idx = self.active_zone_index
        if idx is None:
            return
            
        zone = self.zones[idx]
        
        # Convert our whole-image coordinates back to ROI-local for processing
        roi_local_poly = zone.polygon - np.array([[[self.roi.x, self.roi.y]]], dtype=np.int32)
        search_margin = float(self.margin_spin.value())
        
        # Execute snapping refinement in dry-run mode (does not save changes to active zone)
        _, debug_dict = snap_zone_polygon(
            image_data=self.image_data,
            roi=self.roi,
            polygon=roi_local_poly,
            search_margin=search_margin,
            return_debug=True
        )
        
        # Open Snapping Debug Dialog
        if debug_dict:
            debug_dlg = SnappingDebugDialog(debug_dict, self)
            debug_dlg.exec()

    # ----------------------------------------------------------------------
    # Event Handlers & Modes
    # ----------------------------------------------------------------------

    def _set_mode(self, mode: str) -> None:
        self.mode = mode
        # If we selected a standard mode, uncheck 'Add Zone' drawing button
        if mode != "draw":
            self.add_zone_btn.setChecked(False)
        self._update_status()

    def _on_mouse_coords_changed(self, x: int, y: int) -> None:
        self._update_status(mouse_pos=(x, y))

    def _on_canvas_clicked(self, scene_pos: QPointF) -> None:
        if self.mode == "draw":
            self._add_draw_point(scene_pos)
        elif self.mode == "add_node":
            self.add_vertex(scene_pos)
            # Auto-revert to Select mode
            self.select_btn.setChecked(True)
            self._set_mode("select")
        elif self.mode == "add_multiple_nodes":
            self.add_vertex(scene_pos)

    def keyPressEvent(self, event) -> None:
        # Handle Enter key to finish drawing zone
        if self.mode == "draw" and (event.key() == Qt.Key.Key_Return or event.key() == Qt.Key.Key_Enter):
            self._finish_drawing_zone()
            event.accept()
        else:
            super().keyPressEvent(event)

    def _update_status(self, mouse_pos: tuple[int, int] | None = None) -> None:
        mode_str = self.mode.replace("_", " ").capitalize()
        if self.mode == "add_multiple_nodes":
            mode_str = "Add Multiple Points"
        elif self.mode == "add_node":
            mode_str = "Add Point"
        elif self.mode == "delete_node":
            mode_str = "Delete Point"
            
        pos_str = ""
        if mouse_pos is not None:
            # Check if mouse is inside ROI
            rx = mouse_pos[0] - self.roi.x
            ry = mouse_pos[1] - self.roi.y
            if 0 <= rx < self.roi.width and 0 <= ry < self.roi.height:
                pos_str = f" | Cursor (ROI-local): x={rx}, y={ry}"
            else:
                pos_str = f" | Cursor: x={mouse_pos[0]}, y={mouse_pos[1]}"
                
        # Sub-status depending on mode
        instructions = ""
        if self.mode == "draw":
            pt_cnt = len(self._draw_points)
            instructions = f" | Click to add vertex ({pt_cnt} added). Press Enter to finish."
        elif self.active_zone_index is not None:
            pts_cnt = len(self.zones[self.active_zone_index].polygon)
            instructions = f" | Selected: {self.zones[self.active_zone_index].zone_id} ({pts_cnt} vertices)"

        self.status_label.setText(
            f"<b>Mode:</b> {mode_str}{instructions}{pos_str}"
        )

    # ----------------------------------------------------------------------
    # Confirm & Output
    # ----------------------------------------------------------------------

    def confirm_zones(self) -> None:
        # Validate number of zones: minimum 3, maximum 10
        if not (3 <= len(self.zones) <= 10):
            QMessageBox.critical(
                self,
                "Validation Error",
                f"Blueprint requires between 3 and 10 scoring zones. You currently have {len(self.zones)}."
            )
            return

        # Validate that all zones are closed polygons with at least 3 vertices
        for z in self.zones:
            pts = z.polygon.reshape(-1, 2)
            if len(pts) < 3:
                QMessageBox.critical(
                    self,
                    "Validation Error",
                    f"Zone {z.zone_id} has fewer than 3 vertices and is invalid."
                )
                return
                
            area = cv2.contourArea(z.polygon)
            if area <= 0.0:
                QMessageBox.critical(
                    self,
                    "Validation Error",
                    f"Zone {z.zone_id} has invalid/zero area. Please adjust its coordinates."
                )
                return

        # Prepare final list: convert coordinates back to ROI-local space as requested
        self.final_zones: list[Zone] = []
        for z in self.zones:
            # Subtract roi.x, roi.y to make it ROI-local
            roi_local_poly = z.polygon - np.array([[[self.roi.x, self.roi.y]]], dtype=np.int32)
            self.final_zones.append(Zone(
                zone_id=z.zone_id,
                polygon=roi_local_poly,
                score=z.score,
                name=z.name
            ))
            
        # Add Outer Silhouette zone representing the area between the largest zone and the silhouette
        if self.silhouette_contour is not None:
            lowest_score = min(z.score for z in self.zones) if self.zones else 1.0
            # Convert whole-image silhouette coordinates to ROI-local space
            roi_local_silhouette = self.silhouette_contour - np.array([[[self.roi.x, self.roi.y]]], dtype=np.int32)
            
            # The score defaults to lowest_score - 1.0, clamped to >= 1.0
            outer_score = max(1.0, lowest_score - 1.0)
            self.final_zones.append(Zone(
                zone_id="Outer Silhouette",
                polygon=roi_local_silhouette,
                score=outer_score,
                name="Outer Target Area"
            ))
            
        self.accept()
