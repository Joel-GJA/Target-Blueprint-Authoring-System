from __future__ import annotations

import cv2
import numpy as np
from PySide6.QtCore import Qt, QPointF, QRectF, QTimer
from PySide6.QtGui import QColor, QPen, QPolygonF, QKeySequence, QShortcut
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
    QButtonGroup,
    QListWidget,
    QListWidgetItem,
    QLineEdit,
    QSpinBox,
    QFormLayout,
    QGroupBox,
    QComboBox,
    QTableWidget,
    QTableWidgetItem,
    QHeaderView,
    QInputDialog,
)

from core.models import ImageData, ROI, Zone, FeatureRegion, VisualFeatureSet, FeatureRegionTemplate
from core.ui.image_canvas import ImageCanvas
from core.features.orb_generator import ORBFeatureGenerator, generate_zone_corner_regions
from core.features.robustness_experiment import FeatureRobustnessExperiment
from core.detection.apriltag_detector import AprilTagDetector
from core.templates.feature_template_repository import FeatureTemplateRepository
from core.geometry.target_coordinates import get_target_bounds, pixel_to_normalized, normalized_to_pixel

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT
from matplotlib.figure import Figure


class FeaturesDebugDialog(QDialog):
    """
    A scalable and interactive Matplotlib debug dialog for feature verification.
    Displays subplots for authorized regions, ORB keypoints, search masks, and histograms/survival rates.
    """
    def __init__(
        self,
        image_data: ImageData,
        roi: ROI,
        regions: list[FeatureRegion],
        feature_set: VisualFeatureSet,
        apriltags: list,
        robustness_results: dict | None = None,
        parent: QWidget | None = None
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Feature Extraction & Landmark Debugger")
        self.resize(1100, 850)

        self.image_data = image_data
        self.roi = roi
        self.regions = regions
        self.feature_set = feature_set
        self.apriltags = apriltags
        self.robustness_results = robustness_results

        # Crop original image for plotting
        self.crop = roi.crop(image_data.image)
        self.crop_rgb = cv2.cvtColor(self.crop, cv2.COLOR_BGR2RGB)

        layout = QVBoxLayout(self)

        # Region selector layout
        sel_layout = QHBoxLayout()
        sel_layout.addWidget(QLabel("<b>Inspect Region:</b>"))
        self.region_combo = QComboBox(self)
        self.region_combo.addItem("All Regions")
        for region in regions:
            self.region_combo.addItem(region.id)
        self.region_combo.currentIndexChanged.connect(self.update_plots)
        sel_layout.addWidget(self.region_combo)
        sel_layout.addStretch()
        layout.addLayout(sel_layout)

        # Embed Matplotlib Figure & Canvas
        self.figure = Figure(figsize=(12, 8), dpi=100)
        self.canvas = FigureCanvasQTAgg(self.figure)
        layout.addWidget(self.canvas)

        # Add Matplotlib interactive toolbar (enables zooming & panning)
        self.toolbar = NavigationToolbar2QT(self.canvas, self)
        layout.addWidget(self.toolbar)

        # Close button
        btn_layout = QHBoxLayout()
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.accept)
        btn_layout.addStretch()
        btn_layout.addWidget(close_btn)
        layout.addLayout(btn_layout)

        self.update_plots()

    def update_plots(self) -> None:
        self.figure.clear()
        selected_text = self.region_combo.currentText()

        # Build AprilTag mask (blacking out AprilTag regions)
        h, w = self.crop.shape[:2]
        tag_mask = np.ones((h, w), dtype=np.uint8) * 255
        for tag in self.apriltags:
            corners_roi = tag.corners - np.array([self.roi.x, self.roi.y])
            cv2.fillPoly(tag_mask, [corners_roi.astype(np.int32)], 0)

        if selected_text == "All Regions":
            self._plot_all_regions(tag_mask)
        else:
            region = next(r for r in self.regions if r.id == selected_text)
            self._plot_single_region(region, tag_mask)

        self.canvas.draw()

    def _plot_all_regions(self, tag_mask: np.ndarray) -> None:
        # Subplot 1: Regions Overlay on Cropped Image
        ax1 = self.figure.add_subplot(2, 2, 1)
        ax1.imshow(self.crop_rgb)
        
        # Draw region polygons
        for region in self.regions:
            poly = region.polygon.reshape(-1, 2)
            poly_draw = np.vstack([poly, poly[0]])
            color = "purple" if region.region_type == "stable" else ("dodgerblue" if region.region_type == "optional" else "orange")
            ax1.plot(poly_draw[:, 0], poly_draw[:, 1], color=color, linewidth=2)
            
        ax1.set_title("View 1: Authorized Search Regions")
        ax1.axis("off")

        # Subplot 2: Target Search Mask (Combined Regions minus AprilTags)
        ax2 = self.figure.add_subplot(2, 2, 2)
        combined_mask = np.zeros(tag_mask.shape, dtype=np.uint8)
        for region in self.regions:
            cv2.fillPoly(combined_mask, [region.polygon.astype(np.int32)], 255)
        combined_mask = cv2.bitwise_and(combined_mask, tag_mask)
        
        ax2.imshow(combined_mask, cmap="gray")
        ax2.set_title("View 2: Combined Search Mask (Excl. AprilTags)")
        ax2.axis("off")

        # Subplot 3: Spatially Filtered Keypoints
        ax3 = self.figure.add_subplot(2, 2, 3)
        ax3.imshow(self.crop_rgb)
        
        # Plot ORB keypoints
        xs = [f.x for f in self.feature_set.features]
        ys = [f.y for f in self.feature_set.features]
        ax3.scatter(xs, ys, c="lime", s=8, alpha=0.8, edgecolors="none")
        ax3.set_title(f"View 3: Selected Keypoints (Count: {len(self.feature_set.features)})")
        ax3.axis("off")

        # Subplot 4: Robustness Scores or Strength Histogram
        ax4 = self.figure.add_subplot(2, 2, 4)
        if self.robustness_results:
            ids = list(self.robustness_results.keys())
            scores = [self.robustness_results[r_id]["overall_score"] * 100 for r_id in ids]
            colors = ["green" if s >= 40 else "red" for s in scores]
            bars = ax4.bar(ids, scores, color=colors, alpha=0.7)
            ax4.axhline(40, color="gray", linestyle="--", label="Robust Threshold (40%)")
            ax4.set_ylabel("Repeatability Score (%)")
            ax4.set_title("View 4: Region Robustness Scores")
            ax4.set_ylim(0, 100)
            ax4.legend(loc="upper right")
            
            # Label bars
            for bar in bars:
                height = bar.get_height()
                ax4.text(bar.get_x() + bar.get_width()/2.0, height + 1, f"{height:.1f}%", ha='center', va='bottom', fontsize=8)
        else:
            # Fallback to keypoint response histogram
            responses = [f.response for f in self.feature_set.features]
            if responses:
                ax4.hist(responses, bins=20, color="teal", alpha=0.7, rwidth=0.85)
                ax4.set_xlabel("ORB Keypoint Response Strength")
                ax4.set_ylabel("Count")
                ax4.set_title("View 4: Overall Feature Strength Distribution")
            else:
                ax4.text(0.5, 0.5, "No features generated yet", ha='center', va='center')
                ax4.set_title("View 4: Feature Strengths")

        self.figure.tight_layout()

    def _plot_single_region(self, region: FeatureRegion, tag_mask: np.ndarray) -> None:
        # Get region bounding box to zoom in
        rx, ry, rw, rh = cv2.boundingRect(region.polygon)
        
        # Pad bounding box slightly for context
        pad_x = int(rw * 0.15)
        pad_y = int(rh * 0.15)
        
        x1 = max(0, rx - pad_x)
        y1 = max(0, ry - pad_y)
        x2 = min(self.crop.shape[1], rx + rw + pad_x)
        y2 = min(self.crop.shape[0], ry + rh + pad_y)

        # Bounding box crop
        sub_crop_rgb = self.crop_rgb[y1:y2, x1:x2]

        # Subplot 1: Zoomed Crop with Polygon outline
        ax1 = self.figure.add_subplot(2, 2, 1)
        ax1.imshow(sub_crop_rgb)
        
        # Adjust polygon coords relative to zoom crop
        poly_rel = region.polygon.reshape(-1, 2) - np.array([x1, y1])
        poly_draw = np.vstack([poly_rel, poly_rel[0]])
        ax1.plot(poly_draw[:, 0], poly_draw[:, 1], color="magenta", linewidth=2.5)
        ax1.set_title(f"View 1: Zoomed '{region.id}'")
        ax1.axis("off")

        # Subplot 2: Bounding Box Mask (Excluding Tags)
        ax2 = self.figure.add_subplot(2, 2, 2)
        region_mask = np.zeros(tag_mask.shape, dtype=np.uint8)
        cv2.fillPoly(region_mask, [region.polygon.astype(np.int32)], 255)
        combined_mask = cv2.bitwise_and(region_mask, tag_mask)
        sub_mask = combined_mask[y1:y2, x1:x2]
        
        ax2.imshow(sub_mask, cmap="gray")
        ax2.set_title("View 2: Region Search Mask")
        ax2.axis("off")

        # Subplot 3: Zoomed Keypoints with Grid Lines Overlay
        ax3 = self.figure.add_subplot(2, 2, 3)
        ax3.imshow(sub_crop_rgb)
        
        # Draw 4x4 Grid lines
        for i in range(1, 4):
            # Vertical grid line
            gx_rel = rx + int(rw * i / 4) - x1
            ax3.axvline(gx_rel, color="cyan", linestyle=":", alpha=0.6)
            # Horizontal grid line
            gy_rel = ry + int(rh * i / 4) - y1
            ax3.axhline(gy_rel, color="cyan", linestyle=":", alpha=0.6)

        # Plot keypoints relative to zoom crop
        region_features = [f for f in self.feature_set.features if f.region_id == region.id]
        r_xs = [f.x - x1 for f in region_features]
        r_ys = [f.y - y1 for f in region_features]
        
        ax3.scatter(r_xs, r_ys, c="lime", s=12, alpha=0.8, edgecolors="none")
        ax3.set_title(f"View 3: Spatially Distributed Features ({len(region_features)})")
        ax3.axis("off")

        # Subplot 4: Robustness or Feature Response Histogram
        ax4 = self.figure.add_subplot(2, 2, 4)
        if self.robustness_results and region.id in self.robustness_results:
            # Bar chart of the 6 perturbations
            res = self.robustness_results[region.id]
            labels = list(res["scores"].keys())
            scores = [res["scores"][lbl] * 100 for lbl in labels]
            
            # Shorten labels for clean plot
            short_labels = [lbl.replace("rotate_", "rot_").replace("dim_", "").replace("lighting", "light") for lbl in labels]
            
            bars = ax4.bar(short_labels, scores, color="dodgerblue", alpha=0.7)
            ax4.set_ylabel("Survival Rate (%)")
            ax4.set_title("View 4: Perturbation Survival Rates")
            ax4.set_ylim(0, 100)
            ax4.tick_params(axis='x', rotation=30)
            
            for bar in bars:
                height = bar.get_height()
                ax4.text(bar.get_x() + bar.get_width()/2.0, height + 1, f"{int(round(height))}%", ha='center', va='bottom', fontsize=8)
        else:
            region_features = [f for f in self.feature_set.features if f.region_id == region.id]
            r_responses = [f.response for f in region_features]
            if r_responses:
                ax4.hist(r_responses, bins=10, color="magenta", alpha=0.6, rwidth=0.85)
                ax4.set_xlabel("ORB Response Strength")
                ax4.set_ylabel("Count")
                ax4.set_title("View 4: Feature Strength Histogram")
            else:
                ax4.text(0.5, 0.5, "No features found in region", ha='center', va='center')
                ax4.set_title("View 4: Feature Strengths")

class FeatureKeypointItem(QGraphicsEllipseItem):
    """
    Selectable visual marker representing an individual ORB keypoint/descriptor.
    Allows operators to select and manually delete noisy keypoints.
    """
    def __init__(self, feat_idx: int, x: float, y: float, source: str, parent_editor: FeatureRegionEditor) -> None:
        radius = 3.5
        super().__init__(-radius, -radius, 2 * radius, 2 * radius)
        self.feat_idx = feat_idx
        self.source = source
        self.parent_editor = parent_editor
        self.setPos(x, y)
        self.setPen(QPen(QColor(0, 0, 0), 1))
        self.setFlags(
            QGraphicsItem.GraphicsItemFlag.ItemIsSelectable
        )
        self.setZValue(5)

    def paint(self, painter, option, widget=None) -> None:
        if self.isSelected():
            self.setBrush(QColor(255, 0, 0))  # Red when selected
            self.setPen(QPen(QColor(255, 255, 255), 1))  # White border
        else:
            if self.source == "silhouette_boundary":
                self.setBrush(QColor(0, 191, 255))  # Deep sky blue
            elif self.source == "zone_boundary":
                self.setBrush(QColor(255, 140, 0))  # Dark orange
            else:
                self.setBrush(QColor(0, 255, 0))  # Green (landmark region)
            self.setPen(QPen(QColor(0, 0, 0), 1))
        super().paint(painter, option, widget)


class FeatureVertexHandle(QGraphicsEllipseItem):
    """
    Draggable visual handle representing a single vertex of the active Feature Region.
    """
    def __init__(self, region_id: str, idx: int, pos: QPointF, parent_editor: FeatureRegionEditor) -> None:
        radius = 5.0
        super().__init__(-radius, -radius, 2 * radius, 2 * radius)
        
        self.region_id = region_id
        self.idx = idx
        self.parent_editor = parent_editor
        
        self.setPos(pos)
        self.setPen(QPen(QColor(0, 0, 0), 1))
        self.setBrush(QColor(255, 20, 147))  # Deep pink
        
        self.setFlags(
            QGraphicsItem.GraphicsItemFlag.ItemIsMovable |
            QGraphicsItem.GraphicsItemFlag.ItemSendsGeometryChanges
        )
        self.setAcceptHoverEvents(True)

    def hoverEnterEvent(self, event) -> None:
        self.setBrush(QColor(255, 165, 0))  # Orange highlight on hover
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        super().hoverEnterEvent(event)

    def hoverLeaveEvent(self, event) -> None:
        self.setBrush(QColor(255, 20, 147))
        self.unsetCursor()
        super().hoverLeaveEvent(event)

    def itemChange(self, change, value):
        if change == QGraphicsItem.GraphicsItemChange.ItemPositionChange and self.parent_editor:
            new_pos = value
            roi = self.parent_editor.roi
            # Clamp to ROI boundaries in scene coordinates
            cx = max(float(roi.x), min(new_pos.x(), float(roi.x2 - 1)))
            cy = max(float(roi.y), min(new_pos.y(), float(roi.y2 - 1)))
            clamped = QPointF(cx, cy)
            
            self.parent_editor.update_vertex(self.region_id, self.idx, clamped)
            return clamped

        return super().itemChange(change, value)


class FeatureRegionEditor(QDialog):
    """
    Interactive Stable and Opportunistic Feature Region Editor.
    Provides tools to draw regions (rectangles/polygons), configure priority,
    mask AprilTags, extract ORB features, and run offline robustness experiments.
    """
    def __init__(
        self,
        image_data: ImageData,
        roi: ROI,
        initial_regions: list[FeatureRegion] = None,
        zones: list[Zone] = None,
        silhouette_contour: np.ndarray = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Landmark & Feature Region Authoring")
        self.resize(1300, 850)

        self.image_data = image_data
        self.roi = roi
        self.zones = zones or []
        self.initial_regions = initial_regions or []
        self.silhouette_contour = silhouette_contour

        # Compute target reference frame bounds
        self.target_bounds = get_target_bounds(self.roi, self.silhouette_contour)
        
        # Initialize FeatureTemplateRepository
        self.repository = FeatureTemplateRepository("templates")

        # Convert initial regions to scene space (whole-image coordinates)
        self.regions: dict[str, FeatureRegion] = {}
        for r in self.initial_regions:
            poly_scene = r.polygon + np.array([[[roi.x, roi.y]]], dtype=np.int32)
            self.regions[r.id] = FeatureRegion(
                id=r.id,
                polygon=poly_scene,
                region_type=r.region_type,
                priority=r.priority,
                min_features=r.min_features,
                max_features=r.max_features,
                metadata=r.metadata.copy() if r.metadata else {}
            )

        # Editor state
        self.active_region_id: str | None = None
        self.mode = "select"  # "select", "draw_rect", "draw_poly"
        self.draw_points: list[QPointF] = []
        self.generated_features: VisualFeatureSet | None = None
        self.robustness_results: dict | None = None

        # Graphics items
        self.region_items: dict[str, QGraphicsPolygonItem] = {}
        self.vertex_handles: list[FeatureVertexHandle] = []
        self.kp_items: list[QGraphicsEllipseItem] = []
        self.temp_draw_items: list[QGraphicsEllipseItem] = []
        self.apriltags_detected = []

        # Setup AprilTag detections for masking
        try:
            detector = AprilTagDetector()
            self.apriltags_detected = detector.detect(self.image_data).detections
        except Exception as e:
            print(f"AprilTag detection failed: {e}")

        self._setup_ui()
        self._setup_canvas_overlays()
        self._redraw_all_regions()
        self._update_status()

    def _setup_ui(self) -> None:
        main_layout = QHBoxLayout(self)

        # Left panel: Viewport / Canvas
        left_layout = QVBoxLayout()
        
        # Canvas Toolbar
        toolbar_layout = QHBoxLayout()
        self.select_btn = QPushButton("Select / Edit (S)")
        self.select_btn.setCheckable(True)
        self.select_btn.setChecked(True)
        self.select_btn.clicked.connect(lambda: self._set_mode("select"))
        QShortcut(QKeySequence("S"), self, lambda: self.select_btn.animateClick())

        self.draw_rect_btn = QPushButton("Add Rectangle (R)")
        self.draw_rect_btn.setCheckable(True)
        self.draw_rect_btn.clicked.connect(lambda: self._set_mode("draw_rect"))
        QShortcut(QKeySequence("R"), self, lambda: self.draw_rect_btn.animateClick())

        self.draw_poly_btn = QPushButton("Add Polygon (P)")
        self.draw_poly_btn.setCheckable(True)
        self.draw_poly_btn.clicked.connect(lambda: self._set_mode("draw_poly"))
        QShortcut(QKeySequence("P"), self, lambda: self.draw_poly_btn.animateClick())

        self.delete_btn = QPushButton("Delete Selected")
        self.delete_btn.clicked.connect(self._delete_selected_region)
        QShortcut(QKeySequence("Delete"), self, self._delete_selected_region)

        self.mode_group = QButtonGroup(self)
        self.mode_group.addButton(self.select_btn)
        self.mode_group.addButton(self.draw_rect_btn)
        self.mode_group.addButton(self.draw_poly_btn)
        self.mode_group.setExclusive(True)

        toolbar_layout.addWidget(self.select_btn)
        toolbar_layout.addWidget(self.draw_rect_btn)
        toolbar_layout.addWidget(self.draw_poly_btn)
        toolbar_layout.addWidget(self.delete_btn)
        toolbar_layout.addStretch()

        left_layout.addLayout(toolbar_layout)

        # Canvas
        self.canvas = ImageCanvas(self)
        self.canvas.set_image(self.image_data.image)
        self.canvas.canvas_clicked.connect(self._on_canvas_clicked)
        self.canvas.mouse_coords_changed.connect(self._on_mouse_coords_changed)
        left_layout.addWidget(self.canvas, stretch=1)

        self.status_label = QLabel()
        left_layout.addWidget(self.status_label)
        main_layout.addLayout(left_layout, stretch=2)

        # Right panel: Settings & Information
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(10, 0, 10, 0)

        # Region List Group
        list_group = QGroupBox("Authorized Feature Regions")
        list_group_layout = QVBoxLayout(list_group)
        self.region_list = QListWidget()
        self.region_list.itemSelectionChanged.connect(self._on_list_selection_changed)
        list_group_layout.addWidget(self.region_list)
        
        self.gen_corners_btn = QPushButton("Generate Zone Corner Anchors")
        self.gen_corners_btn.clicked.connect(self._generate_zone_corners)
        list_group_layout.addWidget(self.gen_corners_btn)

        self.delete_unstable_btn = QPushButton("Prune Unstable & Empty Regions")
        self.delete_unstable_btn.setStyleSheet("background-color: #c62828; color: white; font-weight: bold;")
        self.delete_unstable_btn.clicked.connect(self._delete_unstable_or_empty_regions)
        list_group_layout.addWidget(self.delete_unstable_btn)

        self.load_template_btn = QPushButton("Load Normalized Template")
        self.load_template_btn.setStyleSheet("background-color: #0d47a1; color: white; font-weight: bold;")
        self.load_template_btn.clicked.connect(self._load_template_clicked)
        list_group_layout.addWidget(self.load_template_btn)

        self.save_template_btn = QPushButton("Save as Template")
        self.save_template_btn.setStyleSheet("background-color: #1565c0; color: white; font-weight: bold;")
        self.save_template_btn.clicked.connect(self._save_template_clicked)
        list_group_layout.addWidget(self.save_template_btn)
        
        right_layout.addWidget(list_group, stretch=1)

        # Selected Region Properties Group
        props_group = QGroupBox("Selected Region Properties")
        props_layout = QFormLayout(props_group)
        
        self.prop_id = QLineEdit()
        self.prop_id.textEdited.connect(self._on_prop_changed)
        
        self.prop_type = QComboBox()
        self.prop_type.addItems(["stable", "optional", "zone_corner"])
        self.prop_type.currentIndexChanged.connect(self._on_prop_changed)
        
        self.prop_priority = QSpinBox()
        self.prop_priority.setRange(1, 5)
        self.prop_priority.valueChanged.connect(self._on_prop_changed)
        
        self.prop_min_feats = QSpinBox()
        self.prop_min_feats.setRange(1, 100)
        self.prop_min_feats.setValue(5)
        self.prop_min_feats.valueChanged.connect(self._on_prop_changed)

        self.prop_max_feats = QSpinBox()
        self.prop_max_feats.setRange(1, 250)
        self.prop_max_feats.setValue(50)
        self.prop_max_feats.valueChanged.connect(self._on_prop_changed)

        props_layout.addRow("Region ID:", self.prop_id)
        props_layout.addRow("Region Type:", self.prop_type)
        props_layout.addRow("Priority (1-5):", self.prop_priority)
        props_layout.addRow("Min Features:", self.prop_min_feats)
        props_layout.addRow("Max Features:", self.prop_max_feats)
        right_layout.addWidget(props_group)

        # Simulation & Analysis Group
        analysis_group = QGroupBox("Simulation & Calibration")
        analysis_layout = QVBoxLayout(analysis_group)
        
        self.preview_btn = QPushButton("Run Feature Generation (ORB)")
        self.preview_btn.setStyleSheet("background-color: #2e7d32; color: white; font-weight: bold;")
        self.preview_btn.clicked.connect(self._run_feature_generation)
        
        self.robustness_btn = QPushButton("Run Offline Robustness Experiment")
        self.robustness_btn.setStyleSheet("background-color: #0277bd; color: white; font-weight: bold;")
        self.robustness_btn.clicked.connect(self._run_robustness_experiment)

        self.debug_plots_btn = QPushButton("Open Feature Debug Screen")
        self.debug_plots_btn.setStyleSheet("background-color: #7b1fa2; color: white; font-weight: bold;")
        self.debug_plots_btn.setEnabled(False)
        self.debug_plots_btn.clicked.connect(self._open_debug_plots)

        self.delete_kp_btn = QPushButton("Delete Selected Keypoints")
        self.delete_kp_btn.setStyleSheet("background-color: #e65100; color: white; font-weight: bold;")
        self.delete_kp_btn.setEnabled(False)
        self.delete_kp_btn.clicked.connect(self._on_delete_kp_btn_clicked)
        
        analysis_layout.addWidget(self.preview_btn)
        analysis_layout.addWidget(self.robustness_btn)
        analysis_layout.addWidget(self.debug_plots_btn)
        analysis_layout.addWidget(self.delete_kp_btn)
        right_layout.addWidget(analysis_group)

        # Performance / Stats Table
        self.stats_table = QTableWidget(0, 4)
        self.stats_table.setHorizontalHeaderLabels(["Region ID", "Features", "Score", "Robustness"])
        self.stats_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.stats_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        right_layout.addWidget(self.stats_table, stretch=1)

        # OK / Cancel Buttons
        btn_layout = QHBoxLayout()
        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(self.reject)
        confirm_btn = QPushButton("Confirm")
        confirm_btn.setDefault(True)
        confirm_btn.clicked.connect(self.accept)
        btn_layout.addWidget(cancel_btn)
        btn_layout.addWidget(confirm_btn)
        right_layout.addLayout(btn_layout)

        main_layout.addWidget(right_panel, stretch=1)
        self.setLayout(main_layout)

    def _setup_canvas_overlays(self) -> None:
        """
        Dims regions outside the target ROI to keep focus.
        Also visualizes detected AprilTags in red.
        """
        scene = self.canvas.scene
        img_h, img_w = self.image_data.image.shape[:2]

        # ROI Border box
        roi_rect = QRectF(self.roi.x, self.roi.y, self.roi.width, self.roi.height)
        roi_border = scene.addRect(roi_rect, QPen(QColor(0, 255, 0), 2, Qt.PenStyle.DashLine))
        roi_border.setZValue(1)

        # Render AprilTags
        for tag in self.apriltags_detected:
            poly = QPolygonF()
            for corner in tag.corners:
                poly.append(QPointF(corner[0], corner[1]))
            tag_item = scene.addPolygon(poly, QPen(QColor(255, 0, 0), 2, Qt.PenStyle.SolidLine), QColor(255, 0, 0, 30))
            tag_item.setZValue(2)

    def _set_mode(self, mode: str) -> None:
        self.mode = mode
        self.draw_points.clear()
        self._clear_vertex_handles()
        self._clear_temp_draw_items()
        self._update_status()

    def _update_status(self, mouse_pos: tuple[int, int] | None = None) -> None:
        mode_str = self.mode.replace("_", " ").capitalize()
        pos_str = ""
        if mouse_pos is not None:
            rx = mouse_pos[0] - self.roi.x
            ry = mouse_pos[1] - self.roi.y
            if 0 <= rx < self.roi.width and 0 <= ry < self.roi.height:
                pos_str = f" | Cursor (ROI-local): x={rx}, y={ry}"
            else:
                pos_str = f" | Cursor: x={mouse_pos[0]}, y={mouse_pos[1]}"

        inst = ""
        if self.mode == "draw_rect":
            inst = f" | Click first corner, then click opposite corner."
            if len(self.draw_points) == 1:
                inst = f" | Corner 1 set. Click opposite corner to finalize."
        elif self.mode == "draw_poly":
            inst = f" | Click to add vertices. Press Enter to finish."

        self.status_label.setText(f"<b>Mode:</b> {mode_str}{inst}{pos_str}")

    def _on_mouse_coords_changed(self, x: int, y: int) -> None:
        self._update_status(mouse_pos=(x, y))

    def _on_canvas_clicked(self, scene_pos: QPointF) -> None:
        if self.mode in ("draw_rect", "draw_poly"):
            self.draw_points.append(scene_pos)
            # Visual feedback: draw small pink dot immediately
            radius = 4.0
            item = self.canvas.scene.addEllipse(
                scene_pos.x() - radius, scene_pos.y() - radius,
                2 * radius, 2 * radius,
                QPen(QColor(0, 0, 0), 1), QColor(255, 20, 147)
            )
            item.setZValue(6)
            self.temp_draw_items.append(item)

            self._update_status()
            if self.mode == "draw_rect" and len(self.draw_points) == 2:
                self._finish_drawing_rectangle()
            elif self.mode == "draw_poly":
                self._redraw_temp_poly()

    def keyPressEvent(self, event) -> None:
        if self.mode == "draw_poly" and (event.key() == Qt.Key.Key_Return or event.key() == Qt.Key.Key_Enter):
            self._finish_drawing_polygon()
            event.accept()
        else:
            super().keyPressEvent(event)

    def _finish_drawing_rectangle(self) -> None:
        p1, p2 = self.draw_points
        x1, y1 = min(p1.x(), p2.x()), min(p1.y(), p2.y())
        x2, y2 = max(p1.x(), p2.x()), max(p1.y(), p2.y())

        # Construct 4 points
        poly_scene = np.array([
            [[x1, y1]],
            [[x2, y1]],
            [[x2, y2]],
            [[x1, y2]]
        ], dtype=np.int32)

        # Generate a unique ID
        idx = 1
        while f"F{idx}" in self.regions:
            idx += 1
        region_id = f"F{idx}"

        # Create FeatureRegion
        self.regions[region_id] = FeatureRegion(
            id=region_id,
            polygon=poly_scene,
            region_type="stable",
            priority=1,
            min_features=5,
            max_features=50,
            metadata={}
        )

        self._set_mode("select")
        self._redraw_all_regions()
        self._update_region_list()
        self._select_region(region_id)

    def _finish_drawing_polygon(self) -> None:
        if len(self.draw_points) < 3:
            QMessageBox.warning(self, "Invalid Polygon", "A polygon must have at least 3 points.")
            self._set_mode("select")
            return

        poly_scene = np.array([[[p.x(), p.y()]] for p in self.draw_points], dtype=np.int32)
        
        idx = 1
        while f"F{idx}" in self.regions:
            idx += 1
        region_id = f"F{idx}"

        self.regions[region_id] = FeatureRegion(
            id=region_id,
            polygon=poly_scene,
            region_type="stable",
            priority=1,
            min_features=5,
            max_features=50,
            metadata={}
        )

        self._set_mode("select")
        self._redraw_all_regions()
        self._update_region_list()
        self._select_region(region_id)

    def _redraw_temp_poly(self) -> None:
        # Visual feedback during polygon drawing is managed by drawing lines
        # on canvas scene. For simplicity, we can redraw active polygon preview items
        pass

    def _redraw_all_regions(self) -> None:
        # Clear existing region items
        for item in self.region_items.values():
            self.canvas.scene.removeItem(item)
        self.region_items.clear()

        # Add graphics items
        for r_id, region in self.regions.items():
            poly = QPolygonF()
            for pt in region.polygon.reshape(-1, 2):
                poly.append(QPointF(pt[0], pt[1]))

            # Style based on selection and type
            is_active = (r_id == self.active_region_id)
            pen_width = 3 if is_active else 1.5
            
            if region.region_type == "stable":
                color = QColor(147, 112, 219, 255)  # Purple
                fill_color = QColor(147, 112, 219, 30)
            elif region.region_type == "optional":
                color = QColor(30, 144, 255, 255)  # Dodger blue
                fill_color = QColor(30, 144, 255, 30)
            else:
                color = QColor(255, 140, 0, 255)  # Dark orange (opportunistic)
                fill_color = QColor(255, 140, 0, 30)

            pen = QPen(color, pen_width, Qt.PenStyle.SolidLine)
            item = self.canvas.scene.addPolygon(poly, pen, fill_color)
            item.setZValue(3)
            self.region_items[r_id] = item

        self._update_region_list()

    def _update_region_list(self) -> None:
        self.region_list.blockSignals(True)
        self.region_list.clear()
        for r_id, region in self.regions.items():
            item = QListWidgetItem(f"{r_id} ({region.region_type}, P{region.priority})")
            item.setData(Qt.ItemDataRole.UserRole, r_id)
            self.region_list.addItem(item)
            if r_id == self.active_region_id:
                item.setSelected(True)
        self.region_list.blockSignals(False)

    def _select_region(self, region_id: str | None) -> None:
        self.active_region_id = region_id
        self._clear_vertex_handles()
        
        # Redraw contours to show selection thickness
        self._redraw_all_regions()

        if not region_id or region_id not in self.regions:
            self._disable_properties()
            return

        region = self.regions[region_id]
        
        # Enable property edits
        self.prop_id.setEnabled(True)
        self.prop_type.setEnabled(True)
        self.prop_priority.setEnabled(True)
        self.prop_min_feats.setEnabled(True)
        self.prop_max_feats.setEnabled(True)

        self.prop_id.blockSignals(True)
        self.prop_id.setText(region.id)
        self.prop_id.blockSignals(False)

        self.prop_type.blockSignals(True)
        self.prop_type.setCurrentText(region.region_type)
        self.prop_type.blockSignals(False)

        self.prop_priority.blockSignals(True)
        self.prop_priority.setValue(region.priority)
        self.prop_priority.blockSignals(False)

        self.prop_min_feats.blockSignals(True)
        self.prop_min_feats.setValue(region.min_features)
        self.prop_min_feats.blockSignals(False)

        self.prop_max_feats.blockSignals(True)
        self.prop_max_feats.setValue(region.max_features)
        self.prop_max_feats.blockSignals(False)

        # Show vertex handles for moving
        if self.mode == "select":
            for idx, pt in enumerate(region.polygon.reshape(-1, 2)):
                handle = FeatureVertexHandle(region_id, idx, QPointF(pt[0], pt[1]), self)
                self.canvas.scene.addItem(handle)
                handle.setZValue(4)
                self.vertex_handles.append(handle)

    def _disable_properties(self) -> None:
        self.prop_id.clear()
        self.prop_id.setEnabled(False)
        self.prop_type.setEnabled(False)
        self.prop_priority.setEnabled(False)
        self.prop_min_feats.setEnabled(False)
        self.prop_max_feats.setEnabled(False)

    def _clear_vertex_handles(self) -> None:
        for handle in self.vertex_handles:
            self.canvas.scene.removeItem(handle)
        self.vertex_handles.clear()

    def _clear_temp_draw_items(self) -> None:
        for item in self.temp_draw_items:
            self.canvas.scene.removeItem(item)
        self.temp_draw_items.clear()

    def _on_list_selection_changed(self) -> None:
        selected = self.region_list.selectedItems()
        if not selected:
            self._select_region(None)
            return
        r_id = selected[0].data(Qt.ItemDataRole.UserRole)
        self._select_region(r_id)

    def _delete_selected_region(self) -> None:
        selected_items = self.canvas.scene.selectedItems()
        selected_kps = [item for item in selected_items if isinstance(item, FeatureKeypointItem)]
        
        if selected_kps:
            self._delete_selected_keypoints(selected_kps)
            return

        if not self.active_region_id:
            return
        del self.regions[self.active_region_id]
        self._select_region(None)
        self._redraw_all_regions()
        self._update_region_list()

        # Update features if they exist
        if self.generated_features:
            filtered_features = [f for f in self.generated_features.features if f.region_id in self.regions]
            regions_stats = {
                r_id: stat for r_id, stat in self.generated_features.quality_metrics.get("regions", {}).items()
                if r_id in self.regions
            }
            self.generated_features = VisualFeatureSet(
                features=tuple(filtered_features),
                quality_metrics={
                    **self.generated_features.quality_metrics,
                    "regions": regions_stats
                }
            )
            self._replot_keypoints()

        # Update robustness results if they exist
        if self.robustness_results:
            self.robustness_results = {
                r_id: res for r_id, res in self.robustness_results.items()
                if r_id in self.regions
            }

        self._update_stats_table(
            features_stats=self.generated_features.quality_metrics.get("regions", {}) if self.generated_features else None,
            robustness_stats=self.robustness_results
        )

    def update_vertex(self, region_id: str, idx: int, pos: QPointF) -> None:
        if region_id not in self.regions:
            return
        region = self.regions[region_id]
        region.polygon[idx, 0, 0] = int(round(pos.x()))
        region.polygon[idx, 0, 1] = int(round(pos.y()))
        
        # Update polygon rendering
        poly = QPolygonF()
        for pt in region.polygon.reshape(-1, 2):
            poly.append(QPointF(pt[0], pt[1]))
        if region_id in self.region_items:
            self.region_items[region_id].setPolygon(poly)

    def _on_prop_changed(self) -> None:
        if not self.active_region_id:
            return
        region = self.regions[self.active_region_id]
        
        # Check ID change
        new_id = self.prop_id.text().strip()
        if new_id and new_id != self.active_region_id:
            if new_id in self.regions:
                # Conflict! Warn user
                pass
            else:
                # Rename key
                self.regions[new_id] = self.regions.pop(self.active_region_id)
                self.active_region_id = new_id
                region = self.regions[new_id]

        region.region_type = self.prop_type.currentText()
        region.priority = self.prop_priority.value()
        region.min_features = self.prop_min_feats.value()
        region.max_features = self.prop_max_feats.value()

        self._redraw_all_regions()
        self._update_region_list()

    def _generate_zone_corners(self) -> None:
        """
        Creates opportunistic square region boundaries centered on all zone vertices.
        """
        if not self.zones:
            QMessageBox.information(self, "No Zones", "Please define scoring zones first.")
            return

        corner_regions = generate_zone_corner_regions(self.zones, radius=15.0)
        added_count = 0
        for r in corner_regions:
            # Shift back to whole-image coordinates
            poly_scene = r.polygon + np.array([[[self.roi.x, self.roi.y]]], dtype=np.int32)
            
            # Check ID uniqueness
            r_id = r.id
            if r_id in self.regions:
                continue
                
            self.regions[r_id] = FeatureRegion(
                id=r_id,
                polygon=poly_scene,
                region_type=r.region_type,
                priority=r.priority,
                min_features=r.min_features,
                max_features=r.max_features,
                metadata=r.metadata
            )
            added_count += 1

        if added_count > 0:
            QMessageBox.information(self, "Anchors Generated", f"Successfully generated {added_count} zone corner feature anchors.")
            self._redraw_all_regions()
            self._update_region_list()
        else:
            QMessageBox.information(self, "Anchors Generated", "No new anchors generated (anchors already exist).")

    def _get_roi_local_regions(self) -> list[FeatureRegion]:
        """
        Converts the active list of scene-space regions back to ROI-local coordinates.
        """
        roi_local_regions = []
        for r_id, region in self.regions.items():
            poly_roi = region.polygon - np.array([[[self.roi.x, self.roi.y]]], dtype=np.int32)
            roi_local_regions.append(FeatureRegion(
                id=region.id,
                polygon=poly_roi,
                region_type=region.region_type,
                priority=region.priority,
                min_features=region.min_features,
                max_features=region.max_features,
                metadata=region.metadata.copy()
            ))
        return roi_local_regions

    def _run_feature_generation(self) -> None:
        """
        Runs ORB extraction and visualizes keypoints in the scene.
        """
        # Clear existing keypoint items
        for item in self.kp_items:
            self.canvas.scene.removeItem(item)
        self.kp_items.clear()

        local_regions = self._get_roi_local_regions()
        if not local_regions:
            QMessageBox.warning(self, "No Regions", "Define at least one region to generate features.")
            return

        generator = ORBFeatureGenerator()
        self.generated_features = generator.generate(
            self.image_data,
            self.roi,
            local_regions,
            self.apriltags_detected,
            zones=self.zones,
            silhouette_contour=self.silhouette_contour
        )

        self._replot_keypoints()

        self._update_stats_table(features_stats=self.generated_features.quality_metrics.get("regions", {}))
        self.debug_plots_btn.setEnabled(True)
        self.delete_kp_btn.setEnabled(True)
        
        QMessageBox.information(
            self,
            "ORB Generation Completed",
            f"Extracted {len(self.generated_features.features)} stable features across "
            f"{len(local_regions)} regions."
        )

    def _run_robustness_experiment(self) -> None:
        """
        Runs synthetic pertubations to check keypoint survival rate.
        """
        if not self.generated_features:
            QMessageBox.warning(self, "ORB Missing", "Please run 'Feature Generation' first to set reference features.")
            return

        local_regions = self._get_roi_local_regions()
        experiment = FeatureRobustnessExperiment()
        
        results = experiment.run_experiment(
            self.image_data,
            self.roi,
            local_regions,
            self.generated_features
        )
        self.robustness_results = results

        self._update_stats_table(robustness_stats=results)

        # Compute overall stats
        avg_score = np.mean([res["overall_score"] for res in results.values()]) if results else 0.0
        robust_count = sum(1 for res in results.values() if res["robustness"] in ["HIGHLY_ROBUST", "ROBUST"])
        
        QMessageBox.information(
            self,
            "Robustness Experiment Complete",
            f"Average Repeatability: {avg_score * 100:.1f}%\n"
            f"Robust Regions: {robust_count} of {len(results)}"
        )

    def _update_stats_table(self, features_stats: dict = None, robustness_stats: dict = None) -> None:
        self.stats_table.setRowCount(len(self.regions))
        for idx, (r_id, region) in enumerate(self.regions.items()):
            # ID
            self.stats_table.setItem(idx, 0, QTableWidgetItem(r_id))
            
            # Features count
            feat_cnt_str = "-"
            if features_stats and r_id in features_stats:
                feat_cnt_str = str(features_stats[r_id]["selected"])
            elif self.generated_features:
                feat_cnt_str = str(sum(1 for f in self.generated_features.features if f.region_id == r_id))
            self.stats_table.setItem(idx, 1, QTableWidgetItem(feat_cnt_str))

            # Robustness Stats
            score_str = "-"
            robust_str = "-"
            if robustness_stats and r_id in robustness_stats:
                res = robustness_stats[r_id]
                score_str = f"{res['overall_score'] * 100:.1f}%"
                robust_str = res["robustness"]
            
            self.stats_table.setItem(idx, 2, QTableWidgetItem(score_str))
            self.stats_table.setItem(idx, 3, QTableWidgetItem(robust_str))

    @property
    def final_regions(self) -> list[FeatureRegion]:
        """
        Returns finalized list of FeatureRegions in ROI-local coordinates.
        """
        return self._get_roi_local_regions()

    def _open_debug_plots(self) -> None:
        """
        Opens the scalable and interactive Matplotlib diagnostic debugger dialog.
        """
        if not self.generated_features:
            return
        dialog = FeaturesDebugDialog(
            image_data=self.image_data,
            roi=self.roi,
            regions=self._get_roi_local_regions(),
            feature_set=self.generated_features,
            apriltags=self.apriltags_detected,
            robustness_results=self.robustness_results,
            parent=self
        )
        dialog.exec()

    def _delete_unstable_or_empty_regions(self) -> None:
        """
        Delete regions that either have 0 features or have been tested and classified as UNSTABLE.
        """
        if not self.generated_features:
            QMessageBox.warning(
                self,
                "Feature Generation Required",
                "Please run 'Run Feature Generation (ORB)' first to analyze features per region."
            )
            return

        to_delete = []
        for r_id, region in self.regions.items():
            # Count features in this region
            feat_cnt = sum(1 for f in self.generated_features.features if f.region_id == r_id)
            
            # Check robustness status if available
            is_unstable = False
            if self.robustness_results and r_id in self.robustness_results:
                is_unstable = (self.robustness_results[r_id]["robustness"] == "UNSTABLE")
                
            if feat_cnt == 0 or is_unstable:
                to_delete.append(r_id)

        if not to_delete:
            QMessageBox.information(
                self,
                "No Regions Pruned",
                "All regions are valid and robust. No regions were pruned."
            )
            return

        # Perform deletion
        for r_id in to_delete:
            self.regions.pop(r_id, None)

        # Clear active selection if the active region was deleted
        if self.active_region_id in to_delete:
            self._select_region(None)

        # Re-draw and update views
        self._redraw_all_regions()
        self._update_region_list()
        
        # Regenerate stats for remaining regions
        # Re-filter generated features to exclude deleted regions
        filtered_features = [f for f in self.generated_features.features if f.region_id in self.regions]
        # Re-calculate quality metrics for the subset
        regions_stats = {
            r_id: stat for r_id, stat in self.generated_features.quality_metrics.get("regions", {}).items()
            if r_id in self.regions
        }
        self.generated_features = VisualFeatureSet(
            features=tuple(filtered_features),
            quality_metrics={
                **self.generated_features.quality_metrics,
                "regions": regions_stats
            }
        )
        
        # Filter robustness results if they exist
        if self.robustness_results:
            self.robustness_results = {
                r_id: res for r_id, res in self.robustness_results.items()
                if r_id in self.regions
            }

        # Update stats table
        self._update_stats_table(
            features_stats=self.generated_features.quality_metrics.get("regions", {}),
            robustness_stats=self.robustness_results
        )

        QMessageBox.information(
            self,
            "Pruning Completed",
            f"Successfully pruned {len(to_delete)} unstable or empty regions:\n" + 
            ", ".join(to_delete)
        )

    def _replot_keypoints(self) -> None:
        """
        Clears and plots all generated ORB keypoints as selectable FeatureKeypointItem instances.
        """
        for item in self.kp_items:
            if item.scene() is not None:
                self.canvas.scene.removeItem(item)
        self.kp_items.clear()

        if not self.generated_features:
            return

        # Plot keypoints in scene space
        for idx, feat in enumerate(self.generated_features.features):
            # Transform ROI-local keypoint coordinate to scene space
            sx = feat.x + self.roi.x
            sy = feat.y + self.roi.y
            
            kp_item = FeatureKeypointItem(idx, sx, sy, feat.source, self)
            self.canvas.scene.addItem(kp_item)
            self.kp_items.append(kp_item)

    def _on_delete_kp_btn_clicked(self) -> None:
        """
        Triggered when clicking the Delete Selected Keypoints button.
        """
        selected_items = self.canvas.scene.selectedItems()
        selected_kps = [item for item in selected_items if isinstance(item, FeatureKeypointItem)]
        
        if not selected_kps:
            QMessageBox.information(
                self,
                "No Keypoints Selected",
                "Select one or more keypoints on the canvas first (use Ctrl+Click to select multiple)."
            )
            return
            
        self._delete_selected_keypoints(selected_kps)

    def _delete_selected_keypoints(self, selected_kps: list[FeatureKeypointItem]) -> None:
        """
        Deletes the selected keypoints from the active Feature Set.
        """
        deleted_indices = {item.feat_idx for item in selected_kps}
        
        # Filter keypoints
        new_features = [f for idx, f in enumerate(self.generated_features.features) if idx not in deleted_indices]
        
        self.generated_features = VisualFeatureSet(
            features=tuple(new_features),
            quality_metrics=self.generated_features.quality_metrics
        )
        
        # Re-plot remaining
        self._replot_keypoints()
        
        # Update stats table
        self._update_stats_table(
            features_stats=self.generated_features.quality_metrics.get("regions", {})
        )
        
        QMessageBox.information(
            self,
            "Keypoints Deleted",
            f"Successfully deleted {len(selected_kps)} keypoint descriptors manually."
        )

    def _save_template_clicked(self) -> None:
        """
        Converts the current authoring regions to normalized target coordinates
        and saves them as a FeatureRegionTemplate JSON file.
        """
        local_regions = self._get_roi_local_regions()
        if not local_regions:
            QMessageBox.warning(self, "No Regions", "Define at least one feature region before saving as a template.")
            return

        target_type, ok1 = QInputDialog.getText(self, "Save Template", "Enter Target Type (e.g., figure_11):")
        if not ok1 or not target_type.strip():
            return
            
        template_id = target_type.lower().strip()
        version, ok2 = QInputDialog.getInt(self, "Save Template", "Enter Template Version:", 1, 1, 100)
        if not ok2:
            return

        # Convert active region polygons to normalized coordinates relative to target bounds
        normalized_regions = []
        for r in local_regions:
            poly_norm = pixel_to_normalized(r.polygon, self.target_bounds)
            normalized_regions.append(FeatureRegion(
                id=r.id,
                polygon=poly_norm,
                region_type=r.region_type,
                priority=r.priority,
                min_features=r.min_features,
                max_features=r.max_features,
                metadata=r.metadata.copy() if r.metadata else {}
            ))

        # Create template dataclass
        template = FeatureRegionTemplate(
            template_id=template_id,
            target_type=target_type.strip(),
            version=version,
            regions=normalized_regions
        )

        try:
            self.repository.save_template(template)
            QMessageBox.information(
                self,
                "Template Saved",
                f"Successfully saved template '{target_type}' (v{version}) to the templates directory."
            )
        except Exception as e:
            QMessageBox.critical(self, "Error Saving Template", f"Failed to save template: {e}")

    def _load_template_clicked(self) -> None:
        """
        Queries available templates, loads the selected template, converts the
        normalized coordinates back to pixel space using target bounds, and overwrites the active editor regions.
        """
        templates = self.repository.list_available_templates()
        if not templates:
            QMessageBox.information(self, "No Templates Found", "No serialized templates found in the 'templates/' folder.")
            return

        # Prompt choice dialog
        items = [f"{t['target_type']} (v{t['version']})" for t in templates]
        item, ok = QInputDialog.getItem(self, "Load Template", "Select a Template:", items, 0, False)
        
        if not ok or not item:
            return

        # Find selected template metadata
        idx = items.index(item)
        sel_meta = templates[idx]

        try:
            template = self.repository.load_template(sel_meta["target_type"], sel_meta["version"])
            if not template:
                QMessageBox.critical(self, "Error Loading", "Failed to load the selected template file.")
                return

            # Clear existing items
            self._select_region(None)
            self.regions.clear()

            # Instantiate normalized regions to current target_bounds (scene coordinates)
            for r in template.regions:
                # Convert normalized coordinates to ROI-local pixel coordinates
                poly_roi = normalized_to_pixel(r.polygon, self.target_bounds)
                # Convert ROI-local pixel coordinates to scene coordinates
                poly_scene = poly_roi + np.array([[[self.roi.x, self.roi.y]]], dtype=np.int32)
                
                self.regions[r.id] = FeatureRegion(
                    id=r.id,
                    polygon=poly_scene.astype(np.int32),
                    region_type=r.region_type,
                    priority=r.priority,
                    min_features=r.min_features,
                    max_features=r.max_features,
                    metadata=r.metadata.copy() if r.metadata else {}
                )

            self._redraw_all_regions()
            self._update_region_list()
            self._select_region(None)

            QMessageBox.information(
                self,
                "Template Loaded",
                f"Successfully loaded and scaled {len(template.regions)} regions from template '{template.target_type}'."
            )
        except Exception as e:
            QMessageBox.critical(self, "Error Loading Template", f"Failed to load or scale template: {e}")

    def zoom_to_roi(self) -> None:
        """
        Zooms and pans the canvas to center on the target ROI with a comfortable margin.
        """
        img_w, img_h = self.image_data.image.shape[1], self.image_data.image.shape[0]
        margin = max(self.roi.width, self.roi.height) * 0.15
        zoom_rect = QRectF(
            self.roi.x - margin,
            self.roi.y - margin,
            self.roi.width + 2 * margin,
            self.roi.height + 2 * margin
        )
        zoom_rect = zoom_rect.intersected(QRectF(0, 0, img_w, img_h))
        self.canvas.zoom_to_rect(zoom_rect)

    def showEvent(self, event) -> None:
        """
        Delayed call to zoom_to_roi when the widget is shown to ensure viewport size is calculated.
        """
        super().showEvent(event)
        QTimer.singleShot(50, self.zoom_to_roi)





