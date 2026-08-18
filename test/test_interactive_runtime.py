from __future__ import annotations

import sys
import time
from pathlib import Path
import numpy as np
import cv2
import matplotlib.pyplot as plt

from PySide6.QtCore import Qt, QRectF, QPointF, QTimer
from PySide6.QtGui import QColor, QPen, QPolygonF, QBrush, QFont, QPainter
from PySide6.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QPushButton,
    QLabel,
    QFileDialog,
    QMessageBox,
    QCheckBox,
    QGroupBox,
    QTextEdit,
    QDialog,
    QGraphicsView,
    QGraphicsScene,
    QGraphicsPixmapItem,
    QGraphicsPolygonItem,
    QGraphicsEllipseItem,
    QGraphicsSimpleTextItem,
)

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT
from matplotlib.figure import Figure

# Import Runtime pipeline components
from runtime.engine import RuntimeEngine
from runtime.models import RuntimeStatus
from core.detection.apriltag_detector import AprilTagDetector


class RuntimeMatplotlibDialog(QDialog):
    """
    Diagnostics Dialog containing 4 Matplotlib subplots displaying coarse projection,
    feature descriptor correspondences, warped geometries, and shot scoring statistics.
    """
    def __init__(
        self,
        ref_image: np.ndarray,
        curr_image: np.ndarray,
        blueprint_id: str,
        engine: RuntimeEngine,
        parent: QWidget | None = None
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Runtime Alignment & Detection Diagnostics")
        self.resize(1100, 850)

        self.ref_image = ref_image
        self.curr_image = curr_image
        self.engine = engine

        layout = QVBoxLayout(self)
        self.fig = Figure(figsize=(10, 8))
        self.canvas = FigureCanvasQTAgg(self.fig)
        self.toolbar = NavigationToolbar2QT(self.canvas, self)

        layout.addWidget(self.toolbar)
        layout.addWidget(self.canvas)

        self.plot_diagnostics()

    def plot_diagnostics(self) -> None:
        self.fig.clear()
        
        # 2x2 Subplots
        ax1 = self.fig.add_subplot(221)
        ax2 = self.fig.add_subplot(222)
        ax3 = self.fig.add_subplot(223)
        ax4 = self.fig.add_subplot(224)

        rgb_curr = cv2.cvtColor(self.curr_image, cv2.COLOR_BGR2RGB)
        rgb_ref = cv2.cvtColor(self.ref_image, cv2.COLOR_BGR2RGB)
        state = self.engine.state
        bp = self.engine.blueprint

        # ------------------------------------------------------------------
        # SUBPLOT 1: Coarse Registration & AprilTags
        # ------------------------------------------------------------------
        ax1.imshow(rgb_curr)
        ax1.set_title("View 1: Coarse AprilTag Registration")
        if state.geometry is not None:
            # Draw coarse search ROI region boundary
            search_poly = state.geometry.search_region.reshape(-1, 2)
            search_draw = np.vstack([search_poly, search_poly[0]])
            ax1.plot(search_draw[:, 0], search_draw[:, 1], color="yellow", linestyle="--", linewidth=1.5, label="Search ROI")
            
        # Draw observed tags
        for t in bp.april_tags:
            # For drawing, transform reference tag corners to current frame
            if state.homography is not None:
                transformed_corners = cv2.perspectiveTransform(
                    t.corners.reshape(-1, 1, 2), 
                    state.homography
                ).reshape(4, 2)
                tag_draw = np.vstack([transformed_corners, transformed_corners[0]])
                ax1.plot(tag_draw[:, 0], tag_draw[:, 1], color="lime", linewidth=2.0)
                ax1.text(transformed_corners[0][0], transformed_corners[0][1] - 5, f"Tag {t.tag_id}", color="lime", fontweight="bold")
        ax1.legend(loc="upper right")
        ax1.axis("off")

        # ------------------------------------------------------------------
        # SUBPLOT 2: Local Feature Matches
        # ------------------------------------------------------------------
        # Combine reference and current frame side-by-side
        ref_h, ref_w = self.ref_image.shape[:2]
        curr_h, curr_w = self.curr_image.shape[:2]
        max_h = max(ref_h, curr_h)
        combined = np.zeros((max_h, ref_w + curr_w, 3), dtype=np.uint8)
        combined[:ref_h, :ref_w] = rgb_ref
        combined[:curr_h, ref_w:] = rgb_curr
        
        ax2.imshow(combined)
        ax2.set_title("View 2: Refined Descriptor Matches")
        
        matches = self.engine.feature_reg.last_matches
        if matches is not None:
            src_pts = np.array(matches["src"])
            dst_pts = np.array(matches["dst"])
            inliers = np.array(matches["inliers"])
            
            for idx in range(len(src_pts)):
                sx, sy = src_pts[idx][0], src_pts[idx][1]
                dx, dy = dst_pts[idx][0] + ref_w, dst_pts[idx][1]
                
                # Green for RANSAC inlier matches, Red for outliers
                color = "lime" if inliers[idx] == 1 else "red"
                alpha = 0.6 if inliers[idx] == 1 else 0.25
                ax2.plot([sx, dx], [sy, dy], color=color, alpha=alpha, linewidth=1.0)
                ax2.scatter([sx, dx], [sy, dy], color=color, s=8, alpha=alpha)
        ax2.axis("off")

        # ------------------------------------------------------------------
        # SUBPLOT 3: Warped Target Geometry
        # ------------------------------------------------------------------
        ax3.imshow(rgb_curr)
        ax3.set_title("View 3: Warped Target Geometry")
        if state.geometry is not None:
            # Silhouette
            sil = state.geometry.silhouette.reshape(-1, 2)
            if len(sil) > 0:
                sil_draw = np.vstack([sil, sil[0]])
                ax3.plot(sil_draw[:, 0], sil_draw[:, 1], color="blue", linestyle="--", linewidth=2.0, label="Silhouette")
                
            # Zones
            colors = plt.colormaps["tab10"]
            for idx, (zone_id, poly, score) in enumerate(state.geometry.zones):
                poly_pts = poly.reshape(-1, 2)
                poly_draw = np.vstack([poly_pts, poly_pts[0]])
                c = colors(idx)
                ax3.plot(poly_draw[:, 0], poly_draw[:, 1], color=c, linewidth=1.8, label=f"{zone_id} (pts={score})")
                ax3.fill(poly_pts[:, 0], poly_pts[:, 1], color=c, alpha=0.08)
            ax3.legend(loc="upper right", fontsize=8)
        ax3.axis("off")

        # ------------------------------------------------------------------
        # SUBPLOT 4: Shot Scoring Analysis
        # ------------------------------------------------------------------
        ax4.imshow(rgb_curr)
        ax4.set_title("View 4: Bullet Score Diagnostics")
        if state.geometry is not None:
            rx, ry, rw, rh = bp.roi_bounds
            # Transform bounds to current frame using H
            if state.homography is not None:
                roi_warped = cv2.perspectiveTransform(
                    np.array([[[rx, ry], [rx+rw, ry+rh]]], dtype=np.float32), 
                    state.homography
                ).reshape(2, 2)
                xmin = int(min(roi_warped[0][0], roi_warped[1][0]))
                xmax = int(max(roi_warped[0][0], roi_warped[1][0]))
                ymin = int(min(roi_warped[0][1], roi_warped[1][1]))
                ymax = int(max(roi_warped[0][1], roi_warped[1][1]))
                
                # Pad bounds slightly
                pad = 40
                xmin = max(0, xmin - pad)
                xmax = min(curr_w - 1, xmax + pad)
                ymin = max(0, ymin - pad)
                ymax = min(curr_h - 1, ymax + pad)
                
                ax4.set_xlim(xmin, xmax)
                ax4.set_ylim(ymax, ymin)  # Flip y to match image coordinate origin

            # Execute a mock bullet detection internally
            bullet_pipeline = self.engine.bullet_pipe
            detected = bullet_pipeline.detect_bullets(self.curr_image, state.geometry)
            
            for center, diameter_px in detected:
                # Assign to zone
                matched_zone = "Miss"
                for zone_id, poly, score in state.geometry.zones:
                    if cv2.pointPolygonTest(poly, center, False) >= 0:
                        matched_zone = f"{zone_id} ({score} pts)"
                        break
                        
                ax4.scatter(center[0], center[1], color="deeppink", s=25, edgecolors="white", linewidths=1.0)
                ax4.text(center[0] + 12, center[1] + 4, matched_zone, color="deeppink", fontweight="bold", fontsize=9)
                # Plot circular bounds
                circle = plt.Circle(center, diameter_px / 2.0, color="deeppink", fill=False, linewidth=1.5)
                ax4.add_patch(circle)
                
        ax4.axis("off")
        self.canvas.draw()


class RuntimeDebuggerApp(QMainWindow):
    """
    Stateful interactive runtime alignment and scoring system debugger GUI.
    """
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Runtime Registration & Scoring Diagnostics App")
        self.resize(1300, 850)

        # Initialize State Engine
        self.engine = RuntimeEngine()
        self.april_detector = AprilTagDetector(families="tag36h11", nthreads=4)
        
        # Test state
        self.curr_frame: np.ndarray | None = None
        self.curr_frame_path: Path | None = None
        self.blueprint_path: Path | None = None
        self.ref_image: np.ndarray | None = None

        self._setup_ui()
        self._load_defaults()

    def _setup_ui(self) -> None:
        # Central widget
        central = QWidget(self)
        self.setCentralWidget(central)
        main_layout = QHBoxLayout(central)

        # -------------------------------------------------------------
        # LEFT: Sidebar Controls
        # -------------------------------------------------------------
        sidebar = QVBoxLayout()
        main_layout.addLayout(sidebar, stretch=1)

        # Group 1: Package Loader
        load_group = QGroupBox("Load Reference & Observation")
        load_layout = QVBoxLayout(load_group)
        
        self.blueprint_lbl = QLabel("Blueprint: Not Loaded")
        self.blueprint_lbl.setWordWrap(True)
        load_layout.addWidget(self.blueprint_lbl)
        
        self.load_bp_btn = QPushButton("1. Load Blueprint Package")
        self.load_bp_btn.clicked.connect(self._on_load_blueprint)
        load_layout.addWidget(self.load_bp_btn)
        
        self.frame_lbl = QLabel("Camera Frame: Not Loaded")
        self.frame_lbl.setWordWrap(True)
        load_layout.addWidget(self.frame_lbl)
        
        self.load_frame_btn = QPushButton("2. Load Test Frame Image")
        self.load_frame_btn.clicked.connect(self._on_load_frame)
        load_layout.addWidget(self.load_frame_btn)
        
        sidebar.addWidget(load_group)

        # Group 2: Pipeline State Actions
        pipeline_group = QGroupBox("Pipeline Engine Steps")
        pipeline_layout = QVBoxLayout(pipeline_group)

        self.acquire_btn = QPushButton("3. Run Target Acquisition (Full)")
        self.acquire_btn.clicked.connect(self._on_acquire_target)
        pipeline_layout.addWidget(self.acquire_btn)

        self.track_btn = QPushButton("4. Simulate Motion & Track (Fast)")
        self.track_btn.clicked.connect(self._on_simulate_track)
        pipeline_layout.addWidget(self.track_btn)

        self.score_btn = QPushButton("5. Detect & Score Shots")
        self.score_btn.clicked.connect(self._on_score_shot)
        pipeline_layout.addWidget(self.score_btn)

        sidebar.addWidget(pipeline_group)

        # Group 3: Overlays Toggle
        overlay_group = QGroupBox("Layer Overlays")
        overlay_layout = QVBoxLayout(overlay_group)
        
        self.show_tags_cb = QCheckBox("Show AprilTags (Lime)")
        self.show_tags_cb.setChecked(True)
        self.show_tags_cb.stateChanged.connect(self._redraw_overlay)
        overlay_layout.addWidget(self.show_tags_cb)

        self.show_roi_cb = QCheckBox("Show Search ROI (Yellow)")
        self.show_roi_cb.setChecked(True)
        self.show_roi_cb.stateChanged.connect(self._redraw_overlay)
        overlay_layout.addWidget(self.show_roi_cb)

        self.show_sil_cb = QCheckBox("Show Silhouette (Blue)")
        self.show_sil_cb.setChecked(True)
        self.show_sil_cb.stateChanged.connect(self._redraw_overlay)
        overlay_layout.addWidget(self.show_sil_cb)

        self.show_zones_cb = QCheckBox("Show Scoring Zones (Colored)")
        self.show_zones_cb.setChecked(True)
        self.show_zones_cb.stateChanged.connect(self._redraw_overlay)
        overlay_layout.addWidget(self.show_zones_cb)

        self.show_regions_cb = QCheckBox("Show Feature Regions (Purple)")
        self.show_regions_cb.setChecked(True)
        self.show_regions_cb.stateChanged.connect(self._redraw_overlay)
        overlay_layout.addWidget(self.show_regions_cb)

        self.show_bullets_cb = QCheckBox("Show Bullet Hits (Pink)")
        self.show_bullets_cb.setChecked(True)
        self.show_bullets_cb.stateChanged.connect(self._redraw_overlay)
        overlay_layout.addWidget(self.show_bullets_cb)

        sidebar.addWidget(overlay_group)

        # Diagnostics Panel
        self.open_debug_btn = QPushButton("Open Matplotlib Diagnostics")
        self.open_debug_btn.setStyleSheet("background-color: #0d47a1; color: white; font-weight: bold;")
        self.open_debug_btn.clicked.connect(self._on_open_diagnostics)
        self.open_debug_btn.setEnabled(False)
        sidebar.addWidget(self.open_debug_btn)

        # Group 4: Text Logger
        log_group = QGroupBox("Engine Output Logs")
        log_layout = QVBoxLayout(log_group)
        self.logger = QTextEdit()
        self.logger.setReadOnly(True)
        self.logger.setFont(QFont("Consolas", 9))
        log_layout.addWidget(self.logger)
        sidebar.addWidget(log_group, stretch=1)

        # -------------------------------------------------------------
        # RIGHT: Graphics Canvas View
        # -------------------------------------------------------------
        canvas_layout = QVBoxLayout()
        main_layout.addLayout(canvas_layout, stretch=3)

        self.scene = QGraphicsScene(self)
        self.view = QGraphicsView(self.scene, self)
        self.view.setRenderHint(QPainter.RenderHint.Antialiasing)
        self.view.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        canvas_layout.addWidget(self.view)
        
        # Pixmap item
        self.pixmap_item = QGraphicsPixmapItem()
        self.scene.addItem(self.pixmap_item)

        # Overlay graphics elements lists
        self.tag_items: list[QGraphicsPolygonItem] = []
        self.roi_item: QGraphicsPolygonItem | None = None
        self.sil_item: QGraphicsPolygonItem | None = None
        self.zone_items: list[QGraphicsPolygonItem] = []
        self.region_items: list[QGraphicsPolygonItem] = []
        self.bullet_items: list[QGraphicsEllipseItem] = []
        self.text_items: list[QGraphicsSimpleTextItem] = []

    def log(self, text: str) -> None:
        self.logger.append(text)
        print(text)

    def _load_defaults(self) -> None:
        """Loads authored blueprint and test image by default if they exist."""
        default_bp = Path("blueprints/bp_outdoor_target_001")
        default_img = Path("test_images/outdoor target.jpeg")

        if default_bp.exists():
            self._load_blueprint_from_path(default_bp)
        if default_img.exists():
            self._load_frame_from_path(default_img)

    def _on_load_blueprint(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "Select Blueprint Package Folder", "blueprints")
        if path:
            self._load_blueprint_from_path(Path(path))

    def _load_blueprint_from_path(self, path: Path) -> None:
        self.blueprint_path = path
        self.log(f"Loading blueprint package: {path.name}")
        
        if self.engine.load_blueprint_package(path):
            self.blueprint_lbl.setText(f"Blueprint: {path.name} (Ready)")
            self.ref_image = self.engine.ref_image
            self.log(f"  Blueprint loaded successfully. ID={self.engine.blueprint.blueprint_id}")
            self.log(f"  Stored scale: {self.engine.blueprint.pixels_per_mm:.4f} px/mm")
            self._redraw_overlay()
        else:
            self.blueprint_lbl.setText("Blueprint: Load Error")
            self.log("  [ERROR] Failed to load blueprint package.")

    def _on_load_frame(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Select Test Camera Frame Image", "test_images", "Images (*.jpg *.jpeg *.png)")
        if path:
            self._load_frame_from_path(Path(path))

    def _load_frame_from_path(self, path: Path) -> None:
        self.curr_frame_path = path
        self.curr_frame = cv2.imread(str(path))
        if self.curr_frame is not None:
            self.engine.state.status = RuntimeStatus.ACQUIRING
            self.engine.tracker.prev_gray = None
            self.engine.tracker.prev_pts = None
            self.frame_lbl.setText(f"Frame: {path.name}")
            self.log(f"Loaded camera frame image: {path.name}")
            
            # Show image on graphics scene
            h, w = self.curr_frame.shape[:2]
            rgb = cv2.cvtColor(self.curr_frame, cv2.COLOR_BGR2RGB)
            bytes_per_line = 3 * w
            from PySide6.QtGui import QImage, QPixmap
            qimg = QImage(rgb.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
            self.pixmap_item.setPixmap(QPixmap.fromImage(qimg))
            self.scene.setSceneRect(0, 0, w, h)
            
            # Trigger delayed fit
            QTimer.singleShot(50, self._zoom_to_fit)
            self._redraw_overlay()
        else:
            self.frame_lbl.setText("Frame: Load Error")
            self.log("  [ERROR] Failed to read frame image.")

    def _zoom_to_fit(self) -> None:
        self.view.fitInView(self.pixmap_item, Qt.AspectRatioMode.KeepAspectRatio)

    def _on_acquire_target(self) -> None:
        if self.curr_frame is None or self.engine.blueprint is None:
            QMessageBox.warning(self, "Missing Data", "Load both the blueprint package and test image first.")
            return

        self.log("\n--- RUNNING TARGET ACQUISITION PIPELINE ---")
        
        # 1. Detect observed AprilTags in the current frame
        from core.models import ImageData
        img_data_wrapper = type("ImgData", (), {"image": self.curr_frame})()
        tags_raw = self.april_detector.detect(img_data_wrapper)
        
        observed_tags = []
        for tag in tags_raw.detections:
            observed_tags.append({
                "tag_id": tag.tag_id,
                "corners": tag.corners
            })
            
        self.log(f"  AprilTag detection: observed={len(observed_tags)} tags.")
        
        # 2. Feed to state engine
        start_t = time.perf_counter()
        state = self.engine.process_frame(self.curr_frame, observed_tags)
        end_t = time.perf_counter()
        
        self.log(f"  Acquisition completed in {(end_t - start_t)*1000.0:.1f} ms.")
        self.log(f"  Engine Status : {state.status}")
        self.log(f"  Confidence    : {state.registration_confidence:.3f}")
        
        if state.homography is not None:
            self.log(f"  Scale pixels/mm: {state.pixels_per_mm:.4f}")
            self.open_debug_btn.setEnabled(True)
            self._redraw_overlay()
        else:
            self.log("  [WARNING] Target registration failed.")

    def _on_simulate_track(self) -> None:
        if self.curr_frame is None or self.engine.state.status != RuntimeStatus.READY:
            QMessageBox.warning(self, "Ready Status Required", "Engine must be in READY (Acquired/Tracked) status to track shifts.")
            return

        self.log("\n--- SIMULATING MOTION & RUNNING FRAME TRACKER ---")
        # Apply a small perspective translation shift (3px right, 2px down) to self.curr_frame
        h, w = self.curr_frame.shape[:2]
        H_shift = np.array([
            [1.0, 0.0, 3.0],
            [0.0, 1.0, 2.0],
            [0.0, 0.0, 1.0]
        ], dtype=np.float32)
        
        shifted_frame = cv2.warpPerspective(self.curr_frame, H_shift, (w, h))
        self.curr_frame = shifted_frame
        
        # Refresh visual pixmap
        rgb = cv2.cvtColor(self.curr_frame, cv2.COLOR_BGR2RGB)
        bytes_per_line = 3 * w
        from PySide6.QtGui import QImage, QPixmap
        qimg = QImage(rgb.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
        self.pixmap_item.setPixmap(QPixmap.fromImage(qimg))

        # Update engine on fast tracking path (no AprilTags provided)
        start_t = time.perf_counter()
        state = self.engine.process_frame(shifted_frame)
        end_t = time.perf_counter()
        
        self.log(f"  LK optical flow tracking complete in {(end_t - start_t)*1000.0:.1f} ms.")
        self.log(f"  Engine Status : {state.status}")
        self.log(f"  Confidence    : {state.registration_confidence:.3f}")
        self.log(f"  Scale pixels/mm: {state.pixels_per_mm:.4f}")
        
        self._redraw_overlay()

    def _on_score_shot(self) -> None:
        if self.curr_frame is None or self.engine.state.status not in (RuntimeStatus.READY, RuntimeStatus.CALIBRATED):
            QMessageBox.warning(self, "Acquired Target Required", "Acquire target successfully before processing shots.")
            return

        self.log("\n--- PROCESSING FIRED SHOTS ---")
        # Draw a simulated dark bullet hole (circular blob) inside the center zone
        state = self.engine.state
        bp = self.engine.blueprint
        
        center_zone = next((z for z in state.geometry.zones if z[0] == "Zone 1"), state.geometry.zones[0])
        pts = center_zone[1].reshape(-1, 2)
        cx, cy = int(np.mean(pts[:, 0])), int(np.mean(pts[:, 1]))
        
        # Draw bullet hole on current frame
        cv2.circle(self.curr_frame, (cx, cy), 9, (18, 18, 18), -1)
        
        # Redraw frame canvas
        h, w = self.curr_frame.shape[:2]
        rgb = cv2.cvtColor(self.curr_frame, cv2.COLOR_BGR2RGB)
        bytes_per_line = 3 * w
        from PySide6.QtGui import QImage, QPixmap
        qimg = QImage(rgb.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
        self.pixmap_item.setPixmap(QPixmap.fromImage(qimg))

        # Run scoring pipeline
        start_t = time.perf_counter()
        result = self.engine.process_shot(shot_id=200, frame=self.curr_frame)
        end_t = time.perf_counter()
        
        self.log(f"  Shot scoring completed in {(end_t - start_t)*1000.0:.1f} ms.")
        if result:
            self.log(f"  Total Score Scored: {result.total_score} points")
            self.log(f"  Bullets detected  : {len(result.bullets)}")
            for idx, b in enumerate(result.bullets):
                self.log(f"    Bullet {idx+1}: center=({b.center_px[0]:.1f}, {b.center_px[1]:.1f}), diameter={b.diameter_mm:.2f} mm, zone={b.zone_id} (pts={b.score})")
            self._redraw_overlay()
        else:
            self.log("  [ERROR] Shot processing failed.")

    def _on_open_diagnostics(self) -> None:
        if self.ref_image is None or self.curr_frame is None:
            return
        diag = RuntimeMatplotlibDialog(
            ref_image=self.ref_image,
            curr_image=self.curr_frame,
            blueprint_id=self.engine.blueprint.blueprint_id,
            engine=self.engine,
            parent=self
        )
        diag.exec()

    def _redraw_overlay(self) -> None:
        """Clears and redraws selected overlay elements on the scene."""
        # 1. Clear old items
        for item in self.tag_items:
            self.scene.removeItem(item)
        self.tag_items.clear()
        
        if self.roi_item:
            self.scene.removeItem(self.roi_item)
            self.roi_item = None
            
        if self.sil_item:
            self.scene.removeItem(self.sil_item)
            self.sil_item = None

        for item in self.zone_items:
            self.scene.removeItem(item)
        self.zone_items.clear()

        for item in self.region_items:
            self.scene.removeItem(item)
        self.region_items.clear()

        for item in self.bullet_items:
            self.scene.removeItem(item)
        self.bullet_items.clear()

        for item in self.text_items:
            self.scene.removeItem(item)
        self.text_items.clear()

        # Check state geometry
        state = self.engine.state
        if state.geometry is None or self.curr_frame is None:
            return

        # Draw overlays if checkbox checked
        # 1. Coarse Search ROI (Yellow)
        if self.show_roi_cb.isChecked():
            poly = QPolygonF()
            for pt in state.geometry.search_region.reshape(-1, 2):
                poly.append(QPointF(pt[0], pt[1]))
            self.roi_item = QGraphicsPolygonItem(poly)
            self.roi_item.setPen(QPen(QColor(255, 235, 59), 2, Qt.PenStyle.DashLine))
            self.scene.addItem(self.roi_item)

        # 2. AprilTags (Lime)
        if self.show_tags_cb.isChecked() and state.homography is not None:
            for t in self.engine.blueprint.april_tags:
                transformed = cv2.perspectiveTransform(
                    t.corners.reshape(-1, 1, 2), 
                    state.homography
                ).reshape(4, 2)
                
                poly = QPolygonF()
                for pt in transformed:
                    poly.append(QPointF(pt[0], pt[1]))
                tag_item = QGraphicsPolygonItem(poly)
                tag_item.setPen(QPen(QColor(76, 175, 80), 2.5))
                self.scene.addItem(tag_item)
                self.tag_items.append(tag_item)

        # 3. Silhouette (Blue)
        if self.show_sil_cb.isChecked() and len(state.geometry.silhouette) > 0:
            poly = QPolygonF()
            for pt in state.geometry.silhouette.reshape(-1, 2):
                poly.append(QPointF(pt[0], pt[1]))
            self.sil_item = QGraphicsPolygonItem(poly)
            self.sil_item.setPen(QPen(QColor(33, 150, 243), 2, Qt.PenStyle.DashLine))
            self.scene.addItem(self.sil_item)

        # 4. Scoring Zones (Colored solid outlines)
        if self.show_zones_cb.isChecked():
            from matplotlib.colors import to_rgb
            import matplotlib.pyplot as plt
            colors = plt.colormaps["tab10"]
            for idx, (zone_id, poly, score) in enumerate(state.geometry.zones):
                qpoly = QPolygonF()
                pts = poly.reshape(-1, 2)
                for pt in pts:
                    qpoly.append(QPointF(pt[0], pt[1]))
                    
                zone_item = QGraphicsPolygonItem(qpoly)
                c_rgb = [int(x * 255) for x in colors(idx)[:3]]
                color = QColor(c_rgb[0], c_rgb[1], c_rgb[2])
                zone_item.setPen(QPen(color, 2))
                zone_item.setBrush(QBrush(QColor(c_rgb[0], c_rgb[1], c_rgb[2], 25)))
                self.scene.addItem(zone_item)
                self.zone_items.append(zone_item)
                
                # Add text label for score
                lbl = QGraphicsSimpleTextItem(f"{zone_id} ({score} pts)")
                lbl.setBrush(QBrush(color))
                lbl.setFont(QFont("Consolas", 10, QFont.Weight.Bold))
                lbl.setPos(pts[0][0], pts[0][1] - 15)
                self.scene.addItem(lbl)
                self.text_items.append(lbl)

        # 5. Feature Regions (Purple)
        if self.show_regions_cb.isChecked():
            for region_id, poly in state.geometry.feature_regions:
                qpoly = QPolygonF()
                for pt in poly.reshape(-1, 2):
                    qpoly.append(QPointF(pt[0], pt[1]))
                region_item = QGraphicsPolygonItem(qpoly)
                region_item.setPen(QPen(QColor(156, 39, 176), 1.5, Qt.PenStyle.DashDotLine))
                self.scene.addItem(region_item)
                self.region_items.append(region_item)

        # 6. Bullet holes (Pink circles)
        if self.show_bullets_cb.isChecked():
            detected = self.engine.bullet_pipe.detect_bullets(self.curr_frame, state.geometry)
            for center, diameter_px in detected:
                r = diameter_px / 2.0
                bullet_item = QGraphicsEllipseItem(center[0] - r, center[1] - r, diameter_px, diameter_px)
                bullet_item.setPen(QPen(QColor(233, 30, 99), 2))
                bullet_item.setBrush(QBrush(QColor(233, 30, 99, 80)))
                self.scene.addItem(bullet_item)
                self.bullet_items.append(bullet_item)


def main():
    app = QApplication(sys.argv)
    win = RuntimeDebuggerApp()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
