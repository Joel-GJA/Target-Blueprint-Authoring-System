import sys
import os
import time
import numpy as np
import cv2
import matplotlib
matplotlib.use("QtAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtGui import QImage, QPixmap, QPainter, QPen, QColor, QFont
from PySide6.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QPushButton,
    QLabel,
    QComboBox,
    QSplitter,
    QTextBrowser,
    QMessageBox,
    QTabWidget
)

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent.resolve()))

from core.calibration.image_loader import load_image
from runtime.blueprint_loader import BlueprintLoader
from runtime.coarse_registration import CoarseRegistration
from runtime.models import RuntimeStatus, RuntimeState, RuntimeGeometry
from core.detection.apriltag_detector import AprilTagDetector


class RegistrationDebuggerApp(QMainWindow):
    """
    Interactive test application for AprilTag detection and coarse homography registration.
    """
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Runtime Target Acquisition & Registration Debugger")
        self.resize(1400, 900)

        # State fields
        self.blueprint = None
        self.ref_image = None
        self.curr_image_path = None
        self.curr_frame = None
        self.observed_tags = []
        self.H_coarse = None
        self.confidence = 0.0
        self.reproj_err_px = 0.0
        self.det_status = "N/A"
        self.homography_status = "INVALID"
        
        # Load default blueprint package
        self._load_blueprint_package()

        self._setup_ui()
        
        # Initial load of first image
        self._on_image_selected(0)

    def _load_blueprint_package(self) -> None:
        bp_path = Path("blueprints/bp_outdoor_target_001")
        if not bp_path.exists():
            QMessageBox.critical(
                self,
                "Blueprint Error",
                f"Blueprint package not found at {bp_path.resolve()}.\n"
                "Please run test_final_blueprint_contract.py first to create it."
            )
            sys.exit(1)
        try:
            self.blueprint, self.ref_image = BlueprintLoader.load_blueprint(bp_path)
            print(f"Loaded blueprint {self.blueprint.blueprint_id} successfully.")
        except Exception as e:
            QMessageBox.critical(self, "Blueprint Load Failed", f"Error: {e}")
            sys.exit(1)

    def _setup_ui(self) -> None:
        central_widget = QWidget(self)
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)

        # Left Column: Image view + selection controls
        left_panel = QWidget(self)
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(0, 0, 0, 0)

        # Image Selector Combo Box
        selector_layout = QHBoxLayout()
        selector_layout.addWidget(QLabel("<b>Select Test Frame:</b>"))
        self.combo_box = QComboBox(self)
        
        # Check files inside test_images
        self.test_images = [
            "test_images/outdoor target.jpeg",
            "test_images/perspective target.jpeg",
            "test_images/indoor target.jpeg",
            "test_images/target.jpeg",
            "test_images/messed up target.jpeg"
        ]
        # Filter existing files only
        self.test_images = [f for f in self.test_images if os.path.exists(f)]
        for f in self.test_images:
            self.combo_box.addItem(os.path.basename(f))
            
        self.combo_box.currentIndexChanged.connect(self._on_image_selected)
        selector_layout.addWidget(self.combo_box)
        
        # Run button
        self.run_button = QPushButton("Re-run Pipeline", self)
        self.run_button.clicked.connect(self._run_pipeline)
        selector_layout.addWidget(self.run_button)
        left_layout.addLayout(selector_layout)

        # Image view display with tabs for overlays and homographed/rectified view
        self.tabs = QTabWidget(self)
        left_layout.addWidget(self.tabs, stretch=1)

        # Tab 1: Camera view with tag overlays
        self.camera_tab = QWidget()
        cam_layout = QVBoxLayout(self.camera_tab)
        self.image_label = QLabel(self)
        self.image_label.setMinimumSize(600, 700)
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_label.setStyleSheet("border: 2px solid #ccc; background-color: #000;")
        cam_layout.addWidget(self.image_label)
        self.tabs.addTab(self.camera_tab, "Camera View (Overlays)")

        # Tab 2: Warped/Homographed view
        self.rectified_tab = QWidget()
        rect_layout = QVBoxLayout(self.rectified_tab)
        self.rectified_label = QLabel(self)
        self.rectified_label.setMinimumSize(600, 700)
        self.rectified_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.rectified_label.setStyleSheet("border: 2px solid #ccc; background-color: #000;")
        rect_layout.addWidget(self.rectified_label)
        self.tabs.addTab(self.rectified_tab, "Homographed (Rectified) View")
        
        main_layout.addWidget(left_panel, stretch=3)

        # Right Column: Splitter with details and matplotlib subplots
        right_splitter = QSplitter(Qt.Orientation.Vertical, self)

        # Details text panel
        self.details_box = QTextBrowser(self)
        self.details_box.setStyleSheet("background-color: #fafafa; font-family: monospace; font-size: 12px;")
        right_splitter.addWidget(self.details_box)

        # Matplotlib plot panel
        self.canvas_widget = QWidget(self)
        canvas_layout = QVBoxLayout(self.canvas_widget)
        canvas_layout.setContentsMargins(0, 0, 0, 0)
        
        self.figure = Figure(figsize=(5, 4), dpi=100)
        self.canvas = FigureCanvas(self.figure)
        canvas_layout.addWidget(self.canvas)
        right_splitter.addWidget(self.canvas_widget)
        
        right_splitter.setStretchFactor(0, 1)
        right_splitter.setStretchFactor(1, 2)
        
        main_layout.addWidget(right_splitter, stretch=2)

    def _on_image_selected(self, index: int) -> None:
        if index < 0 or index >= len(self.test_images):
            return
        self.curr_image_path = self.test_images[index]
        self.curr_frame = cv2.imread(self.curr_image_path)
        if self.curr_frame is None:
            self.details_box.setHtml(f"<font color='red'>Failed to load {self.curr_image_path}</font>")
            return
        print(f"Loaded test frame: {self.curr_image_path} of shape {self.curr_frame.shape}")
        self._run_pipeline()

    def _run_pipeline(self) -> None:
        if self.curr_frame is None:
            return
            
        t0 = time.time()
        
        # 1. Run AprilTag Detection
        try:
            image_data = load_image(self.curr_image_path)
            detector = AprilTagDetector(families="tag36h11", nthreads=4)
            detection_result = detector.detect(image_data)
            
            self.observed_tags = []
            for tag in detection_result.detections:
                self.observed_tags.append({
                    "tag_id": tag.tag_id,
                    "corners": tag.corners
                })
        except Exception as e:
            self.details_box.setHtml(f"<font color='red'>AprilTag Detection Failed: {e}</font>")
            return
            
        t_detect = (time.time() - t0) * 1000.0
        
        # 2. Compute Coarse Homography
        t1 = time.time()
        H_coarse, confidence = CoarseRegistration.compute_coarse_homography(
            self.blueprint,
            self.observed_tags
        )
        t_homo = (time.time() - t1) * 1000.0
        
        self.H_coarse = H_coarse
        self.confidence = confidence

        # Evaluate homography quality and reprojection error
        self._evaluate_registration_quality()
        
        # Render visualizations
        self._render_view()
        self._render_plots(t_detect, t_homo)
        self._update_details_text(t_detect, t_homo)

    def _evaluate_registration_quality(self) -> None:
        self.det_status = f"{len(self.observed_tags)} / {len(self.blueprint.april_tags)} tags"
        
        if self.H_coarse is None:
            self.homography_status = "INVALID"
            self.reproj_err_px = 0.0
            return
            
        # Check sanity using registration module helper
        sane, reason = CoarseRegistration.evaluate_homography_sanity(self.H_coarse)
        if not sane:
            self.homography_status = f"INVALID ({reason})"
            return
            
        self.homography_status = "VALID"
        
        # 3. Calculate reprojection error
        errors = []
        # Sort reference tag centers to map them to physical corners
        T = self.blueprint.tag_size_mm
        M = 20.0
        W = self.blueprint.target_width_mm
        H = self.blueprint.target_height_mm
        
        corners_mm_by_pos = {
            0: np.array([[M, M], [M + T, M], [M + T, M + T], [M, M + T]], dtype=np.float32),              # TL
            1: np.array([[W - M - T, M], [W - M, M], [W - M, M + T], [W - M - T, M + T]], dtype=np.float32),  # TR
            2: np.array([[W - M - T, H - M - T], [W - M, H - M - T], [W - M, H - M], [W - M - T, H - M]], dtype=np.float32),# BR
            3: np.array([[M, H - M - T], [M + T, H - M - T], [M + T, H - M], [M, H - M]], dtype=np.float32)     # BL
        }
        
        ref_centers = [np.mean(tag.corners, axis=0) for tag in self.blueprint.april_tags]
        ref_centers = np.array(ref_centers, dtype=np.float32)
        sums = ref_centers.sum(axis=1)
        diff = np.diff(ref_centers, axis=1).flatten()
        
        tl_idx = np.argmin(sums)
        br_idx = np.argmax(sums)
        tr_idx = np.argmin(diff)
        bl_idx = np.argmax(diff)
        
        template_corners_by_id = {
            self.blueprint.april_tags[tl_idx].tag_id: corners_mm_by_pos[0],
            self.blueprint.april_tags[tr_idx].tag_id: corners_mm_by_pos[1],
            self.blueprint.april_tags[br_idx].tag_id: corners_mm_by_pos[2],
            self.blueprint.april_tags[bl_idx].tag_id: corners_mm_by_pos[3]
        }
        
        # Project template corners to reference image coordinates
        ref_src_pts = []
        ref_dst_pts = []
        for ref_tag in self.blueprint.april_tags:
            tag_id = ref_tag.tag_id
            if tag_id in template_corners_by_id:
                for idx in range(4):
                    ref_src_pts.append(ref_tag.corners[idx])
                    ref_dst_pts.append(template_corners_by_id[tag_id][idx])
        ref_src_pts = np.array(ref_src_pts, dtype=np.float32)
        ref_dst_pts = np.array(ref_dst_pts, dtype=np.float32)
        H_ref_to_mm, _ = cv2.findHomography(ref_src_pts, ref_dst_pts, 0)
        
        for obs in self.observed_tags:
            tag_id = obs["tag_id"]
            if tag_id in template_corners_by_id:
                # Find matching blueprint corners
                bp_tag = next((t for t in self.blueprint.april_tags if t.tag_id == tag_id), None)
                if bp_tag is not None:
                    # Project blueprint tag corners to current frame using H_coarse
                    pts = bp_tag.corners.reshape(-1, 1, 2)
                    warped_pts = cv2.perspectiveTransform(pts, self.H_coarse).reshape(4, 2)
                    
                    # Align corner indices to find minimal reprojection error (rotation shifts)
                    min_err = float('inf')
                    best_shift_pts = None
                    for shift in range(4):
                        shifted_obs = np.roll(obs["corners"], shift, axis=0)
                        err_dist = np.linalg.norm(shifted_obs - warped_pts, axis=1)
                        mean_err = np.mean(err_dist)
                        if mean_err < min_err:
                            min_err = mean_err
                            best_shift_pts = shifted_obs
                            
                    errors.append(min_err)
                    
        self.reproj_err_px = np.mean(errors) if len(errors) > 0 else 0.0

    def _render_view(self) -> None:
        # Create a display image copy
        display = self.curr_frame.copy()
        
        # 1. Draw detected AprilTags
        for tag in self.observed_tags:
            corners = np.round(tag["corners"]).astype(np.int32)
            cv2.polylines(display, [corners], isClosed=True, color=(0, 255, 0), thickness=2)
            center = tuple(np.round(np.mean(tag["corners"], axis=0)).astype(int))
            cv2.circle(display, center, 5, (0, 0, 255), -1)
            cv2.putText(
                display,
                f"ID {tag['tag_id']}",
                (center[0] - 15, center[1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 0, 0),
                2,
                cv2.LINE_AA
            )
            
        # 2. Draw Warped Target ROI boundary if H_coarse is valid
        if self.H_coarse is not None:
            rx, ry, rw, rh = self.blueprint.roi_bounds
            ref_roi_corners = np.array([
                [rx, ry],
                [rx + rw, ry],
                [rx + rw, ry + rh],
                [rx, ry + rh]
            ], dtype=np.float32).reshape(-1, 1, 2)
            
            warped_roi = cv2.perspectiveTransform(ref_roi_corners, self.H_coarse).reshape(4, 2)
            roi_poly = np.round(warped_roi).astype(np.int32)
            cv2.polylines(display, [roi_poly], isClosed=True, color=(255, 255, 0), thickness=3)
            
            # Label search region
            label_pos = tuple(roi_poly[0])
            cv2.putText(
                display,
                "COARSE TARGET SEARCH REGION",
                (label_pos[0], label_pos[1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 0),
                2,
                cv2.LINE_AA
            )
            
            # Warp reference target silhouette for visual confirmation!
            if self.blueprint.silhouette is not None:
                # Silhouette is stored ROI-local in the blueprint, add ROI offsets first
                sil_ref = self.blueprint.silhouette.copy().reshape(-1, 2)
                sil_ref[:, 0] += rx
                sil_ref[:, 1] += ry
                
                sil_ref_f32 = sil_ref.astype(np.float32)
                sil_warped = cv2.perspectiveTransform(sil_ref_f32.reshape(-1, 1, 2), self.H_coarse).reshape(-1, 2)
                sil_poly = np.round(sil_warped).astype(np.int32)
                cv2.polylines(display, [sil_poly], isClosed=True, color=(0, 255, 255), thickness=1)

        # Convert BGR to RGB for Qt and scale/set main image label
        rgb_img = cv2.cvtColor(display, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_img.shape
        bytes_per_line = ch * w
        q_img = QImage(rgb_img.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
        pixmap = QPixmap.fromImage(q_img)
        scaled_pixmap = pixmap.scaled(
            self.image_label.width() - 4,
            self.image_label.height() - 4,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation
        )
        self.image_label.setPixmap(scaled_pixmap)

        # 3. Generate and Render Homographed (Rectified) View
        if self.H_coarse is not None:
            try:
                # Warp the camera frame back to the blueprint reference space
                h_ref, w_ref = self.ref_image.shape[:2]
                H_inv = np.linalg.inv(self.H_coarse)
                warped_frame = cv2.warpPerspective(self.curr_frame, H_inv, (w_ref, h_ref))
                
                # Draw blueprint reference target silhouette
                if self.blueprint.silhouette is not None:
                    rx, ry, rw, rh = self.blueprint.roi_bounds
                    sil_ref = self.blueprint.silhouette.copy().reshape(-1, 2)
                    sil_ref[:, 0] += rx
                    sil_ref[:, 1] += ry
                    sil_poly = np.round(sil_ref).astype(np.int32)
                    cv2.polylines(warped_frame, [sil_poly], isClosed=True, color=(0, 255, 255), thickness=2)
                    
                    # Draw reference ROI bounding box
                    cv2.rectangle(warped_frame, (rx, ry), (rx + rw, ry + rh), (255, 255, 0), 2)
                    
                # Draw reference AprilTag centers
                for tag in self.blueprint.april_tags:
                    center = tuple(np.round(tag.center).astype(int))
                    cv2.circle(warped_frame, center, 6, (0, 0, 255), -1)
                    cv2.putText(
                        warped_frame,
                        f"Ref {tag.tag_id}",
                        (center[0] - 20, center[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 0, 255),
                        2,
                        cv2.LINE_AA
                    )
                    
                # Draw reference zones
                for zone in self.blueprint.zones:
                    # Zones are ROI-local, add ROI offsets
                    rx, ry, rw, rh = self.blueprint.roi_bounds
                    pts = zone.polygon.copy().reshape(-1, 2)
                    pts[:, 0] += rx
                    pts[:, 1] += ry
                    pts_poly = np.round(pts).astype(np.int32)
                    cv2.polylines(warped_frame, [pts_poly], isClosed=True, color=(255, 100, 100), thickness=2)
                    
                # Convert BGR to RGB for Qt
                warped_rgb = cv2.cvtColor(warped_frame, cv2.COLOR_BGR2RGB)
                wh, ww, wch = warped_rgb.shape
                w_bytes = wch * ww
                qw_img = QImage(warped_rgb.data, ww, wh, w_bytes, QImage.Format.Format_RGB888)
                w_pixmap = QPixmap.fromImage(qw_img)
                scaled_w_pixmap = w_pixmap.scaled(
                    self.rectified_label.width() - 4,
                    self.rectified_label.height() - 4,
                    Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation
                )
                self.rectified_label.setPixmap(scaled_w_pixmap)
            except Exception as e:
                self.rectified_label.setText(f"Homography Warp Error: {e}")
        else:
            self.rectified_label.setText("No valid registration homography available.")

    def _render_plots(self, t_detect: float, t_homo: float) -> None:
        self.figure.clear()
        
        # Draw a bar chart of execution times and confidence score
        ax = self.figure.add_subplot(211)
        stages = ["Tag Detect", "Homography", "Total Pipeline"]
        times = [t_detect, t_homo, t_detect + t_homo]
        colors = ["skyblue", "lightgreen", "coral"]
        
        bars = ax.barh(stages, times, color=colors, height=0.5)
        ax.set_xlabel("Time (ms)")
        ax.set_title("Target Acquisition Execution Times")
        ax.bar_label(bars, fmt="%.1f ms", padding=5)
        ax.set_xlim(0, max(50, (t_detect + t_homo) * 1.3))
        
        # Draw tag layout grid as physically observed vs reference
        ax2 = self.figure.add_subplot(212)
        ax2.set_title("AprilTag Spatial Arrangement")
        ax2.set_xlabel("X (px)")
        ax2.set_ylabel("Y (px)")
        
        # Plot observed tag centers
        obs_x = [np.mean(tag["corners"][:, 0]) for tag in self.observed_tags]
        obs_y = [np.mean(tag["corners"][:, 1]) for tag in self.observed_tags]
        obs_ids = [tag["tag_id"] for tag in self.observed_tags]
        
        ax2.scatter(obs_x, obs_y, color="red", s=80, marker="o", label="Observed")
        for i, txt in enumerate(obs_ids):
            ax2.annotate(f"Tag {txt}", (obs_x[i]+10, obs_y[i]+10), color="red")
            
        ax2.grid(True, linestyle="--", alpha=0.5)
        ax2.legend(loc="lower right")
        
        # Reverse Y axis because image coordinate system has 0 at top
        ax2.invert_yaxis()
        
        self.figure.tight_layout()
        self.canvas.draw()

    def _update_details_text(self, t_detect: float, t_homo: float) -> None:
        status_color = "green" if self.homography_status == "VALID" else "red"
        
        h_matrix_str = "N/A"
        if self.H_coarse is not None:
            rows = []
            for row in self.H_coarse:
                rows.append("  [ " + " ".join(f"{val:10.5f}" for val in row) + " ]")
            h_matrix_str = "\n".join(rows)

        html = f"""
        <h3>Target Acquisition Registration Report</h3>
        <hr>
        <b>TEST FRAME:</b> {os.path.basename(self.curr_image_path)}<br>
        <b>IMAGE PATH:</b> {self.curr_image_path}<br>
        <br>
        <b>1. APRILTAG DETECTION STATUS:</b><br>
        - Observed Tags: {self.det_status}<br>
        - Detected ID list: {[tag['tag_id'] for tag in self.observed_tags]}<br>
        - Tag Detect Time: {t_detect:.1f} ms<br>
        <br>
        <b>2. COARSE REGISTRATION DETAILS:</b><br>
        - Homography Sane Check: <font color='{status_color}'><b>{self.homography_status}</b></font><br>
        - Registration Confidence: <b>{self.confidence:.3f}</b><br>
        - Reprojection Error: <b>{self.reproj_err_px:.3f} px</b> (mean distance)<br>
        - Homography Fit Time: {t_homo:.1f} ms<br>
        <br>
        <b>3. COMPUTED HOMOGRAPHY MATRIX (H_coarse):</b><br>
        <pre>{h_matrix_str}</pre>
        <hr>
        <b>4. TARGET DESIGN DIMENSIONS:</b><br>
        - AprilTag Size: {self.blueprint.tag_size_mm} mm<br>
        - Design Width: {self.blueprint.target_width_mm} mm<br>
        - Design Height: {self.blueprint.target_height_mm} mm<br>
        - Stored scale: {self.blueprint.pixels_per_mm:.6f} px/mm<br>
        """
        self.details_box.setHtml(html)


def main():
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)
    window = RegistrationDebuggerApp()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
