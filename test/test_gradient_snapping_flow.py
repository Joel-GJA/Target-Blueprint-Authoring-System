from __future__ import annotations

import sys
from pathlib import Path

# Add project root to sys.path
sys.path.append(str(Path(__file__).parent.parent.resolve()))

import cv2
import numpy as np
import matplotlib.pyplot as plt

# Matplotlib backends for embedding in PySide6 QDialog
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT
from matplotlib.figure import Figure

from PySide6.QtCore import QRectF, Qt
from PySide6.QtWidgets import (
    QApplication,
    QPushButton,
    QComboBox,
    QLabel,
    QDialog,
    QVBoxLayout,
    QHBoxLayout,
    QMessageBox
)

from core.calibration.image_loader import load_image
from core.roi.selector import ROISelector
from core.ui.zone_editor import ZonePolygonEditor

IMAGE_PATH = "test_images/outdoor target.jpeg"


class GradientDebugDialog(QDialog):
    """
    Native PySide6 resizable dialog wrapping the experimental edge localization plots.
    Embeds the Matplotlib Canvas directly, preventing display scaling, DPI, or screen placement issues.
    """
    def __init__(
        self,
        result,
        crop_rgb,
        p1,
        p2,
        normal,
        search_margin,
        edge_idx,
        zone_id,
        parent=None
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle(f"Scoring Zone Edge Snapping Experiment ({zone_id} - Edge {edge_idx + 1})")
        self.resize(1100, 850)
        
        # Native layouts to ensure scalability
        layout = QVBoxLayout(self)
        
        info = QLabel(
            "<b>Scoring Zone Edge Snapping Experiment</b> — showing perpendicular intensity profile gradients along the chosen boundary.<br>"
            "<i>This window is fully resizable. Drag the corners to scale the plots.</i>"
        )
        layout.addWidget(info)
        
        # Embed Matplotlib Figure & Canvas
        self.figure = Figure(figsize=(15, 9), dpi=100)
        self.canvas = FigureCanvasQTAgg(self.figure)
        layout.addWidget(self.canvas)
        
        # Add Matplotlib interactive toolbar for zooming and panning
        self.toolbar = NavigationToolbar2QT(self.canvas, self)
        layout.addWidget(self.toolbar)
        
        # Close button at the bottom
        btn_layout = QHBoxLayout()
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.accept)
        btn_layout.addStretch()
        btn_layout.addWidget(close_btn)
        layout.addLayout(btn_layout)
        
        # Generate the visualization inside the figure
        self.generate_plots(result, crop_rgb, p1, p2, normal, search_margin, edge_idx, zone_id)
        
    def generate_plots(self, result, crop_rgb, p1, p2, normal, search_margin, edge_idx, zone_id) -> None:
        v_edge = p2.astype(np.float32) - p1.astype(np.float32)
        length = np.linalg.norm(v_edge)
        
        # Clear figure axes
        self.figure.clear()
        
        # View 1 — Original ROI with overlays
        ax1 = self.figure.add_subplot(2, 3, 1)
        ax1.imshow(crop_rgb)
        ax1.plot([p1[0], p2[0]], [p1[1], p2[1]], color="red", linewidth=1.2, label="Original Rough Edge")
        p1_ref, p2_ref = result.refined_edge
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
        ax5.plot(result.offsets, result.aggregate_response, color="blue", linewidth=2.0)
        ax5.axvline(0, color="red", linestyle="--", label="User Edge (0)")
        ax5.axvline(result.offset_pixels, color="lime", linestyle="--", label=f"Refined Peak ({result.offset_pixels:+.1f} px)")
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
            f"Detected Offset: {result.offset_pixels:+.1f} px\n"
            f"Peak Strength: {result.peak_strength:.1f}\n"
            f"Confidence Ratio: {result.confidence:.3f}\n"
            f"Localization Success: {result.success}\n"
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


class GradientZonePolygonEditor(ZonePolygonEditor):
    """
    Subclass of ZonePolygonEditor adding the experimental local gradient snapping button and options.
    Overrides snap_active_zone to rewire the snapping to the new gradient-profile algorithm.
    """
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        
        # Access the refinement layout in the sidebar group
        ref_layout = self.refinement_group.layout()
        
        # Add Combo box to select which of the 4 polygon edges to analyze
        ref_layout.addWidget(QLabel("<b>Gradient Snap Edge:</b>"))
        self.edge_combo = QComboBox(self)
        self.edge_combo.addItems([
            "Edge 1 (P1 -> P2)",
            "Edge 2 (P2 -> P3)",
            "Edge 3 (P3 -> P4)",
            "Edge 4 (P4 -> P1)"
        ])
        ref_layout.addWidget(self.edge_combo)
        
        # Add the experimental "Test Gradient Snapping" button
        self.test_grad_btn = QPushButton("Test Gradient Snapping", self)
        self.test_grad_btn.setToolTip("Run experimental gradient-profile edge refinement on the chosen edge (Matplotlib Popup)")
        self.test_grad_btn.clicked.connect(self.run_gradient_snapping_experiment)
        self.test_grad_btn.setEnabled(False)
        ref_layout.addWidget(self.test_grad_btn)
        
    def _select_zone(self, idx: int | None) -> None:
        super()._select_zone(idx)
        has_selection = idx is not None
        self.test_grad_btn.setEnabled(has_selection)
        
    def snap_active_zone(self) -> None:
        """
        Rewires the production 'Snap Selected Zone' button to use the new
        experimental gradient-based snapping and line intersection algorithm.
        """
        idx = self.active_zone_index
        if idx is None:
            return
            
        zone = self.zones[idx]
        self.save_undo_state()
        
        # Convert to ROI-local coordinates for snapping computations
        roi_local_poly = zone.polygon - np.array([[[self.roi.x, self.roi.y]]], dtype=np.int32)
        search_margin = float(self.margin_spin.value())
        
        print(f"[Experimental Snapper] Snapping {zone.zone_id} with gradient-profile math, margin={search_margin} px...")
        
        from core.experiments.gradient_edge_localizer import snap_polygon_gradient
        snapped_roi_local = snap_polygon_gradient(
            image_data=self.image_data,
            roi=self.roi,
            polygon=roi_local_poly,
            search_margin=search_margin,
            sample_spacing=5.0,
            aggregation_method="median"
        )
        
        # Convert result back to whole-image coordinates
        snapped_whole_image = snapped_roi_local + np.array([[[self.roi.x, self.roi.y]]], dtype=np.int32)
        zone.polygon = snapped_whole_image
        
        # Re-render canvas items
        self.recreate_handles()
        self.redraw_active_lines()
        self._update_status()
        print("Experimental gradient snapping complete.")
        
    def run_gradient_snapping_experiment(self) -> None:
        idx = self.active_zone_index
        if idx is None:
            return
            
        zone = self.zones[idx]
        roi_local_poly = zone.polygon - np.array([[[self.roi.x, self.roi.y]]], dtype=np.int32)
        pts = roi_local_poly.reshape(-1, 2)
        
        if len(pts) != 4:
            QMessageBox.warning(self, "Invalid Zone", "Experimental gradient edge localizer requires a 4-sided scoring zone.")
            return
            
        # Select active edge endpoints
        edge_idx = self.edge_combo.currentIndex()
        p1 = pts[edge_idx]
        p2 = pts[(edge_idx + 1) % 4]
        
        search_margin = float(self.margin_spin.value())
        
        # 1. Run local gradient-based snapping algorithm
        from core.experiments.gradient_edge_localizer import localize_edge
        result = localize_edge(
            image_data=self.image_data,
            roi=self.roi,
            p1=p1,
            p2=p2,
            search_margin=search_margin,
            sample_spacing=5.0,
            aggregation_method="median"
        )
        
        # 2. Extract crops and prepare drawing
        crop = self.roi.crop(self.image_data.image)
        crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        
        # Calculate normal direction for View 4 lines
        v_edge = p2.astype(np.float32) - p1.astype(np.float32)
        length = np.linalg.norm(v_edge)
        tangent = v_edge / (length + 1e-5)
        normal = np.array([-tangent[1], tangent[0]], dtype=np.float32)
        
        # 3. Launch native resizable Qt debug dialog
        debug_dlg = GradientDebugDialog(
            result=result,
            crop_rgb=crop_rgb,
            p1=p1,
            p2=p2,
            normal=normal,
            search_margin=search_margin,
            edge_idx=edge_idx,
            zone_id=zone.zone_id,
            parent=self
        )
        debug_dlg.exec()


def main():
    image_path = IMAGE_PATH
    if len(sys.argv) > 1:
        image_path = sys.argv[1]

    # Initialize Qt
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)

    # 1. Load source image
    print(f"Loading source image: {image_path}")
    image_data = load_image(image_path)

    # 2. Select ROI (Target bounding box)
    roi_path = Path("test_images/roi.json")
    roi = None
    if roi_path.exists():
        from core.models import ROI
        roi = ROI.load(roi_path)
        if roi is not None:
            print(f"Loaded saved ROI coordinates from: {roi_path} (x={roi.x}, y={roi.y}, w={roi.width}, h={roi.height})")

    if roi is None:
        print("Opening ROI Selection Dialog...")
        roi_selector = ROISelector(image_data)
        result = roi_selector.exec()

        if result != ROISelector.DialogCode.Accepted:
            print("Zoning workflow cancelled: ROI selection rejected.")
            return

        roi = roi_selector.roi
        if roi is None:
            print("Zoning workflow cancelled: No valid ROI selected.")
            return
        roi.save(str(roi_path))

    print(f"Using ROI: x={roi.x}, y={roi.y}, w={roi.width}, h={roi.height}")

    # 3. Load target silhouette contour if available
    silhouette_contour = None
    contour_path = Path("test_images/silhouette_contour.npy")
    if contour_path.exists():
        try:
            silhouette_contour = np.load(str(contour_path))
            print(f"Loaded saved target silhouette boundary: {contour_path}")
        except Exception as e:
            print(f"Warning: Failed to load silhouette contour: {e}")

    starter_zones = []

    # 4. Open experimental GradientZonePolygonEditor
    print("Opening Experimental Scoring Zone Editor...")
    editor = GradientZonePolygonEditor(
        image_data=image_data,
        roi=roi,
        initial_zones=starter_zones,
        silhouette_contour=silhouette_contour
    )
    editor.exec()


if __name__ == "__main__":
    main()
