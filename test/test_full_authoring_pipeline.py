from __future__ import annotations

import sys
from pathlib import Path

# Add project root to sys.path
sys.path.append(str(Path(__file__).parent.parent.resolve()))

import cv2
import numpy as np
import matplotlib.pyplot as plt

from PySide6.QtWidgets import QApplication, QMessageBox

from core.calibration.image_loader import load_image
from core.roi.selector import ROISelector
from core.detection.silhouette_detector import detect_silhouette_candidates
from core.ui.candidate_selector import SilhouetteCandidateSelector
from core.ui.contour_editor import ContourEditor
from core.ui.zone_editor import ZonePolygonEditor
from core.ui.feature_editor import FeatureRegionEditor
from core.models import Zone, FeatureRegion


def print_full_pipeline_summary(final_contour: np.ndarray, final_zones: list[Zone], final_regions: list[FeatureRegion]) -> None:
    """
    Print a neat text summary of the finalized target blueprint components.
    """
    print()
    print("=" * 80)
    print("TBAS FULL AUTHORING PIPELINE - COMPLETE BLUEPRINT SUMMARY")
    print("=" * 80)
    
    # 1. Silhouette Statistics
    sil_area = cv2.contourArea(final_contour)
    sil_perimeter = cv2.arcLength(final_contour, closed=True)
    print(f"[1. TARGET SILHOUETTE BOUNDARY]")
    print(f"  Total Vertices   : {len(final_contour)}")
    print(f"  Area (px²)       : {sil_area:.1f}")
    print(f"  Perimeter (px)   : {sil_perimeter:.1f}")
    print("-" * 50)
    
    # 2. Zone Statistics
    print(f"[2. AUTHORED SCORING ZONES] - Count: {len(final_zones)}")
    for idx, zone in enumerate(final_zones):
        area = cv2.contourArea(zone.polygon)
        perimeter = cv2.arcLength(zone.polygon, closed=True)
        pts = zone.polygon.reshape(-1, 2)
        print(f"  Zone '{zone.zone_id}': score={zone.score:.1f}, name='{zone.name}', vertices={len(pts)}, area={area:.1f} px²")
    print("-" * 50)
    
    # 3. Feature Regions Statistics
    stable_cnt = sum(1 for r in final_regions if r.region_type == "stable")
    corner_cnt = sum(1 for r in final_regions if r.region_type == "zone_corner")
    optional_cnt = sum(1 for r in final_regions if r.region_type == "optional")
    print(f"[3. AUTHORED FEATURE REGIONS] - Count: {len(final_regions)}")
    print(f"  Stable Regions       : {stable_cnt}")
    print(f"  Zone Corner Anchors  : {corner_cnt}")
    print(f"  Optional Regions     : {optional_cnt}")
    for idx, region in enumerate(final_regions):
        pts = region.polygon.reshape(-1, 2)
        print(f"    - Region '{region.id}': Type={region.region_type}, Priority={region.priority}, Vertices={len(pts)}")
    print("=" * 80)


def main():
    image_path = "test_images/outdoor target.jpeg"
    if len(sys.argv) > 1:
        image_path = sys.argv[1]

    # Initialize Qt application
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)

    # ---------------------------------------------------------
    # STEP 1: Load Source Image
    # ---------------------------------------------------------
    print(f"Loading source target image: {image_path}")
    image_data = load_image(image_path)

    # ---------------------------------------------------------
    # STEP 2: Select ROI (Target Bounding Box)
    # ---------------------------------------------------------
    print("Opening ROI Selection Dialog...")
    roi_selector = ROISelector(image_data)
    result = roi_selector.exec()

    if result != ROISelector.DialogCode.Accepted:
        print("Authoring pipeline cancelled: ROI selection rejected.")
        return

    roi = roi_selector.roi
    if roi is None:
        print("Authoring pipeline cancelled: No valid ROI selected.")
        return

    print(f"Selected ROI: x={roi.x}, y={roi.y}, w={roi.width}, h={roi.height}")

    # ---------------------------------------------------------
    # STEP 3: Automatic Silhouette Generation
    # ---------------------------------------------------------
    print("Running automatic silhouette candidate generation...")
    detection_result = detect_silhouette_candidates(
        image_data,
        roi,
        hessian_threshold=250,
        max_candidates=10,
    )

    # ---------------------------------------------------------
    # STEP 4: Choose Silhouette Candidate
    # ---------------------------------------------------------
    print("Opening Silhouette Candidate Selector...")
    candidate_selector = SilhouetteCandidateSelector(
        image_data=image_data,
        roi=roi,
        candidates=detection_result.candidates,
        white_contour=detection_result.white_mask_contour
    )
    result = candidate_selector.exec()

    if result != SilhouetteCandidateSelector.DialogCode.Accepted:
        print("Authoring pipeline cancelled: Candidate selection rejected.")
        return

    initial_contour = candidate_selector.selected_contour
    if initial_contour is None:
        print("Authoring pipeline cancelled: No initial contour selected.")
        return

    # ---------------------------------------------------------
    # STEP 5: Human Contour Correction (Contour Editor)
    # ---------------------------------------------------------
    print("Opening Silhouette Contour Editor...")
    contour_editor = ContourEditor(
        image_data=image_data,
        roi=roi,
        contour=initial_contour
    )
    result = contour_editor.exec()

    if result != ContourEditor.DialogCode.Accepted:
        print("Authoring pipeline cancelled: Contour editing discarded.")
        return

    final_contour = contour_editor.final_contour
    if final_contour is None:
        print("Authoring pipeline cancelled: No valid contour returned.")
        return

    print(f"Final Target Silhouette saved with {len(final_contour)} vertices.")

    # ---------------------------------------------------------
    # STEP 6: Scoring Zones Creation (Zone Editor)
    # ---------------------------------------------------------
    print("Opening Scoring Zone Editor...")
    zone_editor = ZonePolygonEditor(
        image_data=image_data,
        roi=roi,
        initial_zones=[],
        silhouette_contour=final_contour
    )
    result = zone_editor.exec()

    if result != ZonePolygonEditor.DialogCode.Accepted:
        print("Authoring pipeline cancelled: Zone editing discarded.")
        return

    final_zones = zone_editor.final_zones
    if not final_zones:
        print("Authoring pipeline completed but no zones were saved.")
        return

    print(f"Authored {len(final_zones)} scoring zones.")

    # ---------------------------------------------------------
    # STEP 7: Landmark & Feature Region Authoring (Feature Editor)
    # ---------------------------------------------------------
    print("Opening Landmark & Feature Region Editor...")
    feature_editor = FeatureRegionEditor(
        image_data=image_data,
        roi=roi,
        initial_regions=[],
        zones=final_zones,
        silhouette_contour=final_contour
    )
    result = feature_editor.exec()

    if result != FeatureRegionEditor.DialogCode.Accepted:
        print("Authoring pipeline cancelled: Feature region editing discarded.")
        return

    final_regions = feature_editor.final_regions

    # ---------------------------------------------------------
    # STEP 8: Print Summary and Render Dual-Mode Visualizations
    # ---------------------------------------------------------
    print_full_pipeline_summary(final_contour, final_zones, final_regions)

    # Crop the ROI to display final result
    cropped_bgr = roi.crop(image_data.image)
    cropped_rgb = cv2.cvtColor(cropped_bgr, cv2.COLOR_BGR2RGB)

    # Mode 1: Target Geometry (Just Silhouette & Scoring Zones)
    fig1, ax1 = plt.subplots(figsize=(8, 8))
    ax1.imshow(cropped_rgb)
    ax1.set_title(f"Mode 1: Target Geometry (Zones: {len(final_zones)})")
    
    # Draw silhouette
    sil_local = final_contour - np.array([[[roi.x, roi.y]]], dtype=np.int32)
    sil_draw = np.vstack([sil_local.reshape(-1, 2), sil_local.reshape(-1, 2)[0]])
    ax1.plot(sil_draw[:, 0], sil_draw[:, 1], color="blue", linestyle="--", linewidth=2.0, label="Silhouette Outline")
        
    # Plot scoring zones (solid lines)
    zone_colors = plt.colormaps["tab10"]
    for idx, zone in enumerate(final_zones):
        pts = zone.polygon.reshape(-1, 2)
        pts_draw = np.vstack([pts, pts[0]])
        c = zone_colors(idx)
        ax1.plot(pts_draw[:, 0], pts_draw[:, 1], color=c, linewidth=2.0, label=f"Zone {zone.zone_id}")
        ax1.fill(pts[:, 0], pts[:, 1], color=c, alpha=0.10)
    ax1.legend(loc="upper right")
    ax1.axis("off")
    fig1.tight_layout()

    # Mode 2: Full Blueprint (Including Feature Regions)
    fig2, ax2 = plt.subplots(figsize=(8, 8))
    ax2.imshow(cropped_rgb)
    ax2.set_title(f"Mode 2: Full Blueprint (Feature Regions: {len(final_regions)})")
    
    # Draw silhouette (faded)
    ax2.plot(sil_draw[:, 0], sil_draw[:, 1], color="blue", linestyle="--", linewidth=1.5, alpha=0.7)
        
    # Plot scoring zones (faded)
    for idx, zone in enumerate(final_zones):
        pts = zone.polygon.reshape(-1, 2)
        pts_draw = np.vstack([pts, pts[0]])
        c = zone_colors(idx)
        ax2.plot(pts_draw[:, 0], pts_draw[:, 1], color=c, linewidth=1.5, alpha=0.6)
        ax2.fill(pts[:, 0], pts[:, 1], color=c, alpha=0.05)

    # Plot stable feature regions (dashed lines)
    for idx, region in enumerate(final_regions):
        pts = region.polygon.reshape(-1, 2)
        pts_draw = np.vstack([pts, pts[0]])
        
        if region.region_type == "stable":
            color = "purple"
            style = "--"
        elif region.region_type == "optional":
            color = "dodgerblue"
            style = "-."
        else:  # zone_corner (opportunistic)
            color = "orange"
            style = ":"
            
        ax2.plot(pts_draw[:, 0], pts_draw[:, 1], color=color, linestyle=style, linewidth=2.0, 
                label=f"Region {region.id} ({region.region_type})" if idx < 5 else "")
                
    ax2.legend(loc="upper right")
    ax2.axis("off")
    fig2.tight_layout()

    print("Displaying final blueprint plots. Close Matplotlib windows to exit.")
    plt.show()


if __name__ == "__main__":
    main()
