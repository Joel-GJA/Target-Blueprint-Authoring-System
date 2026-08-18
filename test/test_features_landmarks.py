import sys
import json
from pathlib import Path
import numpy as np
import cv2
import matplotlib.pyplot as plt

# Add project root to sys.path
sys.path.append(str(Path(__file__).parent.parent.resolve()))

from PySide6.QtWidgets import QApplication

from core.models import ROI, FeatureRegion, Zone
from core.calibration.image_loader import load_image
from core.roi import ROISelector
from core.ui.zone_editor import ZonePolygonEditor
from core.ui.feature_editor import FeatureRegionEditor


def print_test_summary(regions: list[FeatureRegion], zones: list[Zone]) -> None:
    """
    Print a clean statistics summary of the authored blueprint components.
    """
    print()
    print("=" * 80)
    print("TBAS FEATURE & LANDMARK AUTHORING - WORKFLOW SUMMARY")
    print("=" * 80)
    
    # 1. Zone Statistics
    print(f"[1. AUTHORED SCORING ZONES] - Count: {len(zones)}")
    for idx, zone in enumerate(zones):
        pts = zone.polygon.reshape(-1, 2)
        print(f"  Zone '{zone.zone_id}': score={zone.score:.1f}, name='{zone.name}', corners={len(pts)}")
    print("-" * 50)
    
    # 2. Region Statistics
    stable_cnt = sum(1 for r in regions if r.region_type == "stable")
    corner_cnt = sum(1 for r in regions if r.region_type == "zone_corner")
    optional_cnt = sum(1 for r in regions if r.region_type == "optional")
    
    print(f"[2. AUTHORED FEATURE REGIONS] - Count: {len(regions)}")
    print(f"  Stable Regions       : {stable_cnt}")
    print(f"  Zone Corner Anchors  : {corner_cnt}")
    print(f"  Optional Regions     : {optional_cnt}")
    print("  Details (ROI-local):")
    for idx, region in enumerate(regions):
        pts = region.polygon.reshape(-1, 2)
        print(f"    Region #{idx + 1}: ID={region.id}, Type={region.region_type}, Priority={region.priority}, Corners={len(pts)}")
    print("=" * 80)


def main():
    image_path = "test_images/outdoor target.jpeg"
    if len(sys.argv) > 1:
        image_path = sys.argv[1]

    # Initialize Qt Application
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)

    # 1. Load Source Image
    print(f"Loading source target image: {image_path}")
    image_data = load_image(image_path)

    # 2. Select or Load Target ROI (Bounding Box)
    roi_path = Path("test_images/roi.json")
    roi = None
    if roi_path.exists():
        roi = ROI.load(roi_path)
        if roi is not None:
            print(f"Loaded existing ROI coordinates from: {roi_path}")

    if roi is None:
        print("Opening ROI Selection Dialog...")
        roi_selector = ROISelector(image_data)
        result = roi_selector.exec()

        if result != ROISelector.DialogCode.Accepted:
            print("Authoring workflow cancelled: ROI selection rejected.")
            return

        roi = roi_selector.roi
        if roi is None:
            print("Authoring workflow cancelled: No valid ROI selected.")
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

    # 4. Open ZonePolygonEditor to author scoring zones
    print("Opening Scoring Zone Editor...")
    zone_editor = ZonePolygonEditor(
        image_data=image_data,
        roi=roi,
        initial_zones=[],
        silhouette_contour=silhouette_contour
    )
    result = zone_editor.exec()

    if result != ZonePolygonEditor.DialogCode.Accepted:
        print("Authoring workflow cancelled: Zone editing discarded.")
        return

    final_zones = zone_editor.final_zones
    if not final_zones:
        print("Authoring workflow cancelled: No zones saved.")
        return

    print(f"Authored {len(final_zones)} scoring zones.")

    # 5. Open FeatureRegionEditor to author Stable Feature Regions & Landmarks
    print("Opening Landmark & Feature Region Editor...")
    feature_editor = FeatureRegionEditor(
        image_data=image_data,
        roi=roi,
        initial_regions=[],
        zones=final_zones,
        silhouette_contour=silhouette_contour
    )
    result = feature_editor.exec()

    if result != FeatureRegionEditor.DialogCode.Accepted:
        print("Authoring workflow cancelled: Landmark editing discarded.")
        return

    final_regions = feature_editor.final_regions

    # 6. Output Statistics Summary
    print_test_summary(final_regions, final_zones)

    # 7. Render final visual blueprint representation
    cropped_bgr = roi.crop(image_data.image)
    cropped_rgb = cv2.cvtColor(cropped_bgr, cv2.COLOR_BGR2RGB)

    # Figure 1: Target Geometry (Just Silhouette & Scoring Zones)
    fig1, ax1 = plt.subplots(figsize=(8, 8))
    ax1.imshow(cropped_rgb)
    ax1.set_title(f"Mode 1: Target Geometry (Zones: {len(final_zones)})")
    
    # Draw silhouette outline if available
    if silhouette_contour is not None:
        sil_local = silhouette_contour - np.array([[[roi.x, roi.y]]], dtype=np.int32)
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

    # Figure 2: Visual Features & Landmarks (Including Feature Regions)
    fig2, ax2 = plt.subplots(figsize=(8, 8))
    ax2.imshow(cropped_rgb)
    ax2.set_title(f"Mode 2: Full Blueprint (Feature Regions: {len(final_regions)})")
    
    # Draw silhouette outline if available (faded)
    if silhouette_contour is not None:
        ax2.plot(sil_draw[:, 0], sil_draw[:, 1], color="blue", linestyle="--", linewidth=1.5, alpha=0.7)
        
    # Plot scoring zones (faded solid lines)
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
