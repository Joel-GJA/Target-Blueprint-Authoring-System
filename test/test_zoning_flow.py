from __future__ import annotations

import sys
from pathlib import Path

# Add project root to sys.path
sys.path.append(str(Path(__file__).parent.parent.resolve()))

import cv2
import matplotlib.pyplot as plt
import numpy as np

from PySide6.QtWidgets import QApplication

from core.calibration.image_loader import load_image
from core.roi.selector import ROISelector
from core.ui.zone_editor import ZonePolygonEditor

IMAGE_PATH = "test_images/outdoor target.jpeg"


def print_final_zones_info(zones: list) -> None:
    """
    Print the final authored zones' coordinates and metadata.
    """
    print()
    print("=" * 80)
    print("FINAL AUTHORED SCORING ZONES STATISTICS")
    print("=" * 80)
    
    for idx, zone in enumerate(zones):
        area = cv2.contourArea(zone.polygon)
        perimeter = cv2.arcLength(zone.polygon, closed=True)
        pts = zone.polygon.reshape(-1, 2)
        
        print(f"Zone #{idx + 1}:")
        print(f"  ID/Label   : {zone.zone_id}")
        print(f"  Name       : {zone.name}")
        print(f"  Score      : {zone.score:.2f}")
        print(f"  Vertices   : {len(pts)}")
        print(f"  Area (px²) : {area:.1f}")
        print(f"  Perimeter  : {perimeter:.1f}")
        print(f"  Coords (ROI-local):")
        for p_idx, pt in enumerate(pts):
            print(f"    - Point {p_idx + 1}: ({pt[0]}, {pt[1]})")
        print("-" * 50)
    print("=" * 80)


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
        # Save ROI to a file so it can be reused in future zoning flows
        roi.save(str(roi_path))

    print(f"Using ROI: x={roi.x}, y={roi.y}, w={roi.width}, h={roi.height}")

    # 3. Load target silhouette contour if available (from test_authoring_flow)
    silhouette_contour = None
    contour_path = Path("test_images/silhouette_contour.npy")
    if contour_path.exists():
        try:
            silhouette_contour = np.load(str(contour_path))
            print(f"Loaded saved target silhouette boundary: {contour_path}")
        except Exception as e:
            print(f"Warning: Failed to load silhouette contour: {e}")

    starter_zones = []

    # 4. Open ZonePolygonEditor
    print("Opening Scoring Zone Editor...")
    editor = ZonePolygonEditor(
        image_data=image_data,
        roi=roi,
        initial_zones=starter_zones,
        silhouette_contour=silhouette_contour
    )
    result = editor.exec()

    if result != ZonePolygonEditor.DialogCode.Accepted:
        print("Zoning workflow cancelled: Editor session discarded.")
        return

    final_zones = editor.final_zones
    if not final_zones:
        print("Zoning workflow cancelled: No zones saved.")
        return

    # 5. Print and display results
    print_final_zones_info(final_zones)

    # Crop the ROI to display final result
    cropped_bgr = roi.crop(image_data.image)
    cropped_rgb = cv2.cvtColor(cropped_bgr, cv2.COLOR_BGR2RGB)

    # Visualization
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(cropped_rgb)
    ax.set_title(f"Authored Scoring Zones (Count: {len(final_zones)})")
    
    # Generate distinct colors for visualization
    colors = plt.colormaps["tab10"]
    
    for idx, zone in enumerate(final_zones):
        pts = zone.polygon.reshape(-1, 2)
        
        # Close the loop for drawing
        pts_draw = np.vstack([pts, pts[0]])
        
        # Get color
        c = colors(idx)
        
        # Plot outline
        ax.plot(pts_draw[:, 0], pts_draw[:, 1], color=c, linewidth=2.5, label=f"{zone.zone_id} (Score: {zone.score})")
        
        # Fill polygon
        ax.fill(pts[:, 0], pts[:, 1], color=c, alpha=0.15)
        
    ax.legend(loc="upper right")
    ax.axis("off")
    plt.tight_layout()
    print("Displaying final zoning result. Close Matplotlib window to exit.")
    plt.show()


if __name__ == "__main__":
    main()
