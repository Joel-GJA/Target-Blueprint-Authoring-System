from __future__ import annotations

import sys
from pathlib import Path

# Add project root to sys.path
sys.path.append(str(Path(__file__).parent.parent.resolve()))

import cv2
import numpy as np
import matplotlib.pyplot as plt

from PySide6.QtWidgets import QApplication

from core.image_loader import load_image
from core.roi_selector import ROISelector
from core.silhouette_detector import detect_silhouette_candidates
from core.ui.candidate_selector import SilhouetteCandidateSelector
from core.ui.contour_editor import ContourEditor
from core.ui.zone_editor import ZonePolygonEditor

IMAGE_PATH = "test_images/outdoor target.jpeg"


def print_milestone_summary(final_contour: np.ndarray, final_zones: list) -> None:
    """
    Print a neat text summary of the finalized target blueprint components.
    """
    print()
    print("=" * 80)
    print("TBAS OFFLINE AUTHORING PIPELINE - MILESTONE SUMMARY")
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
    print(f"[2. AUTHORED SCORING ZONES]")
    for idx, zone in enumerate(final_zones):
        area = cv2.contourArea(zone.polygon)
        perimeter = cv2.arcLength(zone.polygon, closed=True)
        pts = zone.polygon.reshape(-1, 2)
        print(f"  Zone #{idx + 1}:")
        print(f"    ID/Label   : {zone.zone_id}")
        print(f"    Name       : {zone.name if zone.name else 'N/A'}")
        print(f"    Points     : {zone.score:.2f}")
        print(f"    Vertices   : {len(pts)}")
        print(f"    Area (px²) : {area:.1f}")
        print(f"    Perimeter  : {perimeter:.1f}")
        print(f"    Coordinates (ROI-local):")
        for p_idx, pt in enumerate(pts):
            print(f"      - Point {p_idx + 1}: ({pt[0]}, {pt[1]})")
        print("  " + "." * 40)
        
    print("=" * 80)


def main():
    image_path = IMAGE_PATH
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
    # Pass the final target silhouette contour directly so:
    # 1. It renders as a blue dashed reference line.
    # 2. It automatically adds the 'Outer Silhouette' zone representing the remaining area.
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

    # ---------------------------------------------------------
    # STEP 7: Print Milestone Summary and Display Plots
    # ---------------------------------------------------------
    print_milestone_summary(final_contour, final_zones)

    # Crop the ROI to display final result
    cropped_bgr = roi.crop(image_data.image)
    cropped_rgb = cv2.cvtColor(cropped_bgr, cv2.COLOR_BGR2RGB)

    # Visualization plot
    fig, ax = plt.subplots(figsize=(9, 9))
    ax.imshow(cropped_rgb)
    ax.set_title(f"Final Authorized Blueprint (Zones: {len(final_zones)})")
    
    # Generate distinct colors for visualization
    colors = plt.colormaps["tab10"]
    
    for idx, zone in enumerate(final_zones):
        pts = zone.polygon.reshape(-1, 2)
        pts_draw = np.vstack([pts, pts[0]])
        c = colors(idx)
        
        # Plot outline and fill
        ax.plot(
            pts_draw[:, 0],
            pts_draw[:, 1],
            color=c,
            linewidth=2.5,
            label=f"{zone.zone_id} (Score: {zone.score})"
        )
        ax.fill(pts[:, 0], pts[:, 1], color=c, alpha=0.15)
        
    ax.legend(loc="upper right")
    ax.axis("off")
    plt.tight_layout()
    print("Displaying final blueprint result. Close Matplotlib window to exit.")
    plt.show()


if __name__ == "__main__":
    main()
