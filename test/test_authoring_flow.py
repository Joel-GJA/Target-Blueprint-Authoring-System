from __future__ import annotations

import sys
from pathlib import Path

# Add project root to sys.path
sys.path.append(str(Path(__file__).parent.parent.resolve()))

import cv2
import matplotlib.pyplot as plt
import numpy as np

from PySide6.QtWidgets import QApplication, QMessageBox

from core.image_loader import load_image
from core.roi_selector import ROISelector
from core.silhouette_detector import detect_silhouette_candidates
from core.ui.candidate_selector import SilhouetteCandidateSelector
from core.ui.contour_editor import ContourEditor

# Default image path (can be overridden via command-line arguments)
IMAGE_PATH = "test_images/target_preview_fig11_50mm_tags_with_cutt_out.png"


def print_final_contour_info(contour: np.ndarray) -> None:
    """
    Print the final edited contour's geometric statistics.
    """
    print()
    print("=" * 80)
    print("FINAL AUTHORED CONTOUR STATISTICS")
    print("=" * 80)
    
    area = cv2.contourArea(contour)
    perimeter = cv2.arcLength(contour, closed=True)
    x, y, w, h = cv2.boundingRect(contour)
    
    # Solidity
    hull = cv2.convexHull(contour)
    hull_area = cv2.contourArea(hull)
    solidity = area / hull_area if hull_area > 0 else 0.0
    
    print(f"  Total Vertices : {len(contour)}")
    print(f"  Area (px²)     : {area:.1f}")
    print(f"  Perimeter (px) : {perimeter:.1f}")
    print(f"  Bounding Box   : (x={x}, y={y}, w={w}, h={h})")
    print(f"  Solidity       : {solidity:.4f}")
    print("=" * 80)


def main():
    # ---------------------------------------------------------
    # Command-line image override
    # ---------------------------------------------------------
    image_path = IMAGE_PATH
    if len(sys.argv) > 1:
        image_path = sys.argv[1]

    # ---------------------------------------------------------
    # Qt Application
    # ---------------------------------------------------------
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)

    # ---------------------------------------------------------
    # 1. Load Source Image
    # ---------------------------------------------------------
    print(f"Loading source image from: {image_path}")
    image_data = load_image(image_path)

    # ---------------------------------------------------------
    # 2. Select ROI (Target Boundary Box)
    # ---------------------------------------------------------
    print("Opening ROI Selection Dialog...")
    roi_selector = ROISelector(image_data)
    result = roi_selector.exec()

    if result != ROISelector.DialogCode.Accepted:
        print("Authoring cancelled: ROI selection rejected.")
        return

    roi = roi_selector.roi
    if roi is None:
        print("Authoring cancelled: No valid ROI selected.")
        return

    print(f"Selected ROI: x={roi.x}, y={roi.y}, w={roi.width}, h={roi.height}")

    # ---------------------------------------------------------
    # 3. Run Automatic Silhouette Detection
    # ---------------------------------------------------------
    print("Running automatic silhouette candidate generation...")
    detection_result = detect_silhouette_candidates(
        image_data,
        roi,
        hessian_threshold=250,
        max_candidates=10,
    )

    # ---------------------------------------------------------
    # 4. Open Silhouette Candidate Selector
    # ---------------------------------------------------------
    print("Opening Candidate Selector...")
    candidate_selector = SilhouetteCandidateSelector(
        image_data=image_data,
        roi=roi,
        candidates=detection_result.candidates,
        white_contour=detection_result.white_mask_contour
    )
    result = candidate_selector.exec()

    if result != SilhouetteCandidateSelector.DialogCode.Accepted:
        print("Authoring cancelled: Candidate selection rejected.")
        return

    initial_contour = candidate_selector.selected_contour
    if initial_contour is None:
        print("Authoring cancelled: No initial contour selected.")
        return

    print("Initial contour selected. Opening Contour Editor...")

    # ---------------------------------------------------------
    # 5. Open Contour Editor (Human correction phase)
    # ---------------------------------------------------------
    editor = ContourEditor(
        image_data=image_data,
        roi=roi,
        contour=initial_contour
    )
    result = editor.exec()

    if result != ContourEditor.DialogCode.Accepted:
        print("Authoring cancelled: Editing session discarded.")
        return

    final_contour = editor.final_contour
    if final_contour is None:
        print("Authoring cancelled: No valid final contour returned.")
        return

    # ---------------------------------------------------------
    # 6. Save/Display final authorized result
    # ---------------------------------------------------------
    print_final_contour_info(final_contour)

    # Crop the ROI to display final result
    cropped_bgr = roi.crop(image_data.image)
    
    # The final_contour is in whole-image coordinates. Shift it to ROI-local space for drawing on cropped image.
    roi_local_contour = final_contour - np.array([[[roi.x, roi.y]]], dtype=np.int32)
    
    # Draw the final contour (shifted to ROI-local) on the crop
    final_visualization = cropped_bgr.copy()
    cv2.drawContours(final_visualization, [roi_local_contour], -1, (0, 0, 255), 3) # Draw in bold Red
    
    final_visualization_rgb = cv2.cvtColor(final_visualization, cv2.COLOR_BGR2RGB)
    cropped_rgb = cv2.cvtColor(cropped_bgr, cv2.COLOR_BGR2RGB)

    # Display before vs after
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    
    axes[0].imshow(cropped_rgb)
    axes[0].set_title("Cropped ROI (Raw)")
    axes[0].axis("off")
    
    axes[1].imshow(final_visualization_rgb)
    axes[1].set_title(f"Final Authorized Contour\n(Vertices: {len(final_contour)})")
    axes[1].axis("off")
    
    plt.tight_layout()
    print("Displaying final blueprint result. Close Matplotlib to exit.")
    plt.show()


if __name__ == "__main__":
    main()
