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
from core.detection.hessian_detector import detect_hessian_ridges
from core.detection.hsv_detector import detect_hsv_evidence

# Default image path (can be overridden via command-line arguments)
IMAGE_PATH = "test_images/outdoor target.jpeg"


def print_contour_statistics(contours: list[np.ndarray]) -> None:
    """
    Compute and print statistics for the extracted contours.
    """
    print()
    print("=" * 80)
    print("WHITE MASK CONTOUR ANALYSIS (MANUAL EXTRACTION)")
    print("=" * 80)
    print(f"Total contours found: {len(contours)}")
    
    # Sort contours by area in descending order
    sorted_contours = sorted(contours, key=cv2.contourArea, reverse=True)
    
    for idx, contour in enumerate(sorted_contours[:5]):
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, closed=True)
        x, y, w, h = cv2.boundingRect(contour)
        
        # Calculate solidity
        hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull)
        solidity = area / hull_area if hull_area > 0 else 0.0
        
        print(f"\nContour #{idx + 1}:")
        print(f"  Area (px²)    : {area:.1f}")
        print(f"  Perimeter (px): {perimeter:.1f}")
        print(f"  Bounding Box  : (x={x}, y={y}, w={w}, h={h})")
        print(f"  Solidity      : {solidity:.4f}")


def main():
    # ---------------------------------------------------------
    # Get image path from arguments if provided
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
    # Load Image
    # ---------------------------------------------------------
    print(f"Loading source image from: {image_path}")
    image_data = load_image(image_path)

    # ---------------------------------------------------------
    # ROI Selection
    # ---------------------------------------------------------
    selector = ROISelector(image_data)
    result = selector.exec()

    if result != selector.DialogCode.Accepted:
        print("ROI selection cancelled.")
        return

    roi = selector.roi
    if roi is None:
        print("No ROI selected.")
        return

    print(f"Selected ROI: x={roi.x}, y={roi.y}, w={roi.width}, h={roi.height}")

    # Crop the BGR image for display
    cropped_bgr = roi.crop(image_data.image)
    cropped_rgb = cv2.cvtColor(cropped_bgr, cv2.COLOR_BGR2RGB)

    # ---------------------------------------------------------
    # Hessian Ridge Detection
    # ---------------------------------------------------------
    print("Running Hessian Ridge Detection...")
    hessian_mask = detect_hessian_ridges(
        image_data,
        roi,
        threshold=250,
    )

    # ---------------------------------------------------------
    # HSV Evidence (to get the white mask)
    # ---------------------------------------------------------
    print("Running HSV Evidence Detection...")
    hsv_evidence = detect_hsv_evidence(
        image_data,
        roi,
    )
    white_mask = hsv_evidence.white_mask

    # ---------------------------------------------------------
    # Extract Contours from the Inner Black Region of White Mask
    # ---------------------------------------------------------
    # We invert the white mask to make the inner black region (printed target silhouette)
    # the foreground (white/255) for contour extraction.
    inner_black_mask = cv2.bitwise_not(white_mask)

    print("Extracting contours from Inner Black Mask (Inverted White Mask)...")
    # Use cv2.RETR_EXTERNAL to get the outer boundaries of the inner black shape
    contours, _ = cv2.findContours(
        inner_black_mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    # Print statistics for the largest contours
    print_contour_statistics(contours)

    # ---------------------------------------------------------
    # Draw Contours on ROI Image
    # ---------------------------------------------------------
    contour_visualization = cropped_bgr.copy()
    
    # Sort contours by area to highlight the largest one
    sorted_contours = sorted(contours, key=cv2.contourArea, reverse=True)
    
    if sorted_contours:
        # Highlight only the largest contour in blue with thickness=3
        # (255, 0, 0) in BGR is Blue
        cv2.drawContours(contour_visualization, [sorted_contours[0]], -1, (255, 0, 0), 3)
        
        # Label the largest contour with its area
        x, y, w, h = cv2.boundingRect(sorted_contours[0])
        label = f"Largest: {cv2.contourArea(sorted_contours[0]):.1f} px2"
        cv2.putText(
            contour_visualization,
            label,
            (x, max(20, y - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 0, 0),
            2,
            cv2.LINE_AA,
        )
    
    contour_visualization_rgb = cv2.cvtColor(contour_visualization, cv2.COLOR_BGR2RGB)

    # ---------------------------------------------------------
    # Display Results
    # ---------------------------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Top-Left: ROI Image
    axes[0, 0].imshow(cropped_rgb)
    axes[0, 0].set_title("Selected ROI")
    axes[0, 0].axis("off")

    # Top-Right: Hessian Ridge Mask
    axes[0, 1].imshow(hessian_mask, cmap="gray")
    axes[0, 1].set_title("Hessian Ridge Detection")
    axes[0, 1].axis("off")

    # Bottom-Left: Inner Black Mask (Inverted White Mask)
    axes[1, 0].imshow(inner_black_mask, cmap="gray")
    axes[1, 0].set_title("Inner Black Mask (Inverted White Mask)")
    axes[1, 0].axis("off")

    # Bottom-Right: Contours on ROI
    axes[1, 1].imshow(contour_visualization_rgb)
    axes[1, 1].set_title("Largest Inner Black Mask Contour (Blue)")
    axes[1, 1].axis("off")

    plt.tight_layout()
    print("Displaying results window. Close the matplotlib window to exit.")
    plt.show()


if __name__ == "__main__":
    main()
