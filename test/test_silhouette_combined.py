from __future__ import annotations

import sys
from pathlib import Path

# Add project root to sys.path
sys.path.append(str(Path(__file__).parent.parent.resolve()))

import cv2
import matplotlib.pyplot as plt
import numpy as np

from PySide6.QtWidgets import QApplication

from core.image_loader import load_image
from core.roi_selector import ROISelector
from core.silhouette_detector import detect_silhouette_candidates

# Default image path (can be overridden via command-line arguments)
IMAGE_PATH = "test_images/outdoor target.jpeg"


def draw_candidates(image: np.ndarray, candidates) -> np.ndarray:
    """
    Draw ranked candidate contours on the ROI image.
    """
    output = image.copy()
    for index, candidate in enumerate(candidates):
        contour = candidate.contour

        # Generate colors for different ranks
        if index == 0:
            color = (0, 255, 0)      # Rank 1: Green
        elif index == 1:
            color = (0, 255, 255)    # Rank 2: Yellow
        elif index == 2:
            color = (0, 165, 255)    # Rank 3: Orange
        else:
            color = (255, 192, 203)  # Others: Pink

        cv2.drawContours(output, [contour], -1, color, 2)
    return output


def print_results(detection_result) -> None:
    """
    Print information about the white mask contour and the candidate contours.
    """
    print()
    print("=" * 80)
    print("SILHOUETTE PIPELINE PIPELINE RESULTS (NEW API)")
    print("=" * 80)
    
    # 1. Print White Mask Contour statistics
    white_contour = detection_result.white_mask_contour
    print("\n[White Mask Contour (Inverted White Mask)]")
    if white_contour is not None:
        area = cv2.contourArea(white_contour)
        perimeter = cv2.arcLength(white_contour, closed=True)
        x, y, w, h = cv2.boundingRect(white_contour)
        print(f"  Area (px²)    : {area:.1f}")
        print(f"  Perimeter (px): {perimeter:.1f}")
        print(f"  Bounding Box  : (x={x}, y={y}, w={w}, h={h})")
    else:
        print("  No white mask contour found.")

    # 2. Print Candidate Contours statistics
    print("\n[Ranked Silhouette Candidates (Hessian + HSV Evidence)]")
    for idx, candidate in enumerate(detection_result.candidates):
        x, y, w, h = candidate.bounding_box
        print(
            f"  #{idx + 1} - Score: {candidate.score:.4f} | "
            f"Area Ratio: {candidate.area_ratio:.3f} | "
            f"Hessian: {candidate.hessian_support:.3f} | "
            f"Inside Target: {candidate.inside_target_support:.3f} | "
            f"Outside White: {candidate.outside_white_support:.3f} | "
            f"BB: ({x}, {y}, {w}, {h})"
        )


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
    # Call the New API
    # ---------------------------------------------------------
    print("Invoking detect_silhouette_candidates API...")
    detection_result = detect_silhouette_candidates(
        image_data,
        roi,
        hessian_threshold=250,
        max_candidates=10,
    )

    # Print all results to console
    print_results(detection_result)

    # ---------------------------------------------------------
    # Generate Visualizations
    # ---------------------------------------------------------
    # Image 1: Ranked Candidates (Only draw Candidate #1)
    candidates_vis = draw_candidates(cropped_bgr, detection_result.candidates[:1])
    candidates_vis_rgb = cv2.cvtColor(candidates_vis, cv2.COLOR_BGR2RGB)

    # Image 2: White Mask Contour in Blue
    white_contour_vis = cropped_bgr.copy()
    white_contour = detection_result.white_mask_contour
    if white_contour is not None:
        # Draw in Blue (255, 0, 0)
        cv2.drawContours(white_contour_vis, [white_contour], -1, (255, 0, 0), 3)
    white_contour_vis_rgb = cv2.cvtColor(white_contour_vis, cv2.COLOR_BGR2RGB)

    # Image 3: Combined Visualization (Candidate #1 + White Mask Contour)
    combined_vis = candidates_vis.copy()
    if white_contour is not None:
        # Overlay the white mask contour in thick Blue to contrast with candidate #1
        cv2.drawContours(combined_vis, [white_contour], -1, (255, 0, 0), 3)
    combined_vis_rgb = cv2.cvtColor(combined_vis, cv2.COLOR_BGR2RGB)

    # ---------------------------------------------------------
    # Display Results
    # ---------------------------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Top-Left: Selected ROI
    axes[0, 0].imshow(cropped_rgb)
    axes[0, 0].set_title("Selected ROI")
    axes[0, 0].axis("off")

    # Top-Right: Ranked Candidate #1 from Hessian & HSV Processing
    axes[0, 1].imshow(candidates_vis_rgb)
    axes[0, 1].set_title("Ranked Silhouette Candidate #1\n(Hessian + HSV evidence)")
    axes[0, 1].axis("off")

    # Bottom-Left: White Mask Contour in Blue
    axes[1, 0].imshow(white_contour_vis_rgb)
    axes[1, 0].set_title("White Mask Contour (Inverted White Mask, Blue)")
    axes[1, 0].axis("off")

    # Bottom-Right: Combined Comparison
    axes[1, 1].imshow(combined_vis_rgb)
    axes[1, 1].set_title("Combined Contours\n(Candidate #1 + White Contour in Blue)")
    axes[1, 1].axis("off")

    plt.tight_layout()
    print("Displaying results window. Close the matplotlib window to exit.")
    plt.show()


if __name__ == "__main__":
    main()
