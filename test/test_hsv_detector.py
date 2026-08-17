import sys

import cv2
import matplotlib.pyplot as plt

from PySide6.QtWidgets import QApplication

from core.image_loader import load_image
from core.roi_selector import ROISelector
from core.hsv_detector import detect_hsv_evidence


IMAGE_PATH = "test_images/outdoor target.jpeg"


def main():

    # ---------------------------------------------------------
    # Qt application
    # ---------------------------------------------------------

    app = QApplication.instance()

    if app is None:
        app = QApplication(sys.argv)

    # ---------------------------------------------------------
    # Load image
    # ---------------------------------------------------------

    image_data = load_image(
        IMAGE_PATH
    )

    # ---------------------------------------------------------
    # Select ROI
    # ---------------------------------------------------------

    selector = ROISelector(
        image_data
    )

    result = selector.exec()

    if result != selector.DialogCode.Accepted:
        print("ROI selection cancelled.")
        return

    roi = selector.roi

    if roi is None:
        print("No ROI selected.")
        return

    # ---------------------------------------------------------
    # HSV analysis
    # ---------------------------------------------------------

    evidence = detect_hsv_evidence(
        image_data,
        roi,
    )

    # ---------------------------------------------------------
    # Original ROI
    # ---------------------------------------------------------

    cropped = roi.crop(
        image_data.image
    )

    cropped_rgb = cv2.cvtColor(
        cropped,
        cv2.COLOR_BGR2RGB,
    )

    # ---------------------------------------------------------
    # Display
    # ---------------------------------------------------------

    fig, axes = plt.subplots(
        2,
        4,
        figsize=(18, 10),
    )

    axes[0, 0].imshow(
        cropped_rgb
    )

    axes[0, 0].set_title(
        "Selected ROI"
    )

    axes[0, 1].imshow(
        evidence.white_mask,
        cmap="gray",
    )
    
    axes[0, 1].set_title(
        "White Mask"
    )

    axes[0, 2].imshow(
        evidence.inner_target_candidate,
        cmap="gray",
    )

    axes[0, 2].set_title(
        "Inner Target Candidate"
    )

    axes[0, 3].imshow(
        evidence.combined_target_mask,
        cmap="gray",
    )

    axes[0, 3].set_title(
        "Combined Target Evidence"
    )

    axes[1, 0].imshow(
        evidence.saturation_mask,
        cmap="gray",
    )

    axes[1, 0].set_title(
        "Saturation Mask"
    )

    axes[1, 1].imshow(
        evidence.value_mask,
        cmap="gray",
    )

    axes[1, 1].set_title(
        "Value Mask"
    )

    axes[1, 2].imshow(
        evidence.orange_mask,
        cmap="gray",
    )

    axes[1, 2].set_title(
        "Orange Mask"
    )

    axes[1, 3].imshow(
        evidence.dark_mask,
        cmap="gray",
    )

    axes[1, 3].set_title(
        "Dark Mask"
    )

    for ax in axes.flat:
        ax.axis("off")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()