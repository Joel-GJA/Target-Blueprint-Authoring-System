import sys

import cv2
import matplotlib.pyplot as plt

from PySide6.QtWidgets import QApplication

from core.image_loader import load_image
from core.roi_selector import ROISelector
from core.hessian_detector import detect_hessian_ridges


IMAGE_PATH = "test_images/outdoor target.jpeg"


def main():

    # ---------------------------------------------------------
    # Create Qt application BEFORE creating any QWidget
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

    print()
    print("=" * 60)
    print("SELECTED ROI")
    print("=" * 60)

    print(f"x      : {roi.x}")
    print(f"y      : {roi.y}")
    print(f"width  : {roi.width}")
    print(f"height : {roi.height}")

    # ---------------------------------------------------------
    # Run Hessian Ridge Detection
    # ---------------------------------------------------------

    ridges = detect_hessian_ridges(
        image_data,
        roi,
    )

    # ---------------------------------------------------------
    # Crop original image
    # ---------------------------------------------------------

    cropped = roi.crop(
        image_data.image
    )

    cropped_rgb = cv2.cvtColor(
        cropped,
        cv2.COLOR_BGR2RGB,
    )

    # ---------------------------------------------------------
    # Display results
    # ---------------------------------------------------------

    plt.figure(
        figsize=(14, 8)
    )

    plt.subplot(1, 2, 1)

    plt.imshow(
        cropped_rgb
    )

    plt.title(
        "Selected ROI"
    )

    plt.axis("off")

    plt.subplot(1, 2, 2)

    plt.imshow(
        ridges,
        cmap="gray",
    )

    plt.title(
        "Hessian Ridge Response"
    )

    plt.axis("off")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()