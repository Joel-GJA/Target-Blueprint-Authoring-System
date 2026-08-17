import sys

from PySide6.QtWidgets import QApplication

from core.image_loader import load_image
from core.roi import ROISelector


IMAGE_PATH = "test_images/outdoor target.jpeg"


def main():

    app = QApplication(sys.argv)

    image_data = load_image(
        IMAGE_PATH
    )

    selector = ROISelector(
        image_data
    )

    result = selector.exec()

    if result == selector.DialogCode.Accepted:

        roi = selector.roi

        print()
        print("=" * 60)
        print("ROI SELECTION")
        print("=" * 60)

        print(
            f"x      : {roi.x}"
        )

        print(
            f"y      : {roi.y}"
        )

        print(
            f"width  : {roi.width}"
        )

        print(
            f"height : {roi.height}"
        )

        print(
            f"x2     : {roi.x2}"
        )

        print(
            f"y2     : {roi.y2}"
        )

        print(
            f"area   : {roi.area}"
        )

        # -----------------------------------------------------
        # Verify actual crop
        # -----------------------------------------------------

        cropped = roi.crop(
            image_data.image
        )

        print(
            f"crop shape: {cropped.shape}"
        )

    else:

        print(
            "ROI selection cancelled."
        )


if __name__ == "__main__":
    main()