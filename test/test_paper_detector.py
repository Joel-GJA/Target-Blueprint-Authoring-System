import cv2
import matplotlib.pyplot as plt
import numpy as np

from core.calibration.image_loader import load_image
from core.detection.apriltag_detector import AprilTagDetector
from core.detection.paper_detector import detect_paper_boundary


IMAGE_PATH = "test_images/outdoor target.jpeg"


def create_paper_mask(image: np.ndarray) -> np.ndarray:
    """
    Reproduce the current paper candidate mask so that
    the intermediate result can be visualized.
    """

    hsv = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2HSV,
    )

    lower = np.array(
        [0, 0, 140],
        dtype=np.uint8,
    )

    upper = np.array(
        [180, 90, 255],
        dtype=np.uint8,
    )

    mask = cv2.inRange(
        hsv,
        lower,
        upper,
    )

    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (7, 7),
    )

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_CLOSE,
        kernel,
        iterations=2,
    )

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_OPEN,
        kernel,
        iterations=1,
    )

    return mask


def draw_contour(
    image: np.ndarray,
    contour: np.ndarray,
) -> np.ndarray:

    result = image.copy()

    cv2.drawContours(
        result,
        [contour],
        contourIdx=-1,
        color=(0, 0, 255),
        thickness=5,
    )

    return result


def draw_apriltags(
    image: np.ndarray,
    tag_result,
) -> np.ndarray:

    result = image.copy()

    for detection in tag_result.detections:

        corners = np.round(
            detection.corners
        ).astype(np.int32)

        cv2.polylines(
            result,
            [corners],
            isClosed=True,
            color=(0, 255, 0),
            thickness=3,
        )

        center = tuple(
            np.round(
                detection.center
            ).astype(int)
        )

        cv2.circle(
            result,
            center,
            6,
            (255, 0, 0),
            -1,
        )

        cv2.putText(
            result,
            f"ID {detection.tag_id}",
            center,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 0, 0),
            2,
            cv2.LINE_AA,
        )

    return result


def main():

    print()
    print("=" * 60)
    print("PAPER DETECTOR TEST")
    print("=" * 60)

    # ---------------------------------------------------------
    # 1. Load image
    # ---------------------------------------------------------

    image_data = load_image(
        IMAGE_PATH
    )

    print(
        f"Image       : "
        f"{image_data.metadata.filename}"
    )

    print(
        f"Resolution  : "
        f"{image_data.metadata.width} x "
        f"{image_data.metadata.height}"
    )

    # ---------------------------------------------------------
    # 2. Detect AprilTags
    # ---------------------------------------------------------

    detector = AprilTagDetector(
        families="tag36h11",
        nthreads=4,
    )

    tag_result = detector.detect(
        image_data
    )

    print(
        f"AprilTags   : "
        f"{tag_result.count}"
    )

    # ---------------------------------------------------------
    # 3. Generate candidate mask
    # ---------------------------------------------------------

    mask = create_paper_mask(
        image_data.image
    )

    # ---------------------------------------------------------
    # 4. Run paper detector
    # ---------------------------------------------------------

    try:

        contour = detect_paper_boundary(
            image_data,
            tag_result,
            debug=True,
        )

    except Exception as exc:

        print()
        print("PAPER DETECTION FAILED")
        print(exc)
        return

    # ---------------------------------------------------------
    # 5. Calculate contour information
    # ---------------------------------------------------------

    area = cv2.contourArea(
        contour
    )

    perimeter = cv2.arcLength(
        contour,
        closed=True,
    )

    x, y, width, height = cv2.boundingRect(
        contour
    )

    print()
    print("-" * 60)
    print("SELECTED PAPER CONTOUR")
    print("-" * 60)

    print(
        f"Area        : {area:.2f} px²"
    )

    print(
        f"Perimeter   : {perimeter:.2f} px"
    )

    print(
        f"Bounding box: "
        f"x={x}, y={y}, "
        f"w={width}, h={height}"
    )

    print(
        f"Contour pts : "
        f"{len(contour)}"
    )

    # ---------------------------------------------------------
    # 6. Draw contour over original image
    # ---------------------------------------------------------

    overlay = draw_contour(
        image_data.image,
        contour,
    )

    overlay = draw_apriltags(
        overlay,
        tag_result,
    )

    # Convert BGR → RGB for Matplotlib
    original_rgb = cv2.cvtColor(
        image_data.image,
        cv2.COLOR_BGR2RGB,
    )

    overlay_rgb = cv2.cvtColor(
        overlay,
        cv2.COLOR_BGR2RGB,
    )

    # ---------------------------------------------------------
    # 7. Display results
    # ---------------------------------------------------------

    plt.figure(
        figsize=(18, 10)
    )

    plt.subplot(1, 3, 1)

    plt.imshow(
        original_rgb
    )

    plt.title(
        "Original Image"
    )

    plt.axis("off")

    plt.subplot(1, 3, 2)

    plt.imshow(
        mask,
        cmap="gray",
    )

    plt.title(
        "Paper Candidate Mask"
    )

    plt.axis("off")

    plt.subplot(1, 3, 3)

    plt.imshow(
        overlay_rgb
    )

    plt.title(
        "Detected Paper Contour"
    )

    plt.axis("off")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()