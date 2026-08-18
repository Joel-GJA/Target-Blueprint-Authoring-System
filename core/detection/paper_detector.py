from __future__ import annotations

import cv2
import numpy as np

from core.models import (
    ImageData,
    AprilTagDetectionResult,
)


class PaperDetectionError(Exception):
    """Raised when the paper boundary cannot be detected."""


def detect_paper_boundary(
    image_data: ImageData,
    tag_result: AprilTagDetectionResult,
    debug: bool = False,
) -> np.ndarray:
    """
    Detect the outer paper boundary.

    Parameters
    ----------
    image_data:
        Loaded reference image.

    tag_result:
        AprilTag detections used to establish the approximate
        target region.

    debug:
        If True, intermediate information is printed.

    Returns
    -------
    np.ndarray
        Paper contour with shape (N, 1, 2), dtype int32.
    """

    image = image_data.image

    if image is None or image.size == 0:
        raise PaperDetectionError(
            "Input image is empty."
        )

    if tag_result.count < 2:
        raise PaperDetectionError(
            "At least two AprilTags are required to estimate "
            "the target search region."
        )

    # ---------------------------------------------------------
    # 1. Convert image to HSV
    # ---------------------------------------------------------

    hsv = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2HSV,
    )

    # ---------------------------------------------------------
    # 2. Create a white/light-paper mask
    # ---------------------------------------------------------

    # Paper is expected to have:
    #
    #   relatively high brightness
    #   relatively low saturation
    #
    # These values are intentionally conservative. They should
    # be tuned using the reference image rather than assumed
    # to be universally optimal.

    lower = np.array(
        [0, 0, 140],
        dtype=np.uint8,
    )

    upper = np.array(
        [180, 90, 255],
        dtype=np.uint8,
    )

    paper_mask = cv2.inRange(
        hsv,
        lower,
        upper,
    )

    # ---------------------------------------------------------
    # 3. Morphological cleanup
    # ---------------------------------------------------------

    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (7, 7),
    )

    paper_mask = cv2.morphologyEx(
        paper_mask,
        cv2.MORPH_CLOSE,
        kernel,
        iterations=2,
    )

    paper_mask = cv2.morphologyEx(
        paper_mask,
        cv2.MORPH_OPEN,
        kernel,
        iterations=1,
    )

    # ---------------------------------------------------------
    # 4. Find candidate contours
    # ---------------------------------------------------------

    contours, _ = cv2.findContours(
        paper_mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    if not contours:
        raise PaperDetectionError(
            "No candidate paper contours were found."
        )

    # ---------------------------------------------------------
    # 5. Calculate approximate target region from AprilTags
    # ---------------------------------------------------------

    centers = np.asarray(
        [
            detection.center
            for detection in tag_result.detections
        ],
        dtype=np.float64,
    )

    min_x = float(np.min(centers[:, 0]))
    max_x = float(np.max(centers[:, 0]))
    min_y = float(np.min(centers[:, 1]))
    max_y = float(np.max(centers[:, 1]))

    # Expand the tag bounding box substantially because the
    # target itself extends beyond the AprilTags.

    width = max_x - min_x
    height = max_y - min_y

    roi_min_x = min_x - 0.75 * width
    roi_max_x = max_x + 0.75 * width
    roi_min_y = min_y - 0.75 * height
    roi_max_y = max_y + 0.75 * height

    image_height, image_width = image.shape[:2]

    roi_min_x = max(0, int(roi_min_x))
    roi_max_x = min(image_width - 1, int(roi_max_x))
    roi_min_y = max(0, int(roi_min_y))
    roi_max_y = min(image_height - 1, int(roi_max_y))

    # ---------------------------------------------------------
    # 6. Score candidate contours
    # ---------------------------------------------------------

    image_center = np.array(
        [
            (roi_min_x + roi_max_x) / 2,
            (roi_min_y + roi_max_y) / 2,
        ],
        dtype=np.float64,
    )

    candidates = []

    for contour in contours:

        area = cv2.contourArea(contour)

        if area <= 0:
            continue

        contour_points = contour.reshape(-1, 2)

        contour_center = np.mean(
            contour_points,
            axis=0,
        )

        # Distance from expected target region center.
        distance = np.linalg.norm(
            contour_center - image_center
        )

        # Prefer large contours near the target.
        candidates.append(
            (
                area,
                distance,
                contour,
            )
        )

    if not candidates:
        raise PaperDetectionError(
            "No valid paper contour candidates remain."
        )

    # ---------------------------------------------------------
    # 7. Select candidate
    # ---------------------------------------------------------

    # Normalize area and distance to create a simple score.

    max_area = max(
        candidate[0]
        for candidate in candidates
    )

    max_distance = max(
        candidate[1]
        for candidate in candidates
    )

    if max_distance == 0:
        max_distance = 1.0

    scored_candidates = []

    for area, distance, contour in candidates:

        area_score = area / max_area
        proximity_score = 1.0 - (
            distance / max_distance
        )

        score = (
            0.70 * area_score
            + 0.30 * proximity_score
        )

        scored_candidates.append(
            (
                score,
                contour,
            )
        )

    scored_candidates.sort(
        key=lambda item: item[0],
        reverse=True,
    )

    best_score, best_contour = scored_candidates[0]

    if debug:
        print()
        print("=" * 60)
        print("PAPER BOUNDARY DETECTION")
        print("=" * 60)
        print(f"Candidate contours : {len(contours)}")
        print(f"Selected score     : {best_score:.4f}")
        print(
            f"Selected area      : "
            f"{cv2.contourArea(best_contour):.1f} px²"
        )

    return best_contour
