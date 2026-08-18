from __future__ import annotations

import cv2
import numpy as np

from core.models import ImageData, ROI


class HessianDetectionError(Exception):
    """Raised when Hessian ridge detection fails."""


def detect_hessian_ridges(
    image_data: ImageData,
    roi: ROI,
    threshold: int = 250,
    blur_kernel: tuple[int, int] = (5, 5),
    blur_sigma: float = 2.0,
    detector_ksize: int = 3,
) -> np.ndarray:
    """
    Run the OpenCV Hessian Ridge Detection Filter inside an ROI.

    Returns
    -------
    np.ndarray
        Binary ridge image in ROI-local coordinates.
    """

    cropped = roi.crop(
        image_data.image
    )

    if cropped.size == 0:
        raise HessianDetectionError(
            "ROI produced an empty image."
        )

    gray = cv2.cvtColor(
        cropped,
        cv2.COLOR_BGR2GRAY,
    )

    gray = cv2.GaussianBlur(
        gray,
        blur_kernel,
        blur_sigma,
    )

    detector = (
        cv2.ximgproc.RidgeDetectionFilter_create(
            ddepth=cv2.CV_32F,
            dx=1,
            dy=1,
            ksize=detector_ksize,
            out_dtype=cv2.CV_8U,
            scale=1,
            delta=0,
        )
    )

    ridges = detector.getRidgeFilteredImage(
        gray
    )

    # Adaptive thresholding: instead of a hard threshold of 250 (which causes lines to disappear under blur),
    # use a relative threshold of 20% of the maximum detected ridge intensity.
    max_val = np.max(ridges)
    if max_val > 0:
        thresh_val = max(15, int(0.20 * max_val))
    else:
        thresh_val = threshold

    _, binary = cv2.threshold(
        ridges,
        thresh_val,
        255,
        cv2.THRESH_BINARY,
    )

    return binary
