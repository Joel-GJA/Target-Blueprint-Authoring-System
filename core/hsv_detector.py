from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from core.models import ImageData, ROI


class HSVDetectionError(Exception):
    """Raised when HSV analysis cannot be performed."""


@dataclass(frozen=True)
class HSVEvidence:
    """
    HSV-based evidence extracted from an ROI.

    All masks are in ROI-local coordinates.
    """

    hsv_image: np.ndarray

    saturation_mask: np.ndarray
    value_mask: np.ndarray

    orange_mask: np.ndarray
    dark_mask: np.ndarray
    white_mask: np.ndarray

    combined_target_mask: np.ndarray
    inner_target_candidate: np.ndarray


def detect_hsv_evidence(
    image_data: ImageData,
    roi: ROI,
) -> HSVEvidence:
    """
    Generate HSV-based target evidence inside an ROI.

    This function does NOT attempt to determine the final
    target silhouette.

    It produces independent masks that can later be combined
    with Hessian/ridge evidence.
    """

    cropped = roi.crop(
        image_data.image
    )

    if cropped.size == 0:
        raise HSVDetectionError(
            "ROI produced an empty image."
        )

    # ---------------------------------------------------------
    # BGR -> HSV
    # ---------------------------------------------------------

    hsv = cv2.cvtColor(
        cropped,
        cv2.COLOR_BGR2HSV,
    )

    # ---------------------------------------------------------
    # Channels
    # ---------------------------------------------------------

    hue = hsv[:, :, 0]
    saturation = hsv[:, :, 1]
    value = hsv[:, :, 2]

    # ---------------------------------------------------------
    # 1. Saturation evidence
    # ---------------------------------------------------------
    #
    # High saturation identifies strongly coloured regions.
    #
    # This should capture much of the orange target while
    # rejecting gray/white regions.
    #
    # OpenCV HSV:
    # H: 0-179
    # S: 0-255
    # V: 0-255
    # ---------------------------------------------------------

    saturation_mask = cv2.inRange(
        saturation,
        60,
        255,
    )

    # ---------------------------------------------------------
    # 2. Value evidence
    # ---------------------------------------------------------
    #
    # Separate bright and dark structures.
    #
    # The target contains both, so this is useful evidence
    # rather than a direct target mask.
    # ---------------------------------------------------------

    value_mask = cv2.inRange(
        value,
        40,
        230,
    )

    # ---------------------------------------------------------
    # 3. Orange target evidence
    # ---------------------------------------------------------
    #
    # OpenCV hue for orange is approximately 5-25.
    #
    # This is intentionally broad for the first experiment.
    # ---------------------------------------------------------

    orange_mask = cv2.inRange(
        hsv,
        np.array(
            [5, 70, 60],
            dtype=np.uint8,
        ),
        np.array(
            [25, 255, 255],
            dtype=np.uint8,
        ),
    )

    # ---------------------------------------------------------
    # 4. Dark target evidence
    # ---------------------------------------------------------
    #
    # The black region of the target is important.
    #
    # Low V captures dark printed regions.
    # ---------------------------------------------------------

    dark_mask = cv2.inRange(
        hsv,
        np.array(
            [0, 0, 0],
            dtype=np.uint8,
        ),
        np.array(
            [179, 255, 90],
            dtype=np.uint8,
        ),
    )

    # ---------------------------------------------------------
    # 5. White-region evidence
    # ---------------------------------------------------------
    #
    # White material generally has:
    #
    #   - low saturation
    #   - high value
    #
    # This is deliberately broad because the physical white
    # border can be affected by lighting, shadows and wrinkles.
    #
    # IMPORTANT:
    # This is WHITE-REGION evidence, not yet "white target
    # border" detection.
    # ---------------------------------------------------------

    white_mask = cv2.inRange(
        hsv,
        np.array(
            [0, 0, 160],
            dtype=np.uint8,
        ),
        np.array(
            [179, 70, 255],
            dtype=np.uint8,
        ),
    )

    #inverting the white mask to get the inner target candidate
    inner_target_candidate = cv2.bitwise_not(
        white_mask
    )
    
    # ---------------------------------------------------------
    # 6. Combine orange + dark
    # ---------------------------------------------------------
    #
    # We are NOT claiming this is the silhouette.
    #
    # We're simply asking:
    #
    # "Where does the image have visual characteristics
    #  associated with the printed target?"
    # ---------------------------------------------------------

    combined_target_mask = cv2.bitwise_or(
        orange_mask,
        dark_mask,
    )


    return HSVEvidence(
        hsv_image=hsv,
        saturation_mask=saturation_mask,
        value_mask=value_mask,
        orange_mask=orange_mask,
        dark_mask=dark_mask,
        white_mask=white_mask,
        combined_target_mask=combined_target_mask,
        inner_target_candidate=inner_target_candidate,
    )