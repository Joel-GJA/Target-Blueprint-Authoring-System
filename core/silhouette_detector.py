from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from core.hessian_detector import detect_hessian_ridges
from core.hsv_detector import detect_hsv_evidence
from core.models import ImageData, ROI


@dataclass(frozen=True)
class SilhouetteCandidate:
    """
    A candidate silhouette contour and the evidence supporting it.

    All contour coordinates are ROI-local.
    """

    contour: np.ndarray

    score: float

    area_ratio: float
    height_ratio: float
    width_ratio: float

    solidity: float

    hessian_support: float
    inside_target_support: float
    outside_white_support: float

    boundary_touch_count: int
    boundary_touch_ratio: float

    perimeter: float

    @property
    def bounding_box(self) -> tuple[int, int, int, int]:
        x, y, w, h = cv2.boundingRect(self.contour)

        return x, y, w, h


@dataclass(frozen=True)
class SilhouetteDetectionResult:
    """
    Result of silhouette candidate generation and ranking.
    """

    candidates: list[SilhouetteCandidate]

    hessian_mask: np.ndarray
    inside_target_mask: np.ndarray
    white_mask: np.ndarray
    white_mask_contour: np.ndarray | None = None


class SilhouetteDetectionError(Exception):
    """Raised when silhouette detection cannot be performed."""


# ----------------------------------------------------------------------
# Candidate generation
# ----------------------------------------------------------------------


def _generate_candidate_contours(
    hessian_mask: np.ndarray,
    min_area: int,
) -> list[np.ndarray]:
    """
    Extract closed candidate contours from the Hessian response.

    The Hessian image contains both useful outer boundaries and
    internal target structures. We intentionally keep many candidates
    here and let the scoring stage decide which ones are useful.
    """

    contours, _ = cv2.findContours(
        hessian_mask,
        cv2.RETR_LIST,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    candidates: list[np.ndarray] = []

    for contour in contours:

        area = cv2.contourArea(
            contour
        )

        if area < min_area:
            continue

        candidates.append(
            contour
        )

    return candidates


# ----------------------------------------------------------------------
# Geometry
# ----------------------------------------------------------------------


def _contour_geometry(
    contour: np.ndarray,
    roi: ROI,
) -> tuple[
    float,
    float,
    float,
    float,
    float,
]:
    """
    Calculate basic geometric properties.
    """

    area = cv2.contourArea(
        contour
    )

    perimeter = cv2.arcLength(
        contour,
        True,
    )

    x, y, w, h = cv2.boundingRect(
        contour
    )

    roi_area = (
        roi.width
        * roi.height
    )

    area_ratio = (
        area / roi_area
        if roi_area > 0
        else 0.0
    )

    width_ratio = (
        w / roi.width
        if roi.width > 0
        else 0.0
    )

    height_ratio = (
        h / roi.height
        if roi.height > 0
        else 0.0
    )

    hull = cv2.convexHull(
        contour
    )

    hull_area = cv2.contourArea(
        hull
    )

    if hull_area > 0:
        solidity = (
            area / hull_area
        )
    else:
        solidity = 0.0

    return (
        area_ratio,
        width_ratio,
        height_ratio,
        solidity,
        perimeter,
    )


# ----------------------------------------------------------------------
# Mask scoring
# ----------------------------------------------------------------------


def _draw_contour_mask(
    contour: np.ndarray,
    shape: tuple[int, int],
    thickness: int = -1,
) -> np.ndarray:
    """
    Rasterize a contour into a binary mask.
    """

    mask = np.zeros(
        shape,
        dtype=np.uint8,
    )

    cv2.drawContours(
        mask,
        [contour],
        -1,
        255,
        thickness,
    )

    return mask


def _calculate_hessian_support(
    contour: np.ndarray,
    hessian_mask: np.ndarray,
) -> float:
    """
    Measure how strongly the Hessian response supports the contour.
    """

    boundary_mask = np.zeros_like(
        hessian_mask
    )

    cv2.drawContours(
        boundary_mask,
        [contour],
        -1,
        255,
        thickness=5,
    )

    boundary_pixels = (
        boundary_mask > 0
    )

    if not np.any(boundary_pixels):
        return 0.0

    support = np.mean(
        hessian_mask[boundary_pixels] > 0
    )

    return float(support)


def _calculate_inside_target_support(
    contour: np.ndarray,
    target_mask: np.ndarray,
) -> float:
    """
    Measure target evidence inside the candidate contour.

    A small erosion removes the immediate contour boundary so that
    boundary evidence does not dominate the interior measurement.
    """

    contour_mask = _draw_contour_mask(
        contour,
        target_mask.shape,
    )

    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (9, 9),
    )

    interior_mask = cv2.erode(
        contour_mask,
        kernel,
    )

    interior_pixels = (
        interior_mask > 0
    )

    if not np.any(interior_pixels):
        return 0.0

    support = np.mean(
        target_mask[interior_pixels] > 0
    )

    return float(support)


def _calculate_outside_white_support(
    contour: np.ndarray,
    white_mask: np.ndarray,
) -> float:
    """
    Measure white-region evidence immediately outside the contour.

    A narrow outer ring is used rather than evaluating the entire
    outside region.
    """

    contour_mask = _draw_contour_mask(
        contour,
        white_mask.shape,
    )

    dilated = cv2.dilate(
        contour_mask,
        cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (15, 15),
        ),
    )

    outer_ring = (
        (dilated > 0)
        & (contour_mask == 0)
    )

    if not np.any(outer_ring):
        return 0.0

    support = np.mean(
        white_mask[outer_ring] > 0
    )

    return float(support)

def _calculate_boundary_contact(
    contour: np.ndarray,
    roi: ROI,
    margin: int = 5,
) -> tuple[int, float]:
    """
    Determine whether the candidate is pressed against the ROI
    boundaries.

    This is a soft warning rather than an automatic rejection.
    """

    x, y, w, h = cv2.boundingRect(contour)

    roi_width = roi.width
    roi_height = roi.height

    touches_left = x <= margin
    touches_top = y <= margin
    touches_right = (
        x + w >= roi_width - margin
    )
    touches_bottom = (
        y + h >= roi_height - margin
    )

    touches = [
        touches_left,
        touches_top,
        touches_right,
        touches_bottom,
    ]

    count = sum(touches)

    return count, count / 4.0


# ----------------------------------------------------------------------
# Geometry scoring
# ----------------------------------------------------------------------


def _score_area(
    area_ratio: float,
) -> float:
    """
    Score contour size.

    We want a substantial contour, but not one that simply occupies
    the entire ROI.
    """

    if area_ratio < 0.05:
        return 0.0

    if area_ratio < 0.20:
        return area_ratio / 0.20

    if area_ratio <= 0.80:
        return 1.0

    # Penalize contours that fill nearly the entire ROI.
    return max(
        0.0,
        1.0 - (
            area_ratio - 0.80
        ) / 0.20,
    )


def _score_height(
    height_ratio: float,
) -> float:
    """
    The target is strongly vertically oriented.

    This is deliberately a weak geometric prior.
    """

    if height_ratio < 0.40:
        return 0.0

    if height_ratio < 0.70:
        return (
            height_ratio - 0.40
        ) / 0.30

    return 1.0


def _score_width(
    width_ratio: float,
) -> float:
    """
    The target should occupy a meaningful portion of the ROI width.
    """

    if width_ratio < 0.20:
        return 0.0

    if width_ratio < 0.50:
        return (
            width_ratio - 0.20
        ) / 0.30

    return 1.0

def _score_boundary_contact(
    boundary_touch_count: int,
) -> float:
    """
    Return a geometric confidence score based on ROI boundary contact.

    Fewer contacts are preferable.

    This is deliberately a soft penalty because a correctly selected
    ROI can legitimately touch the target boundary.
    """

    penalties = {
        0: 1.00,
        1: 0.90,
        2: 0.70,
        3: 0.40,
        4: 0.10,
    }

    return penalties.get(
        boundary_touch_count,
        0.10,
    )

# ----------------------------------------------------------------------
# Candidate scoring
# ----------------------------------------------------------------------


def _score_candidate(
    contour: np.ndarray,
    roi: ROI,
    hessian_mask: np.ndarray,
    target_mask: np.ndarray,
    white_mask: np.ndarray,
) -> SilhouetteCandidate:

    (
        area_ratio,
        width_ratio,
        height_ratio,
        solidity,
        perimeter,
    ) = _contour_geometry(
        contour,
        roi,
    )

    hessian_support = (
        _calculate_hessian_support(
            contour,
            hessian_mask,
        )
    )

    inside_target_support = (
        _calculate_inside_target_support(
            contour,
            target_mask,
        )
    )

    outside_white_support = (
        _calculate_outside_white_support(
            contour,
            white_mask,
        )
    )

    # --------------------------------------------------------------
    # Geometry
    # --------------------------------------------------------------

    area_score = _score_area(
        area_ratio
    )

    height_score = _score_height(
        height_ratio
    )

    width_score = _score_width(
        width_ratio
    )

    boundary_touch_count, boundary_touch_ratio = (
        _calculate_boundary_contact(
            contour,
            roi,
        )
    )

    boundary_score = _score_boundary_contact(
        boundary_touch_count
    )

    # --------------------------------------------------------------
    # Combined score
    # --------------------------------------------------------------
    #
    # The weights are intentionally exposed and easy to tune.
    #
    # HSV interior evidence and white exterior evidence are the
    # primary discriminators.
    #
    # Hessian confirms that the candidate corresponds to an actual
    # structural boundary.
    # --------------------------------------------------------------

    score = (
        0.25 * hessian_support
        +
        0.30 * inside_target_support
        +
        0.25 * outside_white_support
        +
        0.10 * area_score
        +
        0.05 * height_score
        +
        0.05 * width_score
        +
        0.07 * solidity
        +
        0.03 * boundary_score
    )

    return SilhouetteCandidate(
        contour=contour,
        score=float(score),
        area_ratio=area_ratio,
        height_ratio=height_ratio,
        width_ratio=width_ratio,
        solidity=solidity,
        hessian_support=hessian_support,
        inside_target_support=inside_target_support,
        outside_white_support=outside_white_support,
        perimeter=perimeter,
        boundary_touch_count=boundary_touch_count,
        boundary_touch_ratio=boundary_touch_ratio,
    )


def _extract_white_mask_contour(
    inner_target_candidate: np.ndarray,
) -> np.ndarray | None:
    """
    Extract the largest contour representing the boundary of the inner target
    from the inverted white mask (inner target candidate).
    """
    contours, _ = cv2.findContours(
        inner_target_candidate,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    if not contours:
        return None

    # Return the largest contour by area
    return max(contours, key=cv2.contourArea)


# ----------------------------------------------------------------------
# Public API
# ----------------------------------------------------------------------


def detect_silhouette_candidates(
    image_data: ImageData,
    roi: ROI,
    *,
    hessian_threshold: int = 250,
    min_area_ratio: float = 0.005,
    max_candidates: int = 10,
) -> SilhouetteDetectionResult:
    """
    Generate and rank silhouette candidates.

    This is intentionally an analysis stage. The highest-scoring
    candidate should NOT yet be treated as the final silhouette
    without visual inspection.
    """

    if roi.width <= 0 or roi.height <= 0:
        raise SilhouetteDetectionError(
            "ROI has invalid dimensions."
        )

    # --------------------------------------------------------------
    # Hessian evidence
    # --------------------------------------------------------------

    hessian_mask = detect_hessian_ridges(
        image_data,
        roi,
        threshold=hessian_threshold,
    )

    # --------------------------------------------------------------
    # HSV evidence
    # --------------------------------------------------------------

    hsv_evidence = detect_hsv_evidence(
        image_data,
        roi,
    )

    # --------------------------------------------------------------
    # Interior target evidence
    #
    # Use orange + dark rather than counting the inverted white mask
    # as a second independent target vote.
    # --------------------------------------------------------------

    target_mask = cv2.bitwise_or(
        hsv_evidence.orange_mask,
        hsv_evidence.dark_mask,
    )

    # --------------------------------------------------------------
    # Candidate generation
    # --------------------------------------------------------------

    roi_area = (
        roi.width
        * roi.height
    )

    min_area = int(
        roi_area
        * min_area_ratio
    )

    contours = _generate_candidate_contours(
        hessian_mask,
        min_area,
    )

    # --------------------------------------------------------------
    # Score candidates
    # --------------------------------------------------------------

    scored: list[
        SilhouetteCandidate
    ] = []

    for contour in contours:

        candidate = _score_candidate(
            contour,
            roi,
            hessian_mask,
            target_mask,
            hsv_evidence.white_mask,
        )

        scored.append(
            candidate
        )

    scored.sort(
        key=lambda candidate: candidate.score,
        reverse=True,
    )

    scored = scored[
        :max_candidates
    ]

    white_mask_contour = _extract_white_mask_contour(
        hsv_evidence.inner_target_candidate
    )

    return SilhouetteDetectionResult(
        candidates=scored,
        hessian_mask=hessian_mask,
        inside_target_mask=target_mask,
        white_mask=hsv_evidence.white_mask,
        white_mask_contour=white_mask_contour,
    )