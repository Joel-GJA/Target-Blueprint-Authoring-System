import numpy as np

from core.models import (
    AprilTagDetectionResult,
    ScaleCalibrationResult,
    TagScaleMeasurement,
)


class ScaleCalibrationError(Exception):
    """Raised when physical scale cannot be determined."""


def calibrate_scale(
    tag_result: AprilTagDetectionResult,
    tag_size_mm: float,
    minimum_quality: float = 0.90,
) -> ScaleCalibrationResult:
    """
    Estimate pixels-per-millimeter from detected AprilTags.

    Parameters
    ----------
    tag_result:
        AprilTag detections from the reference image.

    tag_size_mm:
        Known physical width of the AprilTag in millimeters.

    minimum_quality:
        Minimum geometric quality required for a tag
        to contribute to the final scale estimate.

    Returns
    -------
    ScaleCalibrationResult
    """

    if tag_size_mm <= 0:
        raise ScaleCalibrationError(
            "AprilTag physical size must be greater than zero."
        )

    if tag_result.count == 0:
        raise ScaleCalibrationError(
            "No AprilTags available for scale calibration."
        )

    measurements: list[TagScaleMeasurement] = []

    for detection in tag_result.detections:

        corners = np.asarray(
            detection.corners,
            dtype=np.float64,
        )

        if corners.shape != (4, 2):
            continue

        # -----------------------------------------------------
        # Calculate the four side lengths
        # -----------------------------------------------------

        sides = np.linalg.norm(
            np.roll(corners, -1, axis=0) - corners,
            axis=1,
        )

        mean_side = float(np.mean(sides))

        if mean_side <= 0:
            continue

        # -----------------------------------------------------
        # Measure how square the detected tag is
        # -----------------------------------------------------

        side_ratio = float(
            np.min(sides) / np.max(sides)
        )

        # A perfectly front-facing square gives:
        #
        # min(side) / max(side) = 1.0
        #
        # Perspective or distortion moves this toward zero.

        quality = side_ratio

        pixels_per_mm = mean_side / tag_size_mm

        measurements.append(
            TagScaleMeasurement(
                tag_id=detection.tag_id,
                mean_side_pixels=mean_side,
                pixels_per_mm=pixels_per_mm,
                quality=quality,
            )
        )

    # ---------------------------------------------------------
    # Keep only geometrically reliable tags
    # ---------------------------------------------------------

    valid_measurements = [
        measurement
        for measurement in measurements
        if measurement.quality >= minimum_quality
    ]

    if not valid_measurements:
        raise ScaleCalibrationError(
            "No AprilTags passed the geometric quality threshold."
        )

    # ---------------------------------------------------------
    # Weighted scale estimate
    # ---------------------------------------------------------

    weights = np.asarray(
        [m.quality for m in valid_measurements],
        dtype=np.float64,
    )

    scales = np.asarray(
        [m.pixels_per_mm for m in valid_measurements],
        dtype=np.float64,
    )

    pixels_per_mm = float(
        np.average(
            scales,
            weights=weights,
        )
    )

    if pixels_per_mm <= 0:
        raise ScaleCalibrationError(
            "Calculated pixels-per-millimeter is invalid."
        )

    millimeters_per_pixel = 1.0 / pixels_per_mm

    contributing_ids = tuple(
        measurement.tag_id
        for measurement in valid_measurements
    )

    return ScaleCalibrationResult(
        pixels_per_mm=pixels_per_mm,
        millimeters_per_pixel=millimeters_per_pixel,
        reference_tag_size_mm=tag_size_mm,
        measurements=tuple(measurements),
        contributing_tag_ids=contributing_ids,
    )
