import cv2
import numpy as np
from pupil_apriltags import Detector

from core.models import (
    ImageData,
    AprilTagDetection,
    AprilTagDetectionResult,
)


class AprilTagDetectionError(Exception):
    """Raised when AprilTag detection cannot be performed."""


class AprilTagDetector:
    """
    Detect AprilTags in validated source images.

    This class performs detection only. It does not perform
    homography estimation, pose estimation, scale calibration,
    or perspective rectification.
    """

    def __init__(
        self,
        families: str = "tag36h11",
        nthreads: int = 1,
        quad_decimate: float = 1.0,
        quad_sigma: float = 0.0,
        refine_edges: bool = True,
        decode_sharpening: float = 0.25,
        debug: bool = False,
    ) -> None:

        self._detector = Detector(
            families=families,
            nthreads=nthreads,
            quad_decimate=quad_decimate,
            quad_sigma=quad_sigma,
            refine_edges=int(refine_edges),
            decode_sharpening=decode_sharpening,
            debug=int(debug),
        )

    def detect(
        self,
        image_data: ImageData,
    ) -> AprilTagDetectionResult:
        """
        Detect AprilTags in an ImageData object.

        Parameters
        ----------
        image_data:
            Validated source image.

        Returns
        -------
        AprilTagDetectionResult
            Structured collection of detected AprilTags.
        """

        image = image_data.image

        if image is None or image.size == 0:
            raise AprilTagDetectionError(
                "Input ImageData contains no valid image."
            )

        # pupil-apriltags expects a grayscale image.
        gray = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2GRAY,
        )

        raw_detections = self._detector.detect(
            gray,
            estimate_tag_pose=False,
        )

        detections = []

        for detection in raw_detections:

            tag = AprilTagDetection(
                tag_id=int(detection.tag_id),
                tag_family=self._decode_family(
                    detection.tag_family
                ),
                center=np.asarray(
                    detection.center,
                    dtype=np.float64,
                ),
                corners=np.asarray(
                    detection.corners,
                    dtype=np.float64,
                ),
                decision_margin=float(
                    detection.decision_margin
                ),
                hamming=int(detection.hamming),
            )

            detections.append(tag)

        return AprilTagDetectionResult(
            detections=tuple(detections)
        )

    @staticmethod
    def _decode_family(tag_family) -> str:
        """Convert pupil-apriltags family representation to string."""

        if isinstance(tag_family, bytes):
            return tag_family.decode("utf-8")

        return str(tag_family)
