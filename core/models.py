from dataclasses import dataclass
from pathlib import Path
import numpy as np


@dataclass(frozen=True)
class ImageMetadata:
    """Metadata describing the imported source image."""

    path: Path
    filename: str
    width: int
    height: int
    channels: int
    dtype: str
    file_size_bytes: int


@dataclass(frozen=True)
class ImageData:
    """Validated source image and its metadata."""

    image: np.ndarray
    metadata: ImageMetadata


@dataclass(frozen=True)
class AprilTagDetection:
    """A single detected AprilTag."""

    tag_id: int
    tag_family: str

    center: np.ndarray
    corners: np.ndarray

    decision_margin: float
    hamming: int


@dataclass(frozen=True)
class AprilTagDetectionResult:
    """Result of running AprilTag detection on an image."""

    detections: tuple[AprilTagDetection, ...]

    @property
    def count(self) -> int:
        return len(self.detections)

    @property
    def tag_ids(self) -> tuple[int, ...]:
        return tuple(tag.tag_id for tag in self.detections)

@dataclass(frozen=True)
class TagScaleMeasurement:
    """Scale measurement derived from one AprilTag."""

    tag_id: int

    mean_side_pixels: float
    pixels_per_mm: float

    quality: float


@dataclass(frozen=True)
class ScaleCalibrationResult:
    """Physical scale estimated from AprilTag measurements."""

    pixels_per_mm: float
    millimeters_per_pixel: float

    reference_tag_size_mm: float

    measurements: tuple[TagScaleMeasurement, ...]

    contributing_tag_ids: tuple[int, ...]

@dataclass(frozen=True)
class PaperDetectionResult:
    """Result of automatic paper boundary detection."""

    contour: np.ndarray
    area_pixels: float
    perimeter_pixels: float
    bounding_box: tuple[int, int, int, int]