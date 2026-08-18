from __future__ import annotations

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

@dataclass(frozen=True)
class ROI:
    """
    Region of interest expressed in original-image coordinates.

    All coordinates refer to the original, full-resolution image.
    """

    x: int
    y: int
    width: int
    height: int

    @property
    def x2(self) -> int:
        return self.x + self.width

    @property
    def y2(self) -> int:
        return self.y + self.height

    @property
    def area(self) -> int:
        return self.width * self.height

    def crop(self, image: np.ndarray) -> np.ndarray:
        """Return the ROI cropped from an image."""

        return image[
            self.y:self.y2,
            self.x:self.x2,
        ]

    def to_tuple(self) -> tuple[int, int, int, int]:
        """Return (x, y, width, height)."""

        return (
            self.x,
            self.y,
            self.width,
            self.height,
        )

    def save(self, filepath: str | Path) -> None:
        """Save the ROI properties to a JSON file."""
        import json
        data = {
            "x": self.x,
            "y": self.y,
            "width": self.width,
            "height": self.height
        }
        with open(filepath, "w") as f:
            json.dump(data, f, indent=4)

    @classmethod
    def load(cls, filepath: str | Path) -> ROI | None:
        """Load the ROI properties from a JSON file."""
        import json
        from pathlib import Path
        p = Path(filepath)
        if not p.exists():
            return None
        try:
            with open(p, "r") as f:
                data = json.load(f)
            return cls(
                x=data["x"],
                y=data["y"],
                width=data["width"],
                height=data["height"]
            )
        except Exception as e:
            print(f"Error loading ROI: {e}")
            return None

    def contains(
        self,
        x: int,
        y: int,
    ) -> bool:
        """Check whether an image coordinate lies inside the ROI."""

        return (
            self.x <= x <= self.x2
            and
            self.y <= y <= self.y2
        )


@dataclass
class Zone:
    """A scoring zone defined as a polygon with score and name metadata."""

    zone_id: str
    polygon: np.ndarray  # OpenCV contour shape (N, 1, 2), ROI-local coordinates
    score: float
    name: str = ""


@dataclass
class FeatureRegion:
    """A persistent region in the target reference frame designated for visual feature extraction."""
    id: str
    polygon: np.ndarray  # OpenCV contour shape (N, 1, 2), ROI-local coordinates
    region_type: str = "stable"  # "stable", "zone_corner", "optional"
    priority: int = 1
    min_features: int = 5
    max_features: int = 50
    metadata: dict = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class FeatureRegionTemplate:
    """A reusable template defining normalized visual feature regions for a target type."""
    template_id: str
    target_type: str
    version: int
    regions: list[FeatureRegion]  # Stored in normalized coordinates relative to target reference bounds
    metadata: dict = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class ZoneFeatureAnchor:
    """An opportunistic feature anchor centered on a scoring zone vertex."""
    zone_id: str
    corner_index: int
    x: float  # ROI-local x coordinate
    y: float  # ROI-local y coordinate
    optional: bool = True


@dataclass
class VisualFeature:
    """A single visual feature (e.g. ORB keypoint and its metadata)."""
    x: float
    y: float
    size: float
    angle: float
    response: float
    octave: int
    class_id: int
    descriptor: np.ndarray  # 1D array of uint8 for ORB (typically 32 bytes)
    source: str = "landmark_region"  # "landmark_region", "silhouette_boundary", "zone_boundary"
    region_id: str | None = None  # Association with a FeatureRegion
    anchor_info: dict | None = None  # e.g., {"zone_id": str, "corner_index": int} if opportunistic


@dataclass
class VisualFeatureSet:
    """Collection of generated visual features and metadata."""
    features: tuple[VisualFeature, ...]
    quality_metrics: dict  # e.g., {"distribution_entropy": float, "mean_response": float, "repeatability": float}
