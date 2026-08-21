from __future__ import annotations

import json
from pathlib import Path
import cv2
import numpy as np
from dataclasses import dataclass


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


@dataclass
class AprilTagReference:
    """Reference geometry and identity of a calibration AprilTag on the target."""
    tag_id: int
    corners: np.ndarray  # Shape (4, 2) corners in reference image coordinates
    center: tuple[float, float]  # Center (x, y) in reference image coordinates


@dataclass
class Blueprint:
    """
    Offline Blueprint Package.
    Represents the complete, self-contained target registration and detection reference contract.
    """
    blueprint_id: str
    target_type: str
    name: str
    format_version: str
    created_at: str
    
    # ROI bounding box in reference image coordinates: (x, y, w, h)
    roi_bounds: tuple[int, int, int, int]
    
    # Physical scale parameters
    pixels_per_mm: float
    mm_per_pixel: float
    tag_size_mm: float
    target_width_mm: float
    target_height_mm: float
    
    # Calibration AprilTags detected in the reference image
    april_tags: list[AprilTagReference]
    
    # Silhouette contour in ROI-local coordinates (shape (N, 1, 2))
    silhouette: np.ndarray | None
    
    # Scoring zones in ROI-local coordinates
    zones: list[Zone]
    
    # Feature regions in target-normalized coordinates
    feature_regions: list[FeatureRegion]
    
    # Final Visual feature set (keypoints and ORB descriptors)
    features: VisualFeatureSet | None = None

    def save_package(self, package_dir: str | Path, reference_image: np.ndarray) -> None:
        """
        Saves the blueprint package (JSON metadata + copy of the reference image)
        to the designated package directory.
        """
        p_dir = Path(package_dir)
        p_dir.mkdir(parents=True, exist_ok=True)
        
        # Save reference image
        img_path = p_dir / "reference_image.jpg"
        cv2.imwrite(str(img_path), reference_image)
        
        # Serialize AprilTags
        tags_data = []
        for tag in self.april_tags:
            tags_data.append({
                "tag_id": tag.tag_id,
                "corners": tag.corners.tolist(),
                "center": list(tag.center)
            })
            
        # Serialize silhouette
        sil_data = self.silhouette.tolist() if self.silhouette is not None else None
        
        # Serialize zones
        zones_data = []
        for z in self.zones:
            zones_data.append({
                "zone_id": z.zone_id,
                "polygon": z.polygon.tolist(),
                "score": z.score,
                "name": z.name
            })
            
        # Serialize feature regions
        regions_data = []
        for r in self.feature_regions:
            regions_data.append({
                "id": r.id,
                "polygon": r.polygon.tolist(),  # Stored in normalized coordinates
                "region_type": r.region_type,
                "priority": r.priority,
                "min_features": r.min_features,
                "max_features": r.max_features,
                "metadata": r.metadata
            })
            
        # Serialize features
        features_data = []
        if self.features:
            for f in self.features.features:
                features_data.append({
                    "x": f.x,
                    "y": f.y,
                    "size": f.size,
                    "angle": f.angle,
                    "response": f.response,
                    "octave": f.octave,
                    "class_id": f.class_id,
                    "descriptor": f.descriptor.tolist(),  # List of integers
                    "source": f.source,
                    "region_id": f.region_id,
                    "anchor_info": f.anchor_info
                })
                
        metadata = {
            "blueprint_id": self.blueprint_id,
            "target_type": self.target_type,
            "name": self.name,
            "format_version": self.format_version,
            "created_at": self.created_at,
            "roi_bounds": list(self.roi_bounds),
            "scale": {
                "pixels_per_mm": self.pixels_per_mm,
                "mm_per_pixel": self.mm_per_pixel,
                "tag_size_mm": self.tag_size_mm,
                "target_width_mm": self.target_width_mm,
                "target_height_mm": self.target_height_mm
            },
            "april_tags": tags_data,
            "silhouette": sil_data,
            "zones": zones_data,
            "feature_regions": regions_data,
            "features": {
                "features": features_data,
                "quality_metrics": self.features.quality_metrics if self.features else {}
            }
        }
        
        json_path = p_dir / "blueprint.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=4)

    @classmethod
    def load_package(cls, package_dir: str | Path) -> tuple[Blueprint, np.ndarray]:
        """
        Loads and reconstructs a blueprint package and its reference image from the package directory.
        """
        p_dir = Path(package_dir)
        json_path = p_dir / "blueprint.json"
        img_path = p_dir / "reference_image.jpg"
        
        if not json_path.exists():
            raise FileNotFoundError(f"Blueprint JSON not found: {json_path}")
        if not img_path.exists():
            raise FileNotFoundError(f"Reference image not found: {img_path}")
            
        ref_image = cv2.imread(str(img_path))
        
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            
        # AprilTags
        tags = []
        for tag_data in data["april_tags"]:
            tags.append(AprilTagReference(
                tag_id=tag_data["tag_id"],
                corners=np.array(tag_data["corners"], dtype=np.float32),
                center=tuple(tag_data["center"])
            ))
            
        # Silhouette
        sil = np.array(data["silhouette"], dtype=np.int32) if data["silhouette"] is not None else None
        
        # Zones
        zones = []
        for z_data in data["zones"]:
            zones.append(Zone(
                zone_id=z_data["zone_id"],
                polygon=np.array(z_data["polygon"], dtype=np.int32),
                score=z_data["score"],
                name=z_data.get("name", "")
            ))
            
        # Feature regions
        regions = []
        for r_data in data["feature_regions"]:
            regions.append(FeatureRegion(
                id=r_data["id"],
                polygon=np.array(r_data["polygon"], dtype=np.float32),
                region_type=r_data["region_type"],
                priority=r_data["priority"],
                min_features=r_data["min_features"],
                max_features=r_data["max_features"],
                metadata=r_data.get("metadata", {})
            ))
            
        # Features
        feats_list = []
        f_data = data["features"]
        for f in f_data.get("features", []):
            feats_list.append(VisualFeature(
                x=f["x"],
                y=f["y"],
                size=f["size"],
                angle=f["angle"],
                response=f["response"],
                octave=f["octave"],
                class_id=f["class_id"],
                descriptor=np.array(f["descriptor"], dtype=np.uint8),
                source=f.get("source", "landmark_region"),
                region_id=f.get("region_id"),
                anchor_info=f.get("anchor_info")
            ))
        features_set = VisualFeatureSet(
            features=tuple(feats_list),
            quality_metrics=f_data.get("quality_metrics", {})
        )
        
        blueprint = cls(
            blueprint_id=data["blueprint_id"],
            target_type=data["target_type"],
            name=data["name"],
            format_version=data["format_version"],
            created_at=data["created_at"],
            roi_bounds=tuple(data["roi_bounds"]),
            pixels_per_mm=data["scale"]["pixels_per_mm"],
            mm_per_pixel=data["scale"]["mm_per_pixel"],
            tag_size_mm=data["scale"]["tag_size_mm"],
            target_width_mm=data["scale"].get("target_width_mm", 490.0),
            target_height_mm=data["scale"].get("target_height_mm", 1220.0),
            april_tags=tags,
            silhouette=sil,
            zones=zones,
            feature_regions=regions,
            features=features_set
        )
        
        return blueprint, ref_image
