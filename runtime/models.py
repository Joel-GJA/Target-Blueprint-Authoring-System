from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from typing import Optional, Any

@dataclass
class AprilTagRef:
    """AprilTag reference coordinate data in the reference image space."""
    tag_id: int
    corners: np.ndarray  # Shape (4, 2), coordinates relative to reference image
    center: tuple[float, float]

@dataclass
class ZoneRef:
    """Scoring zone specification from the blueprint."""
    zone_id: str
    polygon: np.ndarray  # Shape (N, 1, 2) ROI-local coordinates
    score: float
    name: str

@dataclass
class FeatureRegionRef:
    """Feature search region in normalized coordinates."""
    id: str
    polygon: np.ndarray  # Shape (N, 1, 2) target-normalized
    region_type: str
    priority: int

@dataclass
class VisualFeatureRef:
    """Visual feature (keypoint + ORB descriptor) stored in the blueprint."""
    x: float
    y: float
    size: float
    angle: float
    response: float
    octave: int
    class_id: int
    descriptor: np.ndarray  # 1D array of uint8 (length 32 for ORB)
    source: str
    region_id: Optional[str]
    anchor_info: Optional[dict]

@dataclass
class RuntimeBlueprint:
    """Reconstituted Blueprint representation used by the runtime engine."""
    blueprint_id: str
    target_type: str
    name: str
    format_version: str
    created_at: str
    roi_bounds: tuple[int, int, int, int]
    pixels_per_mm: float
    mm_per_pixel: float
    tag_size_mm: float
    target_width_mm: float
    target_height_mm: float
    april_tags: list[AprilTagRef]
    silhouette: Optional[np.ndarray]  # ROI-local coordinates
    zones: list[ZoneRef]
    feature_regions: list[FeatureRegionRef]
    features: list[VisualFeatureRef]

@dataclass
class RuntimeGeometry:
    """Blueprint geometry warped into the current camera frame coordinates."""
    silhouette: np.ndarray  # Shape (N, 1, 2), camera frame pixel coordinates
    zones: list[tuple[str, np.ndarray, float]]  # List of (zone_id, polygon, score)
    feature_regions: list[tuple[str, np.ndarray]]  # List of (region_id, polygon)
    search_region: np.ndarray  # Polygon representing search bounds in camera frame

class RuntimeStatus:
    NO_TARGET = "NO_TARGET"
    ACQUIRING = "ACQUIRING"
    COARSE_REGISTERED = "COARSE_REGISTERED"
    CALIBRATING = "CALIBRATING"
    CALIBRATED = "CALIBRATED"
    READY = "READY"
    CORRECTION_FAILED = "CORRECTION_FAILED"
    SHOT_PROCESSING = "SHOT_PROCESSING"
    ERROR = "ERROR"

@dataclass
class RuntimeState:
    """State of the runtime engine tracking registration calibration."""
    status: str
    homography: Optional[np.ndarray]  # Maps reference image to current frame
    pixels_per_mm: Optional[float]
    geometry: Optional[RuntimeGeometry]
    registration_confidence: float
    last_frame_timestamp: Optional[float]
    frame_id: int

@dataclass
class Bullet:
    """Observed bullet hit and its assigned scoring attributes."""
    center_px: tuple[float, float]  # (x, y) in camera frame
    diameter_px: float
    diameter_mm: float
    zone_id: Optional[str]
    score: float
    confidence: float

@dataclass
class ShotResult:
    """Complete measurement output for a single processed frame."""
    shot_id: int
    bullets: list[Bullet]
    registration_confidence: float
    pixels_per_mm: float
    total_score: float
