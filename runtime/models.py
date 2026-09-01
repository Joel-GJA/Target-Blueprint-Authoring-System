from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np
import cv2
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


# =========================================================================
# M2 / M3 Runtime Layout Refinement Models & Configuration
# =========================================================================

@dataclass
class FeatureRegistrationConfig:
    """Centralized configuration parameters for M2 & M3 refinement stages."""
    # M2 — Search & ROI Constraints
    search_margin_px: float = 30.0            # Margin to dilate transformed regions
    max_prediction_error_px: float = 40.0     # Point-level Euclidean distance gate
    orb_nfeatures: int = 5000                 # Max ORB keypoints to detect in runtime ROI
    orb_scale_factor: float = 1.2             # ORB scale factor
    orb_nlevels: int = 8                      # Number of pyramid levels
    orb_fast_threshold: int = 20              # FAST corner detection threshold
    
    # M2 — Descriptor Matching
    max_descriptor_distance: int = 50         # Max Hamming distance threshold for ORB
    descriptor_ratio: float = 0.85            # Ratio test threshold when matching candidates
    
    # M3 — Spatial Diversity
    spatial_grid_cells: tuple[int, int] = (3, 3)    # Grid division for spatial diversity check
    min_distinct_regions: int = 2             # Minimum number of active landmark regions with matches
    min_spatial_coverage_ratio: float = 0.10  # Min convex hull / bounding box area ratio
    
    # M3 — RANSAC & Homography Estimation
    ransac_threshold_px: float = 4.0          # Inlier distance threshold for RANSAC
    min_matches: int = 8                      # Minimum candidate matches needed for RANSAC
    min_inliers: int = 6                      # Minimum inliers required to accept H_refined
    min_inlier_ratio: float = 0.45            # Minimum inliers / total matches ratio
    
    # M3 — Transformation Sanity
    max_residual_shift_px: float = 20.0       # Max allowed shift between H_coarse and H_refined on ROI corners
    max_scale_distortion: float = 0.15        # Max allowed scale change between H_coarse and H_refined


@dataclass
class RuntimeFeature:
    """A feature keypoint and descriptor detected in the runtime frame."""
    x: float
    y: float
    keypoint: cv2.KeyPoint
    descriptor: np.ndarray
    region_id: Optional[str] = None
    source: Optional[str] = None


@dataclass
class FeatureCorrespondence:
    """A verified correspondence between a reference blueprint feature and runtime feature."""
    ref_pt: tuple[float, float]          # (x, y) in reference image space
    runtime_pt: tuple[float, float]      # (x, y) in camera frame space
    pred_pt: tuple[float, float]         # (x, y) predicted runtime position from H_coarse
    ref_feature: VisualFeatureRef        # Source blueprint visual feature
    runtime_feature: RuntimeFeature      # Matched runtime feature
    descriptor_distance: float           # Hamming distance
    prediction_error_px: float           # Euclidean distance ||pred_pt - runtime_pt||
    region_id: Optional[str]             # Associated region ID
    source: str                          # Feature source (landmark_region, silhouette_boundary, zone_boundary)
    is_inlier: bool = False              # Set by M3 RANSAC step


@dataclass
class M2Diagnostics:
    """Diagnostic statistics for M2 (Local Feature Localization)."""
    total_detected_kps: int = 0
    roi_detected_kps: int = 0
    region_associated_kps: int = 0
    raw_descriptor_matches: int = 0
    geometrically_gated_matches: int = 0
    matches_per_region: dict[str, int] = field(default_factory=dict)
    matches_per_source: dict[str, int] = field(default_factory=dict)
    mean_descriptor_distance: float = 0.0
    mean_prediction_error_px: float = 0.0
    localization_time_ms: float = 0.0


@dataclass
class FeatureCorrespondenceSet:
    """Output container for M2 (Local Feature Localization)."""
    correspondences: list[FeatureCorrespondence]
    diagnostics: M2Diagnostics
    runtime_roi_poly: np.ndarray          # Polygon of runtime target ROI in camera frame (Shape N, 1, 2 or N, 2)
    runtime_feature_mask: np.ndarray      # Dilated search mask used for ORB detection


@dataclass
class M3Diagnostics:
    """Diagnostic statistics for M3 (Homography Refinement)."""
    spatial_coverage_ratio: float = 0.0
    active_regions_count: int = 0
    reprojection_error_mean: float = 0.0
    reprojection_error_median: float = 0.0
    reprojection_error_max: float = 0.0
    max_corner_shift_px: float = 0.0
    scale_deviation: float = 0.0
    refinement_time_ms: float = 0.0
    rejection_reason: Optional[str] = None


@dataclass
class RegistrationResult:
    """Final output container for M3 (Geometric Refinement)."""
    homography: np.ndarray                # H_refined if is_refined=True, else fallback H_coarse
    is_refined: bool                      # True if H_refined passed all sanity checks, False if fallback
    success: bool                         # True if either H_refined or valid H_coarse is usable
    confidence: float                     # Composite confidence metric [0.0, 1.0]
    inlier_count: int
    total_matches: int
    inlier_ratio: float
    reprojection_error: float
    correspondences: list[FeatureCorrespondence]  # All correspondences with is_inlier populated
    diagnostics: M3Diagnostics
    correspondence_set: Optional[FeatureCorrespondenceSet] = None


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
    registration_result: Optional[RegistrationResult] = None


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
