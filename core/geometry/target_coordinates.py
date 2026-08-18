import cv2
import numpy as np
from core.models import ROI

def get_target_bounds(roi: ROI, silhouette_contour: np.ndarray | None) -> tuple[float, float, float, float]:
    """
    Gets target reference bounds (xmin, ymin, width, height) relative to ROI coordinates.
    If silhouette_contour is provided, bounds are the bounding box of the silhouette.
    Otherwise, falls back to ROI bounds (0.0, 0.0, roi.width, roi.height).
    """
    if silhouette_contour is not None and len(silhouette_contour) > 0:
        # Silhouette contour is already in ROI-local coordinates
        rx, ry, rw, rh = cv2.boundingRect(silhouette_contour)
        return float(rx), float(ry), float(rw), float(rh)
    return 0.0, 0.0, float(roi.width), float(roi.height)

def pixel_to_normalized(pts: np.ndarray, target_bounds: tuple[float, float, float, float]) -> np.ndarray:
    """
    Converts ROI-local pixel coordinates to normalized target coordinates.
    pts: numpy array of shape (..., 2) or (N, 1, 2)
    """
    bx, by, bw, bh = target_bounds
    pts_norm = pts.copy().astype(np.float32)
    pts_norm[..., 0] = (pts[..., 0] - bx) / bw
    pts_norm[..., 1] = (pts[..., 1] - by) / bh
    return pts_norm

def normalized_to_pixel(pts_norm: np.ndarray, target_bounds: tuple[float, float, float, float]) -> np.ndarray:
    """
    Converts normalized target coordinates back to ROI-local pixel coordinates.
    pts_norm: numpy array of shape (..., 2) or (N, 1, 2)
    """
    bx, by, bw, bh = target_bounds
    pts_px = pts_norm.copy().astype(np.float32)
    pts_px[..., 0] = bx + pts_norm[..., 0] * bw
    pts_px[..., 1] = by + pts_norm[..., 1] * bh
    return pts_px
