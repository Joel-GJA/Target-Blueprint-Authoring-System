from __future__ import annotations

import cv2
import numpy as np
from runtime.models import (
    RuntimeBlueprint,
    RuntimeGeometry,
    RuntimeState,
    RuntimeStatus,
)

class Calibration:
    """
    Maintains and updates runtime calibration parameters.
    Maps geometries and scale from the blueprint to the active camera coordinates.
    """
    @staticmethod
    def compute_current_scale(blueprint: RuntimeBlueprint, H: np.ndarray) -> float:
        """
        Estimates the scale factor of the current frame relative to the reference blueprint
        by warping reference test points and measuring scale deviation.
        """
        rx, ry, rw, rh = blueprint.roi_bounds
        cx, cy = rx + rw / 2.0, ry + rh / 2.0
        
        # Two reference points separated by 100 pixels horizontally
        pts = np.array([
            [cx, cy],
            [cx + 100.0, cy]
        ], dtype=np.float32).reshape(-1, 1, 2)
        
        transformed = cv2.perspectiveTransform(pts, H)
        p1 = transformed[0][0]
        p2 = transformed[1][0]
        
        warped_dist = np.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)
        scale_factor = warped_dist / 100.0
        
        return blueprint.pixels_per_mm * scale_factor

    @staticmethod
    def generate_runtime_geometry(
        blueprint: RuntimeBlueprint,
        H: np.ndarray,
        frame_shape: tuple[int, int]
    ) -> RuntimeGeometry:
        """
        Transforms silhouette, zone polygons, and feature regions from the reference blueprint
        into active camera frame pixel coordinates.
        """
        rx, ry, rw, rh = blueprint.roi_bounds
        ref_offset = np.array([[[rx, ry]]], dtype=np.float32)
        
        # 1. Transform Target Silhouette
        if blueprint.silhouette is not None and len(blueprint.silhouette) > 0:
            # Silhouette in blueprint is ROI-local
            sil_ref = (blueprint.silhouette.astype(np.float32) + ref_offset).reshape(-1, 1, 2)
            sil_camera = cv2.perspectiveTransform(sil_ref, H).reshape(-1, 1, 2).astype(np.int32)
        else:
            sil_camera = np.zeros((0, 1, 2), dtype=np.int32)
            
        # 2. Transform Scoring Zones
        warped_zones = []
        for zone in blueprint.zones:
            # Zone polygon is ROI-local
            zone_ref = (zone.polygon.astype(np.float32) + ref_offset).reshape(-1, 1, 2)
            zone_camera = cv2.perspectiveTransform(zone_ref, H).reshape(-1, 1, 2).astype(np.int32)
            warped_zones.append((zone.zone_id, zone_camera, zone.score))
            
        # 3. Transform Feature Regions
        from core.geometry.target_coordinates import get_target_bounds, normalized_to_pixel
        # Calculate target bounds relative to ROI (in pixels)
        target_bounds = get_target_bounds(
            roi=cv2.selectROI if False else None,  # Just placeholder helper
            silhouette_contour=blueprint.silhouette
        )
        # Note: target_bounds is (xmin, ymin, w, h) ROI-local.
        # But wait! We can bypass if we don't have roi object:
        # target_bounds is derived relative to ROI.
        # Let's compute it locally to avoid importing core ROI selector dependencies:
        if blueprint.silhouette is not None and len(blueprint.silhouette) > 0:
            tbx, tby, tbw, tbh = cv2.boundingRect(blueprint.silhouette)
        else:
            tbx, tby, tbw, tbh = 0.0, 0.0, float(rw), float(rh)
            
        tbounds = (float(tbx), float(tby), float(tbw), float(tbh))
        
        warped_regions = []
        for region in blueprint.feature_regions:
            # Convert normalized to ROI-local pixels
            pts_norm = region.polygon.copy()
            pts_px = pts_norm.copy()
            pts_px[..., 0] = tbounds[0] + pts_norm[..., 0] * tbounds[2]
            pts_px[..., 1] = tbounds[1] + pts_norm[..., 1] * tbounds[3]
            
            # Convert to absolute reference coordinates
            region_ref = (pts_px.astype(np.float32) + ref_offset).reshape(-1, 1, 2)
            region_camera = cv2.perspectiveTransform(region_ref, H).reshape(-1, 1, 2).astype(np.int32)
            warped_regions.append((region.id, region_camera))
            
        # 4. Transform Target ROI (to search region polygon)
        corners_roi = np.array([
            [rx, ry],
            [rx + rw, ry],
            [rx + rw, ry + rh],
            [rx, ry + rh]
        ], dtype=np.float32).reshape(-1, 1, 2)
        search_camera = cv2.perspectiveTransform(corners_roi, H).reshape(-1, 1, 2).astype(np.int32)
        
        return RuntimeGeometry(
            silhouette=sil_camera,
            zones=warped_zones,
            feature_regions=warped_regions,
            search_region=search_camera
        )
