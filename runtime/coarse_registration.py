from __future__ import annotations

import cv2
import numpy as np
from runtime.models import RuntimeBlueprint, RuntimeGeometry

class CoarseRegistration:
    """
    Computes coarse homography between reference blueprint space and current camera frame space.
    """
    @staticmethod
    def compute_coarse_homography(
        blueprint: RuntimeBlueprint,
        observed_tags: list[dict]  # List of dicts, e.g. [{"tag_id": 0, "corners": array(4,2)}]
    ) -> tuple[np.ndarray | None, float]:
        """
        Computes the registration homography directly by matching observed tag corners to 
        the reference blueprint tag corners. This bypasses the template millimeter layout 
        to avoid scale mismatches if the stand size or tag placement offsets differ from 
        the target paper sheet boundaries.
        """
        ref_pts = []
        obs_pts = []
        
        for obs in observed_tags:
            tag_id = obs["tag_id"]
            # Find the matching reference tag in the blueprint
            ref_tag = next((t for t in blueprint.april_tags if t.tag_id == tag_id), None)
            if ref_tag is not None:
                obs_corners = obs["corners"]
                ref_corners = ref_tag.corners
                
                # Align corners of the observed tag with the reference tag (to handle rotation shifts)
                best_shift = 0
                min_err = float('inf')
                for shift in range(4):
                    shifted_obs = np.roll(obs_corners, shift, axis=0)
                    err = np.sum(np.linalg.norm(shifted_obs - ref_corners, axis=1))
                    if err < min_err:
                        min_err = err
                        best_shift = shift
                        
                aligned_obs = np.roll(obs_corners, best_shift, axis=0)
                for idx in range(4):
                    ref_pts.append(ref_corners[idx])
                    obs_pts.append(aligned_obs[idx])
                    
        if len(ref_pts) < 12:  # We need at least 3 matching tags
            return None, 0.0
            
        ref_pts = np.array(ref_pts, dtype=np.float32)
        obs_pts = np.array(obs_pts, dtype=np.float32)
        
        H_coarse, inliers = cv2.findHomography(ref_pts, obs_pts, cv2.RANSAC, 5.0)
        if H_coarse is None:
            return None, 0.0
            
        # Calculate confidence
        inliers_count = int(np.sum(inliers)) if inliers is not None else len(ref_pts)
        inlier_ratio = inliers_count / len(ref_pts) if len(ref_pts) > 0 else 0.0
        
        tags_matched = len(ref_pts) // 4
        total_tags = len(blueprint.april_tags)
        tag_ratio = tags_matched / max(1, total_tags)
        confidence = tag_ratio * inlier_ratio
        
        return H_coarse, confidence

    @staticmethod
    def evaluate_homography_sanity(H: np.ndarray) -> tuple[bool, str]:
        """
        Runs mathematical checks on the homography matrix to verify physical sanity.
        Returns (is_sane, reason_string).
        """
        if H is None:
            return False, "Homography matrix is None"
            
        # Determinant of the top-left 2x2 matrix must be positive and non-zero
        det = np.linalg.det(H[:2, :2])
        if det <= 0:
            return False, f"Invalid determinant ({det:.4f}): contains reflection or fold"
        if det < 1e-4:
            return False, f"Near-singular determinant ({det:.6e})"
            
        # Singular values check for extreme aspect ratio distortion (shear/skew)
        u, s, vh = np.linalg.svd(H[:2, :2])
        skew = s[0] / max(1e-6, s[1])
        if skew > 5.0:
            return False, f"Excessive scale skew ratio ({skew:.2f} > 5.0)"
            
        return True, "Homography is sane"

    @staticmethod
    def get_runtime_search_region(
        blueprint: RuntimeBlueprint,
        H_coarse: np.ndarray,
        frame_shape: tuple[int, int]
    ) -> np.ndarray:
        """
        Warp reference ROI bounds to current frame space to define the target search region boundary.
        """
        rx, ry, rw, rh = blueprint.roi_bounds
        # Define reference ROI bounding box corners: (top-left, top-right, bottom-right, bottom-left)
        corners = np.array([
            [rx, ry],
            [rx + rw, ry],
            [rx + rw, ry + rh],
            [rx, ry + rh]
        ], dtype=np.float32).reshape(-1, 1, 2)
        
        # Transform corners to current frame coordinates
        transformed = cv2.perspectiveTransform(corners, H_coarse)
        
        # Clamp to frame boundary
        h, w = frame_shape[:2]
        transformed[..., 0] = np.clip(transformed[..., 0], 0, w - 1)
        transformed[..., 1] = np.clip(transformed[..., 1], 0, h - 1)
        
        return transformed.astype(np.int32)
