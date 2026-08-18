from __future__ import annotations

import cv2
import numpy as np
from runtime.models import RuntimeBlueprint, RuntimeGeometry

class CoarseRegistration:
    """
    Computes coarse homography between reference blueprint space and current camera frame space.
    Calculates exact physical tag-to-millimeter transformations (matching PILSS coordinates)
    to handle relative scaling, rotation, and lens distortion.
    """
    @staticmethod
    def compute_coarse_homography(
        blueprint: RuntimeBlueprint,
        observed_tags: list[dict]  # List of dicts, e.g. [{"tag_id": 0, "corners": array(4,2)}]
    ) -> tuple[np.ndarray | None, float]:
        """
        Calculates physical pixels-to-millimeter homographies for both the reference and current frame,
        and cascades them: H_coarse = H_mm_to_pixel * H_ref_to_mm.
        """
        T = blueprint.tag_size_mm
        M = 20.0  # Default margin in mm
        W = 210.0 # A4 width in mm
        H = 297.0 # A4 height in mm
        
        # Standard physical corners in mm for each position (TL, TR, BR, BL)
        corners_mm_by_pos = {
            0: np.array([[M, M], [M + T, M], [M + T, M + T], [M, M + T]], dtype=np.float32),              # TL
            1: np.array([[W - M - T, M], [W - M, M], [W - M, M + T], [W - M - T, M + T]], dtype=np.float32),  # TR
            2: np.array([[W - M - T, H - M - T], [W - M, H - M - T], [W - M, H - M], [W - M - T, H - M]], dtype=np.float32),# BR
            3: np.array([[M, H - M - T], [M + T, H - M - T], [M + T, H - M], [M, H - M]], dtype=np.float32)     # BL
        }
        
        # Determine tag-to-corner mapping by sorting blueprint tags geometrically
        ref_centers = []
        ref_tags = []
        for tag in blueprint.april_tags:
            ref_centers.append(np.mean(tag.corners, axis=0))
            ref_tags.append(tag)
            
        if len(ref_tags) < 3:
            return None, 0.0
            
        ref_centers = np.array(ref_centers, dtype=np.float32)
        s = ref_centers.sum(axis=1)
        diff = np.diff(ref_centers, axis=1).flatten()
        
        tl_idx = np.argmin(s)
        br_idx = np.argmax(s)
        tr_idx = np.argmin(diff)
        bl_idx = np.argmax(diff)
        
        # Map tag_id to physical template corners
        template_corners_by_id = {
            ref_tags[tl_idx].tag_id: corners_mm_by_pos[0],
            ref_tags[tr_idx].tag_id: corners_mm_by_pos[1],
            ref_tags[br_idx].tag_id: corners_mm_by_pos[2],
            ref_tags[bl_idx].tag_id: corners_mm_by_pos[3]
        }

        # 1. Compute H_ref_to_mm (Reference Image Pixels -> Physical mm)
        ref_src_pts = []
        ref_dst_pts = []
        for ref_tag in blueprint.april_tags:
            tag_id = ref_tag.tag_id
            if tag_id in template_corners_by_id:
                for idx in range(4):
                    ref_src_pts.append(ref_tag.corners[idx])
                    ref_dst_pts.append(template_corners_by_id[tag_id][idx])
                    
        if len(ref_src_pts) < 12:
            return None, 0.0
            
        ref_src_pts = np.array(ref_src_pts, dtype=np.float32)
        ref_dst_pts = np.array(ref_dst_pts, dtype=np.float32)
        
        H_ref_to_mm, _ = cv2.findHomography(ref_src_pts, ref_dst_pts, cv2.RANSAC, 5.0)
        if H_ref_to_mm is None:
            return None, 0.0

        # 2. Compute H_pixel_to_mm (Current Frame Pixels -> Physical mm)
        src_centers = []
        dst_centers = []
        matched_pairs = []
        
        for obs in observed_tags:
            tag_id = obs["tag_id"]
            if tag_id not in template_corners_by_id:
                continue
            corners_dst = obs["corners"]
            center_dst = np.mean(corners_dst, axis=0)
            center_src = np.mean(template_corners_by_id[tag_id], axis=0)
            
            src_centers.append(center_dst)
            dst_centers.append(center_src)
            matched_pairs.append((tag_id, obs))
            
        if len(matched_pairs) < 3:
            return None, 0.0
            
        src_centers = np.array(src_centers, dtype=np.float32)
        dst_centers = np.array(dst_centers, dtype=np.float32)
        
        H_centers, _ = cv2.findHomography(src_centers, dst_centers, cv2.RANSAC, 10.0)
        if H_centers is None:
            return None, 0.0
            
        # Align corners individually for each tag to find correct rotation/shift
        src_pts = []
        dst_pts = []
        for tag_id, obs in matched_pairs:
            obs_corners = obs["corners"]
            tpl_corners = template_corners_by_id[tag_id]
            
            try:
                H_centers_inv = np.linalg.inv(H_centers)
            except np.linalg.LinAlgError:
                return None, 0.0
                
            tpl_corners_h = tpl_corners.reshape(-1, 1, 2)
            proj_corners = cv2.perspectiveTransform(tpl_corners_h, H_centers_inv).reshape(4, 2)
            
            best_shift = 0
            min_err = float('inf')
            for shift in range(4):
                shifted_obs = np.roll(obs_corners, shift, axis=0)
                err = np.sum(np.linalg.norm(shifted_obs - proj_corners, axis=1))
                if err < min_err:
                    min_err = err
                    best_shift = shift
                    
            aligned_obs = np.roll(obs_corners, best_shift, axis=0)
            
            for idx in range(4):
                src_pts.append(aligned_obs[idx])  # Pixels
                dst_pts.append(tpl_corners[idx])   # Millimeters
                
        src_pts = np.array(src_pts, dtype=np.float32)
        dst_pts = np.array(dst_pts, dtype=np.float32)
        
        H_pixel_to_mm, inliers = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        if H_pixel_to_mm is None:
            H_pixel_to_mm = H_centers
            
        # 3. Compute H_coarse (Reference Image Pixels -> Current Camera Pixels)
        # H_coarse = H_mm_to_pixel * H_ref_to_mm
        try:
            H_mm_to_pixel = np.linalg.inv(H_pixel_to_mm)
        except np.linalg.LinAlgError:
            return None, 0.0
            
        H_coarse = H_mm_to_pixel @ H_ref_to_mm
        
        inliers_count = int(np.sum(inliers)) if inliers is not None else len(src_pts)
        inlier_ratio = inliers_count / len(src_pts) if len(src_pts) > 0 else 0.0
        
        tags_matched = len(matched_pairs)
        total_tags = len(blueprint.april_tags)
        tag_ratio = tags_matched / max(1, total_tags)
        confidence = tag_ratio * inlier_ratio
        
        return H_coarse, confidence

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
