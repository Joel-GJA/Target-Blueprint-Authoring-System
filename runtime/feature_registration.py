from __future__ import annotations

import cv2
import numpy as np
from runtime.models import RuntimeBlueprint, VisualFeatureRef

class FeatureRegistration:
    """
    Refines target alignment using localized descriptor matching.
    Utilizes H_coarse to predict target locations and constrains search window size.
    """
    def __init__(self, search_radius: float = 40.0, max_hamming_dist: int = 45) -> None:
        self.search_radius = search_radius
        self.max_hamming_dist = max_hamming_dist
        self.orb = cv2.ORB_create(nfeatures=6000)
        self.last_matches = None

    def refine_registration(
        self,
        blueprint: RuntimeBlueprint,
        frame: np.ndarray,
        H_coarse: np.ndarray,
        search_region_poly: np.ndarray
    ) -> tuple[np.ndarray | None, float]:
        """
        Extracts features within the search region, performs localized matching,
        and computes a refined homography H_refined.
        """
        # Convert frame to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape[:2]
        
        # Create mask for the search region
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(mask, [search_region_poly.astype(np.int32)], 255)
        
        # Detect ORB features in the search region
        kps_dst = self.orb.detect(gray, mask=mask)
        if not kps_dst:
            return None, 0.0
            
        kps_dst, descs_dst = self.orb.compute(gray, kps_dst)
        if descs_dst is None or len(kps_dst) == 0:
            return None, 0.0
            
        # Build spatial coordinates map for observed features
        obs_pts = np.array([kp.pt for kp in kps_dst], dtype=np.float32)
        
        src_matched = []
        dst_matched = []
        
        rx, ry, _, _ = blueprint.roi_bounds
        
        # Match each reference feature locally
        for ref_feat in blueprint.features:
            # Reference coordinate in image space
            x_ref = ref_feat.x + rx
            y_ref = ref_feat.y + ry
            
            # Predict its location in the current camera frame
            pred_pt = cv2.perspectiveTransform(
                np.array([[[x_ref, y_ref]]], dtype=np.float32), 
                H_coarse
            ).reshape(-1)
            
            px, py = pred_pt[0], pred_pt[1]
            
            # Find candidate keypoints within local radius
            dists = np.sqrt((obs_pts[:, 0] - px)**2 + (obs_pts[:, 1] - py)**2)
            candidates_idx = np.where(dists < self.search_radius)[0]
            
            if len(candidates_idx) == 0:
                continue
                
            # Find the best descriptor match in the local candidates
            best_idx = -1
            best_dist = 999
            
            for idx in candidates_idx:
                desc_dst = descs_dst[idx]
                dist = int(cv2.norm(ref_feat.descriptor, desc_dst, cv2.NORM_HAMMING))
                if dist < best_dist:
                    best_dist = dist
                    best_idx = idx
                    
            if best_idx != -1 and best_dist < self.max_hamming_dist:
                # Correspondence found!
                # Store source point (absolute reference image space)
                src_matched.append([x_ref, y_ref])
                # Store destination point (absolute camera frame space)
                dst_matched.append([obs_pts[best_idx, 0], obs_pts[best_idx, 1]])
                
        if len(src_matched) < 4:
            return None, 0.0
            
        src_matched = np.array(src_matched, dtype=np.float32)
        dst_matched = np.array(dst_matched, dtype=np.float32)
        
        # Refined RANSAC Homography
        H_refined, inliers = cv2.findHomography(src_matched, dst_matched, cv2.RANSAC, 3.0)
        
        self.last_matches = {
            "src": src_matched.tolist(),
            "dst": dst_matched.tolist(),
            "inliers": inliers.flatten().tolist() if inliers is not None else []
        }
        
        if H_refined is None:
            return None, 0.0
            
        # Calculate confidence metric based on number of matches and inlier ratio
        inliers_count = int(np.sum(inliers))
        inlier_ratio = inliers_count / len(inliers) if len(inliers) > 0 else 0.0
        
        # Logarithmic saturation confidence weighting
        conf = float(min(1.0, (inliers_count / 15.0)) * inlier_ratio)
        
        return H_refined, conf
