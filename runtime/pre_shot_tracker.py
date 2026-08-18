from __future__ import annotations

import cv2
import numpy as np
from runtime.models import RuntimeState

class PreShotTracker:
    """
    Fast pre-shot tracker tracking small target motions frame-to-frame.
    Uses Lucas-Kanade optical flow to calculate incremental registration updates.
    """
    def __init__(self, max_corners: int = 100, quality_level: float = 0.01, min_distance: float = 10.0) -> None:
        self.max_corners = max_corners
        self.quality_level = quality_level
        self.min_distance = min_distance
        
        # LK parameters
        self.lk_params = dict(
            winSize=(21, 21),
            maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.03)
        )
        
        self.prev_gray: np.ndarray | None = None
        self.prev_pts: np.ndarray | None = None

    def initialize(self, frame: np.ndarray, search_region_poly: np.ndarray) -> bool:
        """
        Detects initial tracking features inside the target region of the baseline frame.
        """
        self.prev_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        h, w = self.prev_gray.shape[:2]
        
        # Create mask for target region
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(mask, [search_region_poly.astype(np.int32)], 255)
        
        # Find strong corners to track
        pts = cv2.goodFeaturesToTrack(
            self.prev_gray,
            maxCorners=self.max_corners,
            qualityLevel=self.quality_level,
            minDistance=self.min_distance,
            mask=mask
        )
        
        if pts is not None and len(pts) >= 4:
            self.prev_pts = pts
            return True
        else:
            self.prev_pts = None
            return False

    def update(self, frame: np.ndarray, prev_H: np.ndarray) -> tuple[np.ndarray | None, float]:
        """
        Tracks keypoints in the new frame and computes H_current = dH * prev_H.
        Returns (H_current, tracking_confidence).
        """
        if self.prev_gray is None or self.prev_pts is None:
            return None, 0.0
            
        curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Check for lack of texture/contrast (e.g., blank or constant frame)
        if np.std(curr_gray) < 10.0:
            self.prev_gray = None
            self.prev_pts = None
            return None, 0.0
        
        # Calculate optical flow
        curr_pts, status, err = cv2.calcOpticalFlowPyrLK(
            self.prev_gray,
            curr_gray,
            self.prev_pts,
            None,
            **self.lk_params
        )
        
        if curr_pts is None or status is None:
            self.prev_gray = None
            self.prev_pts = None
            return None, 0.0
            
        # Filter successful tracking points
        valid_idx = np.where(status == 1)[0]
        if len(valid_idx) < 4:
            self.prev_gray = None
            self.prev_pts = None
            return None, 0.0
            
        good_prev = self.prev_pts[valid_idx]
        good_curr = curr_pts[valid_idx]
        
        # Estimate incremental homography dH
        dH, inliers = cv2.findHomography(good_prev, good_curr, cv2.RANSAC, 3.0)
        
        if dH is None:
            self.prev_gray = None
            self.prev_pts = None
            return None, 0.0
            
        # Update tracking status for next frame
        self.prev_gray = curr_gray
        # Retain only successfully tracked inlier points to avoid drift
        inlier_idx = np.where(inliers.flatten() == 1)[0]
        if len(inlier_idx) >= 4:
            self.prev_pts = good_curr[inlier_idx].reshape(-1, 1, 2)
        else:
            self.prev_pts = good_curr.reshape(-1, 1, 2)
            
        # Compute updated homography
        H_current = dH @ prev_H
        
        # Calculate confidence score
        inlier_ratio = float(np.sum(inliers) / len(inliers)) if len(inliers) > 0 else 0.0
        points_ratio = float(len(good_curr) / len(self.prev_pts)) if len(self.prev_pts) > 0 else 0.0
        confidence = inlier_ratio * points_ratio
        
        return H_current, confidence
