from __future__ import annotations

import cv2
import numpy as np
from runtime.models import Bullet, ShotResult, RuntimeGeometry

class BulletPipeline:
    """
    Detects bullet holes in the target, measures their physical dimensions,
    and assigns scores based on transformed scoring zones.
    """
    def __init__(self, min_area: float = 30.0, max_area: float = 800.0, min_circularity: float = 0.6) -> None:
        self.min_area = min_area
        self.max_area = max_area
        self.min_circularity = min_circularity

    def detect_bullets(
        self,
        frame: np.ndarray,
        geometry: RuntimeGeometry
    ) -> list[tuple[tuple[float, float], float]]:
        """
        Detects circular dark holes (bullets) within the target silhouette region.
        Returns a list of ((cx, cy), diameter_px).
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape[:2]
        
        # Mask out everything outside the target silhouette bounds
        mask = np.zeros((h, w), dtype=np.uint8)
        if len(geometry.silhouette) > 0:
            cv2.fillPoly(mask, [geometry.silhouette], 255)
        else:
            cv2.fillPoly(mask, [geometry.search_region], 255)
            
        # Apply threshold to find dark contours (bullet holes)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        _, thresh = cv2.threshold(blurred, 60, 255, cv2.THRESH_BINARY_INV)
        
        # Combine threshold with target mask
        thresh = cv2.bitwise_and(thresh, mask)
        
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        bullets = []
        for c in contours:
            area = cv2.contourArea(c)
            if not (self.min_area <= area <= self.max_area):
                continue
                
            perimeter = cv2.arcLength(c, True)
            if perimeter == 0:
                continue
                
            circularity = (4 * np.pi * area) / (perimeter ** 2)
            if circularity < self.min_circularity:
                continue
                
            # Compute center and bounding size
            M = cv2.moments(c)
            if M["m00"] == 0:
                continue
            cx = float(M["m10"] / M["m00"])
            cy = float(M["m01"] / M["m00"])
            
            # Approximate diameter as mean of bounding rect dimensions
            _, _, bw, bh = cv2.boundingRect(c)
            diameter_px = float((bw + bh) / 2.0)
            
            bullets.append(((cx, cy), diameter_px))
            
        return bullets

    def process_shot_frame(
        self,
        shot_id: int,
        frame: np.ndarray,
        geometry: RuntimeGeometry,
        pixels_per_mm: float,
        registration_confidence: float
    ) -> ShotResult:
        """
        Executes the entire bullet detection, measurement, and scoring pipeline.
        """
        detected_raw = self.detect_bullets(frame, geometry)
        
        bullets = []
        total_score = 0.0
        
        for idx, (center, diameter_px) in enumerate(detected_raw):
            diameter_mm = diameter_px / pixels_per_mm
            
            # Find the best matching zone (matching polygon with the highest score)
            matched_zone_id = None
            matched_score = 0.0
            
            for zone_id, poly, score in geometry.zones:
                # Point-in-polygon test (returns >= 0 if inside or on boundary)
                dist = cv2.pointPolygonTest(poly, center, False)
                if dist >= 0:
                    # If nested/overlapping, assign to the one yielding the higher score
                    if score > matched_score:
                        matched_score = score
                        matched_zone_id = zone_id
            
            total_score += matched_score
            
            bullets.append(Bullet(
                center_px=center,
                diameter_px=diameter_px,
                diameter_mm=diameter_mm,
                zone_id=matched_zone_id,
                score=matched_score,
                confidence=1.0  # Detection confidence
            ))
            
        return ShotResult(
            shot_id=shot_id,
            bullets=bullets,
            registration_confidence=registration_confidence,
            pixels_per_mm=pixels_per_mm,
            total_score=total_score
        )
