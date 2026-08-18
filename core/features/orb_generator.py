from __future__ import annotations

import cv2
import numpy as np
from core.models import (
    ImageData,
    ROI,
    Zone,
    FeatureRegion,
    VisualFeature,
    VisualFeatureSet,
    AprilTagDetection,
)


def generate_zone_corner_regions(zones: list[Zone], radius: float = 15.0) -> list[FeatureRegion]:
    """
    Automatically generate opportunistic feature regions around the corners of all scoring zones.
    """
    regions = []
    for zone in zones:
        pts = zone.polygon.reshape(-1, 2)
        for idx, pt in enumerate(pts):
            px, py = pt[0], pt[1]
            square = np.array([
                [[px - radius, py - radius]],
                [[px + radius, py - radius]],
                [[px + radius, py + radius]],
                [[px - radius, py + radius]]
            ], dtype=np.int32)
            
            regions.append(FeatureRegion(
                id=f"zone_{zone.zone_id}_corner_{idx}",
                polygon=square,
                region_type="zone_corner",
                priority=2,
                min_features=1,
                max_features=5,
                metadata={"zone_id": zone.zone_id, "corner_index": idx}
            ))
    return regions


class ORBFeatureGenerator:
    """
    ORB Feature Generator that extracts descriptors from multiple evidence sources
    (feature regions, silhouette boundaries, and scoring zone boundaries), masks out AprilTags,
    and applies a radius-based spatial suppression filter for diversity.
    """

    def __init__(
        self,
        nfeatures: int = 5000,
        scaleFactor: float = 1.2,
        nlevels: int = 8,
        edgeThreshold: int = 31,
        firstLevel: int = 0,
        WTA_K: int = 2,
        scoreType: int = cv2.ORB_FAST_SCORE,
        patchSize: int = 31,
        fastThreshold: int = 20,
    ) -> None:
        self.orb = cv2.ORB_create(
            nfeatures=nfeatures,
            scaleFactor=scaleFactor,
            nlevels=nlevels,
            edgeThreshold=edgeThreshold,
            firstLevel=firstLevel,
            WTA_K=WTA_K,
            scoreType=scoreType,
            patchSize=patchSize,
            fastThreshold=fastThreshold,
        )

    def _sample_contour_boundary(self, contour: np.ndarray, step_px: float = 15.0) -> list[tuple[float, float]]:
        """
        Samples points along the perimeter of a polygon/contour at regular pixel intervals.
        """
        pts = contour.reshape(-1, 2)
        if len(pts) < 2:
            return [(float(p[0]), float(p[1])) for p in pts]
            
        sampled = []
        for i in range(len(pts)):
            p1 = pts[i]
            p2 = pts[(i + 1) % len(pts)]
            dist = np.linalg.norm(p2 - p1)
            if dist == 0:
                continue
            
            num_steps = int(dist / step_px)
            for s in range(max(1, num_steps)):
                t = s / max(1, num_steps)
                pt = p1 + t * (p2 - p1)
                sampled.append((float(pt[0]), float(pt[1])))
        return sampled

    def generate(
        self,
        image_data: ImageData,
        roi: ROI,
        regions: list[FeatureRegion],
        apriltags: list[AprilTagDetection] = None,
        zones: list[Zone] = None,
        silhouette_contour: np.ndarray = None,
        min_distance: float = 8.0,
    ) -> VisualFeatureSet:
        """
        Generate visual features from regions, silhouette boundaries, and zone boundaries.
        """
        crop = roi.crop(image_data.image)
        if crop.size == 0:
            return VisualFeatureSet(features=(), quality_metrics={"error": "Empty ROI crop"})

        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape

        # 1. Build AprilTag exclusion mask
        tag_mask = np.ones((h, w), dtype=np.uint8) * 255
        if apriltags:
            for tag in apriltags:
                corners_roi = tag.corners - np.array([roi.x, roi.y])
                cv2.fillPoly(tag_mask, [corners_roi.astype(np.int32)], 0)

        candidates = []  # Stores list of (cv2.KeyPoint, descriptor, source, region_id, anchor_info)

        # 2. Source A: User-defined Landmark/Feature Regions
        for region in regions:
            region_mask = np.zeros((h, w), dtype=np.uint8)
            cv2.fillPoly(region_mask, [region.polygon.astype(np.int32)], 255)
            combined_mask = cv2.bitwise_and(region_mask, tag_mask)

            kps = self.orb.detect(gray, mask=combined_mask)
            if not kps:
                continue

            kps, descs = self.orb.compute(gray, kps)
            if descs is not None:
                for kp, desc in zip(kps, descs):
                    candidates.append((
                        kp, desc, "landmark_region", region.id,
                        region.metadata if region.region_type == "zone_corner" else None
                    ))

        # 3. Source B: Silhouette Boundary (sampled)
        if silhouette_contour is not None and len(silhouette_contour) > 0:
            sil_pts = self._sample_contour_boundary(silhouette_contour, step_px=15.0)
            valid_sil_pts = []
            for sx, sy in sil_pts:
                if 0 <= int(sx) < w and 0 <= int(sy) < h and tag_mask[int(sy), int(sx)] > 0:
                    valid_sil_pts.append((sx, sy))
            
            sil_kps = [cv2.KeyPoint(x, y, 15.0) for x, y in valid_sil_pts]
            sil_kps, sil_descs = self.orb.compute(gray, sil_kps)
            if sil_descs is not None:
                for kp, desc in zip(sil_kps, sil_descs):
                    candidates.append((kp, desc, "silhouette_boundary", None, None))

        # 4. Source C: Scoring Zone Boundaries (sampled)
        if zones:
            for zone in zones:
                zone_pts = self._sample_contour_boundary(zone.polygon, step_px=15.0)
                valid_zone_pts = []
                for zx, zy in zone_pts:
                    if 0 <= int(zx) < w and 0 <= int(zy) < h and tag_mask[int(zy), int(zx)] > 0:
                        valid_zone_pts.append((zx, zy))

                zone_kps = [cv2.KeyPoint(x, y, 15.0) for x, y in valid_zone_pts]
                zone_kps, zone_descs = self.orb.compute(gray, zone_kps)
                if zone_descs is not None:
                    for kp, desc in zip(zone_kps, zone_descs):
                        candidates.append((kp, desc, "zone_boundary", zone.zone_id, None))

        # 5. Apply Spatial Diversity Suppression Filter (Non-Maximum Suppression)
        # Sort candidates by response score descending (best quality first)
        candidates.sort(key=lambda x: x[0].response, reverse=True)

        selected_features = []
        selected_pts = []

        for kp, desc, source, region_id, anchor_info in candidates:
            kpx, kpy = kp.pt[0], kp.pt[1]
            
            # Check if this point is too close to any already selected keypoints
            too_close = False
            for sx, sy in selected_pts:
                dist = np.sqrt((kpx - sx)**2 + (kpy - sy)**2)
                if dist < min_distance:
                    too_close = True
                    break
            
            if not too_close:
                selected_pts.append((kpx, kpy))
                selected_features.append(VisualFeature(
                    x=float(kpx),
                    y=float(kpy),
                    size=float(kp.size),
                    angle=float(kp.angle),
                    response=float(kp.response),
                    octave=int(kp.octave),
                    class_id=int(kp.class_id),
                    descriptor=desc,
                    source=source,
                    region_id=region_id,
                    anchor_info=anchor_info
                ))

        # 6. Compile quality metrics & stats breakdown
        landmark_cnt = sum(1 for f in selected_features if f.source == "landmark_region")
        sil_cnt = sum(1 for f in selected_features if f.source == "silhouette_boundary")
        zone_cnt = sum(1 for f in selected_features if f.source == "zone_boundary")

        region_stats = {}
        for region in regions:
            cnt = sum(1 for f in selected_features if f.region_id == region.id)
            region_stats[region.id] = {
                "detected": sum(1 for c in candidates if c[3] == region.id),
                "selected": cnt,
                "status": "HIGH" if cnt >= region.min_features else "LOW"
            }

        mean_response = float(np.mean([f.response for f in selected_features])) if selected_features else 0.0

        quality_metrics = {
            "total_features": len(selected_features),
            "mean_response": mean_response,
            "sources": {
                "landmark_region": landmark_cnt,
                "silhouette_boundary": sil_cnt,
                "zone_boundary": zone_cnt
            },
            "regions": region_stats,
            "coverage_ratio": sum(1 for stat in region_stats.values() if stat["selected"] > 0) / (len(regions) + 1e-5)
        }

        return VisualFeatureSet(
            features=tuple(selected_features),
            quality_metrics=quality_metrics
        )
