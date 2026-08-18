from __future__ import annotations

import cv2
import numpy as np
from core.models import ImageData, ROI, FeatureRegion, VisualFeature, VisualFeatureSet
from core.features.orb_generator import ORBFeatureGenerator


def apply_affine_transform(
    image: np.ndarray,
    angle: float,
    scale: float,
    tx: float,
    ty: float,
    brightness: float = 0.0,
    blur_ksize: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Apply synthetic geometric and photometric perturbations to an image.
    
    Parameters
    ----------
    image : np.ndarray
        Source BGR or grayscale image.
    angle : float
        Rotation angle in degrees.
    scale : float
        Scale factor.
    tx : float
        Horizontal translation in pixels.
    ty : float
        Vertical translation in pixels.
    brightness : float
        Brightness offset (adds to pixel values).
    blur_ksize : int
        Gaussian blur kernel size (must be odd, 0 for no blur).
        
    Returns
    -------
    perturbed_image : np.ndarray
    transform_matrix : np.ndarray
        2x3 affine transformation matrix mapping original to perturbed coordinates.
    """
    h, w = image.shape[:2]
    center = (w / 2.0, h / 2.0)
    
    # 1. Get rotation and scaling matrix
    rot_mat = cv2.getRotationMatrix2D(center, angle, scale)
    
    # 2. Add translation
    rot_mat[0, 2] += tx
    rot_mat[1, 2] += ty
    
    # 3. Warp image
    perturbed = cv2.warpAffine(image, rot_mat, (w, h), borderMode=cv2.BORDER_REPLICATE)
    
    # 4. Photometric perturbations
    if brightness != 0:
        perturbed = cv2.convertScaleAbs(perturbed, alpha=1.0, beta=brightness)
        
    if blur_ksize > 0:
        if blur_ksize % 2 == 0:
            blur_ksize += 1
        perturbed = cv2.GaussianBlur(perturbed, (blur_ksize, blur_ksize), 0)
        
    return perturbed, rot_mat


def transform_polygon(polygon: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """
    Transform a polygon (N, 1, 2) using a 2x3 affine matrix.
    """
    pts = polygon.reshape(-1, 2)
    # Add a column of ones for homogeneous coordinates
    pts_homg = np.hstack([pts, np.ones((len(pts), 1))])
    pts_trans = pts_homg.dot(matrix.T)
    return pts_trans.reshape(-1, 1, 2).astype(np.int32)


class FeatureRobustnessExperiment:
    """
    Performs offline robustness experiments to test whether designated
    FeatureRegions continue producing repeatable features under synthetic perturbations.
    """

    def __init__(self, generator: ORBFeatureGenerator = None) -> None:
        self.generator = generator or ORBFeatureGenerator()

    def run_experiment(
        self,
        image_data: ImageData,
        roi: ROI,
        regions: list[FeatureRegion],
        reference_features: VisualFeatureSet,
        max_inlier_dist: float = 6.0,
    ) -> dict[str, dict]:
        """
        Run the robustness experiment on a set of regions.
        
        Evaluates the following perturbations:
        - Small Rotation (10 degrees)
        - Scale Change (0.9x and 1.1x)
        - Light Dimming (-35 brightness offset)
        - Gaussian Blur (5x5 kernel)
        
        Returns
        -------
        dict
            Results dictionary mapping region_id to reliability metrics.
        """
        # Perturbations list: (label, angle, scale, tx, ty, brightness, blur_ksize)
        perturbations = [
            ("rotate_ccw", 10.0, 1.0, 0.0, 0.0, 0.0, 0),
            ("rotate_cw", -10.0, 1.0, 0.0, 0.0, 0.0, 0),
            ("scale_down", 0.0, 0.9, 0.0, 0.0, 0.0, 0),
            ("scale_up", 0.0, 1.1, 0.0, 0.0, 0.0, 0),
            ("dim_lighting", 0.0, 1.0, 0.0, 0.0, -35.0, 0),
            ("blur", 0.0, 1.0, 0.0, 0.0, 0.0, 5),
        ]

        # Initialize results table
        region_results = {}
        for region in regions:
            region_results[region.id] = {
                "inlier_counts": [],
                "match_rates": [],
                "scores": {},
                "overall_score": 0.0,
                "robustness": "UNSTABLE"
            }

        crop = roi.crop(image_data.image)
        if crop.size == 0:
            return region_results

        # Group reference features by region
        ref_features_by_region = {}
        for region in regions:
            ref_features_by_region[region.id] = [
                f for f in reference_features.features if f.region_id == region.id
            ]

        # Process each perturbation
        for label, angle, scale, tx, ty, brightness, blur_ksize in perturbations:
            perturbed_img, M = apply_affine_transform(
                crop, angle, scale, tx, ty, brightness, blur_ksize
            )
            perturbed_image_data = ImageData(
                image=perturbed_img,
                metadata=image_data.metadata  # Reusing metadata format
            )
            
            # Map the regions to the perturbed coordinates
            perturbed_regions = []
            for region in regions:
                perturbed_poly = transform_polygon(region.polygon, M)
                perturbed_regions.append(FeatureRegion(
                    id=region.id,
                    polygon=perturbed_poly,
                    region_type=region.region_type,
                    min_features=region.min_features,
                    max_features=region.max_features,
                    metadata=region.metadata
                ))
            
            # Extract features from the perturbed image
            perturbed_feature_set = self.generator.generate(
                perturbed_image_data,
                ROI(0, 0, crop.shape[1], crop.shape[0]),  # ROI is the crop dimensions now
                perturbed_regions,
                apriltags=None
            )
            
            # Match features per region
            for region in regions:
                ref_feats = ref_features_by_region[region.id]
                pert_feats = [f for f in perturbed_feature_set.features if f.region_id == region.id]
                
                if not ref_feats or not pert_feats:
                    region_results[region.id]["scores"][label] = 0.0
                    region_results[region.id]["inlier_counts"].append(0)
                    region_results[region.id]["match_rates"].append(0.0)
                    continue
                
                # Match descriptors using brute force Hamming distance
                ref_descs = np.array([f.descriptor for f in ref_feats], dtype=np.uint8)
                pert_descs = np.array([f.descriptor for f in pert_feats], dtype=np.uint8)
                
                # Compute pairwise Hamming distances
                # ref_descs: (N, 32), pert_descs: (M, 32)
                # cv2.BFMatcher with NORM_HAMMING
                bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
                matches = bf.match(ref_descs, pert_descs)
                
                inliers_count = 0
                for match in matches:
                    ref_idx = match.queryIdx
                    pert_idx = match.trainIdx
                    
                    ref_f = ref_feats[ref_idx]
                    pert_f = pert_feats[pert_idx]
                    
                    # Compute expected position of reference feature in perturbed image
                    rx, ry = ref_f.x, ref_f.y
                    expected_x = M[0, 0] * rx + M[0, 1] * ry + M[0, 2]
                    expected_y = M[1, 0] * rx + M[1, 1] * ry + M[1, 2]
                    
                    # Compute spatial distance between expected and actual detected position
                    dist = np.sqrt((pert_f.x - expected_x) ** 2 + (pert_f.y - expected_y) ** 2)
                    if dist <= max_inlier_dist:
                        inliers_count += 1
                        
                match_rate = inliers_count / len(ref_feats)
                region_results[region.id]["scores"][label] = match_rate
                region_results[region.id]["inlier_counts"].append(inliers_count)
                region_results[region.id]["match_rates"].append(match_rate)

        # 5. Summarize overall scores
        for region in regions:
            rates = region_results[region.id]["match_rates"]
            overall = float(np.mean(rates)) if rates else 0.0
            region_results[region.id]["overall_score"] = overall
            
            # Robustness thresholds
            if overall >= 0.70:
                region_results[region.id]["robustness"] = "HIGHLY_ROBUST"
            elif overall >= 0.40:
                region_results[region.id]["robustness"] = "ROBUST"
            elif overall >= 0.15:
                region_results[region.id]["robustness"] = "MODERATE"
            else:
                region_results[region.id]["robustness"] = "UNSTABLE"

        return region_results
