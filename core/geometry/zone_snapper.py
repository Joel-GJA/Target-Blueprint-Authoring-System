from __future__ import annotations

from dataclasses import dataclass
import cv2
import numpy as np
from core.models import ImageData, ROI


@dataclass
class EdgeLocalizationResult:
    success: bool
    original_edge: tuple[tuple[float, float], tuple[float, float]]
    refined_edge: tuple[tuple[float, float], tuple[float, float]]
    offset_pixels: float
    peak_strength: float
    confidence: float
    offsets: np.ndarray
    aggregate_response: np.ndarray
    profile_responses: np.ndarray


def localize_edge(
    image_data: ImageData,
    roi: ROI,
    p1: np.ndarray,  # (2,) ROI-local coordinates
    p2: np.ndarray,  # (2,) ROI-local coordinates
    search_margin: float = 15.0,
    sample_spacing: float = 5.0,
    aggregation_method: str = "median",
) -> EdgeLocalizationResult:
    """
    Local edge refinement using perpendicular gradient intensity profiles.
    Calculates gradient magnitudes and aggregates them along the edge normals.
    """
    crop = roi.crop(image_data.image)
    if crop.size == 0:
        return EdgeLocalizationResult(
            success=False,
            original_edge=((float(p1[0]), float(p1[1])), (float(p2[0]), float(p2[1]))),
            refined_edge=((float(p1[0]), float(p1[1])), (float(p2[0]), float(p2[1]))),
            offset_pixels=0.0,
            peak_strength=0.0,
            confidence=0.0,
            offsets=np.array([]),
            aggregate_response=np.array([]),
            profile_responses=np.array([]),
        )

    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

    # Compute unsigned gradient magnitude using Scharr filter
    gx = cv2.Scharr(gray, cv2.CV_32F, 1, 0)
    gy = cv2.Scharr(gray, cv2.CV_32F, 0, 1)
    grad_mag = cv2.magnitude(gx, gy)

    v_edge = p2.astype(np.float32) - p1.astype(np.float32)
    length = np.linalg.norm(v_edge)
    
    if length == 0:
        return EdgeLocalizationResult(
            success=False,
            original_edge=((float(p1[0]), float(p1[1])), (float(p2[0]), float(p2[1]))),
            refined_edge=((float(p1[0]), float(p1[1])), (float(p2[0]), float(p2[1]))),
            offset_pixels=0.0,
            peak_strength=0.0,
            confidence=0.0,
            offsets=np.array([]),
            aggregate_response=np.array([]),
            profile_responses=np.array([]),
        )

    tangent = v_edge / length
    normal = np.array([-tangent[1], tangent[0]], dtype=np.float32)

    # Generate profile sample coordinates along the edge
    num_samples = max(2, int(np.ceil(length / sample_spacing)))
    sample_factors = np.linspace(0.0, 1.0, num_samples)

    # Define range of search offsets
    margin_int = int(round(search_margin))
    offsets = np.arange(-margin_int, margin_int + 1, dtype=np.int32)

    # Extract perpendicular gradient profile for each sample point
    profile_responses_list = []
    for t in sample_factors:
        pt = p1 + t * v_edge
        profile = []
        for offset in offsets:
            pos = pt + normal * offset
            cx = max(0.0, min(float(grad_mag.shape[1] - 1), float(pos[0])))
            cy = max(0.0, min(float(grad_mag.shape[0] - 1), float(pos[1])))
            val = cv2.getRectSubPix(grad_mag, (1, 1), (cx, cy))[0, 0]
            profile.append(val)
        profile_responses_list.append(profile)

    profile_responses = np.array(profile_responses_list, dtype=np.float32)

    # Aggregate profile evidence along the edge
    if aggregation_method == "mean":
        aggregate_response = np.mean(profile_responses, axis=0)
    elif aggregation_method == "trimmed_mean":
        sorted_profiles = np.sort(profile_responses, axis=0)
        trim = max(1, int(num_samples * 0.1))
        if num_samples > 2 * trim:
            aggregate_response = np.mean(sorted_profiles[trim:-trim], axis=0)
        else:
            aggregate_response = np.median(profile_responses, axis=0)
    else:  # 'median'
        aggregate_response = np.median(profile_responses, axis=0)

    # Evaluate peak strength and confidence metrics
    peak_idx = np.argmax(aggregate_response)
    peak_strength = float(aggregate_response[peak_idx])
    offset_pixels = float(offsets[peak_idx])

    # Find local maxima to calculate prominence/confidence ratio
    local_maxima = []
    for idx in range(1, len(aggregate_response) - 1):
        if aggregate_response[idx] > aggregate_response[idx - 1] and aggregate_response[idx] > aggregate_response[idx + 1]:
            local_maxima.append(idx)
            
    if len(aggregate_response) > 1:
        if aggregate_response[0] > aggregate_response[1]:
            local_maxima.append(0)
        if aggregate_response[-1] > aggregate_response[-2]:
            local_maxima.append(len(aggregate_response) - 1)

    second_best_val = 0.0
    for idx in local_maxima:
        if idx != peak_idx:
            second_best_val = max(second_best_val, float(aggregate_response[idx]))

    if second_best_val == 0.0:
        background_strength = float(np.mean(aggregate_response))
    else:
        background_strength = second_best_val

    confidence = peak_strength / (background_strength + 1e-5)
    success = confidence > 1.15 and peak_strength > 10.0

    p1_refined = p1 + normal * offset_pixels
    p2_refined = p2 + normal * offset_pixels

    return EdgeLocalizationResult(
        success=success,
        original_edge=((float(p1[0]), float(p1[1])), (float(p2[0]), float(p2[1]))),
        refined_edge=((float(p1_refined[0]), float(p1_refined[1])), (float(p2_refined[0]), float(p2_refined[1]))),
        offset_pixels=offset_pixels,
        peak_strength=peak_strength,
        confidence=confidence,
        offsets=offsets,
        aggregate_response=aggregate_response,
        profile_responses=profile_responses,
    )


def snap_zone_polygon(
    image_data: ImageData,
    roi: ROI,
    polygon: np.ndarray,  # (4, 1, 2) or (4, 2) ROI-local
    search_margin: float = 15.0,
    sample_spacing: float = 5.0,
    aggregation_method: str = "median",
    return_debug: bool = False,
) -> np.ndarray | tuple[np.ndarray, dict]:
    """
    Production scoring zone snapping using local perpendicular gradient-profile edge refinement.
    Reconstructs corners by intersecting adjacent refined boundary lines.
    
    Parameters
    ----------
    image_data : ImageData
        Source image data.
    roi : ROI
        Region of interest bounding box.
    polygon : np.ndarray
        Rough polygon boundary contour (4 vertices) in ROI-local coordinates.
    search_margin : float
        Corridor margin (pixels).
    sample_spacing : float
        Sample profile line spacing (pixels).
    aggregation_method : str
        Aggregation mode ('median', 'mean', 'trimmed_mean').
    return_debug : bool
        If True, returns a tuple of (snapped_polygon, debug_data_dict) for visualizer UI.
        
    Returns
    -------
    np.ndarray or tuple[np.ndarray, dict]
        Reconstructed snapped/refined polygon (4, 1, 2) in ROI-local coordinates.
    """
    pts = polygon.reshape(-1, 2)
    if len(pts) != 4:
        # Fallback for non-quad polygons
        if return_debug:
            return polygon.copy(), {}
        return polygon.copy()

    # Localize each of the 4 edges and compute line equations: A x + B y + C = 0
    line_eqs = []
    edges_results = []
    
    for i in range(4):
        p1 = pts[i]
        p2 = pts[(i + 1) % 4]
        
        # Run localizer on this edge
        res = localize_edge(
            image_data=image_data,
            roi=roi,
            p1=p1,
            p2=p2,
            search_margin=search_margin,
            sample_spacing=sample_spacing,
            aggregation_method=aggregation_method,
        )
        edges_results.append(res)
        
        # Line normal calculation from the refined points
        p1_ref = np.array(res.refined_edge[0], dtype=np.float32)
        p2_ref = np.array(res.refined_edge[1], dtype=np.float32)
        
        v_edge = p2_ref - p1_ref
        A = -v_edge[1]
        B = v_edge[0]
        C = -(A * p1_ref[0] + B * p1_ref[1])
        line_eqs.append((A, B, C))

    # Reconstruct the 4 corners by intersecting adjacent refined lines
    new_pts = []
    for i in range(4):
        A1, B1, C1 = line_eqs[(i - 1) % 4]
        A2, B2, C2 = line_eqs[i]
        
        det = A1 * B2 - A2 * B1
        if abs(det) > 1e-5:
            x = (-C1 * B2 + C2 * B1) / det
            y = (-A1 * C2 + A2 * C1) / det
            new_pts.append([int(round(x)), int(round(y))])
        else:
            # Fallback to original vertex if lines are parallel
            new_pts.append([int(round(pts[i][0])), int(round(pts[i][1]))])

    snapped_polygon = np.array(new_pts, dtype=np.int32).reshape(4, 1, 2)

    if return_debug:
        crop = roi.crop(image_data.image)
        crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        
        debug_data = {
            "cropped_image": crop_rgb,
            "edges_data": edges_results,
            "pts": pts.copy(),
            "search_margin": search_margin,
        }
        return snapped_polygon, debug_data

    return snapped_polygon
