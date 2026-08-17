from __future__ import annotations

import cv2
import numpy as np

from core.models import ImageData, ROI
from core.hessian_detector import detect_hessian_ridges


def line_from_points(p1: tuple[float, float], p2: tuple[float, float]) -> tuple[float, float, float]:
    """
    Returns the line equation coefficients (a, b, c) for ax + by + c = 0
    passing through p1 and p2.
    """
    x1, y1 = p1
    x2, y2 = p2
    a = y2 - y1
    b = x1 - x2
    c = x2 * y1 - x1 * y2
    return a, b, c


def intersect_lines(
    line1: tuple[float, float, float],
    line2: tuple[float, float, float],
    fallback_pt: tuple[float, float]
) -> tuple[float, float]:
    """
    Finds the intersection point of two lines:
    a1*x + b1*y + c1 = 0
    a2*x + b2*y + c2 = 0
    If parallel, returns fallback_pt.
    """
    a1, b1, c1 = line1
    a2, b2, c2 = line2
    
    det = a1 * b2 - a2 * b1
    if abs(det) < 1e-5:
        return fallback_pt
        
    x = (b1 * c2 - b2 * c1) / det
    y = (c1 * a2 - c2 * a1) / det
    return x, y


def point_line_distance(pt: tuple[float, float], line_pts: tuple[tuple[float, float], tuple[float, float]]) -> float:
    """
    Computes the perpendicular distance from pt to the infinite line passing through line_pts.
    """
    x0, y0 = pt
    (x1, y1), (x2, y2) = line_pts
    
    denom = np.sqrt((y2 - y1)**2 + (x2 - x1)**2)
    if denom == 0:
        return np.sqrt((x0 - x1)**2 + (y0 - y1)**2)
        
    num = abs((y2 - y1) * x0 - (x2 - x1) * y0 + x2 * y1 - y2 * x1)
    return num / denom


def snap_zone_polygon(
    image_data: ImageData,
    roi: ROI,
    polygon: np.ndarray,
    search_margin: float = 15.0,
    hessian_threshold: int = 250,
    return_debug: bool = False,
) -> np.ndarray | tuple[np.ndarray, dict]:
    """
    Refines a user's rough polygon by searching for nearby printed boundaries
    using Hessian ridges and Hough transforms, then intersecting the refined edges.

    Parameters
    -------
    image_data : ImageData
        Source image data.
    roi : ROI
        The Region of Interest within which the coordinates are defined.
    polygon : np.ndarray
        Array of shape (N, 1, 2) or (N, 2) representing the ROI-local polygon vertices.
    search_margin : float
        Width of the search band around each edge (in pixels).
    hessian_threshold : int
        Threshold for Hessian ridge detection.
    return_debug : bool
        If True, return a tuple of (snapped_polygon, debug_data_dict) instead of just the polygon.

    Returns
    -------
    np.ndarray or tuple[np.ndarray, dict]
        The snapped/refined polygon contour (ROI-local coordinates) as shape (N, 1, 2).
        If return_debug is True, returns (snapped_polygon, debug_data_dict).
    """
    # Parse inputs
    pts = polygon.reshape(-1, 2).astype(np.float32)
    N = len(pts)
    if N < 3:
        if return_debug:
            return polygon.copy(), {}
        return polygon.copy()

    # 1. Run Hessian Ridge detection inside ROI
    try:
        ridges = detect_hessian_ridges(image_data, roi, threshold=hessian_threshold)
    except Exception as e:
        print(f"Warning: Hessian ridge detection failed: {e}. Snapping will fallback to original coordinates.")
        if return_debug:
            try:
                cropped = roi.crop(image_data.image)
            except Exception:
                cropped = None
            fallback_debug = {
                "cropped_image": cropped,
                "ridges": None,
                "corridors": [],
                "corridor_ridges": [],
                "detected_lines": [],
                "selected_lines": [],
                "original_polygon": polygon.copy(),
            }
            return polygon.copy(), fallback_debug
        return polygon.copy()

    # Initialize debug data if requested
    cropped = roi.crop(image_data.image)
    debug_data = {
        "cropped_image": cropped,
        "ridges": ridges,
        "corridors": [],
        "corridor_ridges": [],
        "detected_lines": [],
        "selected_lines": [],
        "original_polygon": polygon.copy(),
    }

    # 2. Refine each edge independently
    refined_lines: list[tuple[float, float, float]] = []

    for i in range(N):
        p1 = pts[i]
        p2 = pts[(i + 1) % N]
        
        # Original edge representation
        orig_line = line_from_points((p1[0], p1[1]), (p2[0], p2[1]))
        
        # Direction and length of the user's rough edge
        v_user = p2 - p1
        length = np.linalg.norm(v_user)
        if length == 0:
            refined_lines.append(orig_line)
            if return_debug:
                debug_data["corridors"].append(np.zeros_like(ridges))
                debug_data["corridor_ridges"].append(np.zeros_like(ridges))
                debug_data["detected_lines"].append([])
                debug_data["selected_lines"].append(None)
            continue
            
        # Draw a thick search corridor mask
        mask = np.zeros(ridges.shape, dtype=np.uint8)
        cv2.line(
            mask,
            (int(round(p1[0])), int(round(p1[1]))),
            (int(round(p2[0])), int(round(p2[1]))),
            255,
            thickness=int(round(2 * search_margin))
        )
        
        # Extract ridges inside the corridor
        corridor_ridges = cv2.bitwise_and(ridges, ridges, mask=mask)
        
        # Detect candidate lines
        min_line_len = max(5.0, length * 0.3)
        lines = cv2.HoughLinesP(
            corridor_ridges,
            rho=1,
            theta=np.pi / 180,
            threshold=8,
            minLineLength=int(round(min_line_len)),
            maxLineGap=12
        )
        
        if return_debug:
            debug_data["corridors"].append(mask.copy())
            debug_data["corridor_ridges"].append(corridor_ridges.copy())
            if lines is not None:
                lines_flat = []
                for l in lines:
                    pts_l = l.ravel().tolist()
                    if len(pts_l) == 4:
                        lines_flat.append(pts_l)
                debug_data["detected_lines"].append(lines_flat)
            else:
                debug_data["detected_lines"].append([])

        if lines is None or len(lines) == 0:
            # Fallback to original user line
            refined_lines.append(orig_line)
            if return_debug:
                debug_data["selected_lines"].append(None)
            continue
            
        # Evaluate and rank candidate lines
        best_line_eq = orig_line
        best_score = -float('inf')
        best_line_pts = None
        
        for line in lines:
            pts_coords = line.ravel()
            if len(pts_coords) != 4:
                continue
            x1, y1, x2, y2 = pts_coords
            v_cand = np.array([x2 - x1, y2 - y1], dtype=np.float32)
            cand_len = np.linalg.norm(v_cand)
            if cand_len == 0:
                continue
                
            # Compute distance penalty (average perp distance of user's endpoints to candidate line)
            d1 = point_line_distance((p1[0], p1[1]), ((x1, y1), (x2, y2)))
            d2 = point_line_distance((p2[0], p2[1]), ((x1, y1), (x2, y2)))
            dist_penalty = (d1 + d2) / 2.0
            
            # Compute angular deviation
            dot_prod = abs(np.dot(v_user, v_cand)) / (length * cand_len)
            dot_prod = max(0.0, min(1.0, dot_prod)) # Clamp numerical noise
            angle_rad = np.arccos(dot_prod)
            angle_deg = np.degrees(angle_rad)
            
            # Apply hard thresholds to avoid snapping to wrong orientations or far away details
            # Allow less than 3 degrees of angular change
            if dist_penalty > search_margin or angle_deg > 3.0:
                continue
                
            # Score formula: encourage longer lines with low distance/angle deviation
            score = cand_len - (1.5 * dist_penalty) - (2.0 * angle_deg)
            
            if score > best_score:
                best_score = score
                best_line_eq = line_from_points((x1, y1), (x2, y2))
                best_line_pts = (x1, y1, x2, y2)
                
        refined_lines.append(best_line_eq)
        if return_debug:
            debug_data["selected_lines"].append(best_line_pts)

    # 3. Intersect adjacent refined lines to reconstruct polygon corners
    snapped_pts = []
    for i in range(N):
        prev_line = refined_lines[(i - 1 + N) % N]
        curr_line = refined_lines[i]
        
        # Fallback is the original rough point coordinates
        orig_pt = (pts[i][0], pts[i][1])
        ix, iy = intersect_lines(prev_line, curr_line, orig_pt)
        
        # Clamp snapped coordinate to ROI dimensions
        ix = max(0.0, min(ix, float(roi.width - 1)))
        iy = max(0.0, min(iy, float(roi.height - 1)))
        
        snapped_pts.append([int(round(ix)), int(round(iy))])

    result_poly = np.array(snapped_pts, dtype=np.int32).reshape(-1, 1, 2)
    if return_debug:
        return result_poly, debug_data
    return result_poly
