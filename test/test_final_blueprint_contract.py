from __future__ import annotations

import sys
import json
from pathlib import Path
from datetime import datetime
import numpy as np
import cv2
import matplotlib.pyplot as plt

from PySide6.QtWidgets import QApplication

from core.calibration.image_loader import load_image
from core.detection.apriltag_detector import AprilTagDetector
from core.calibration.scale_calibrator import calibrate_scale
from core.roi.selector import ROISelector
from core.detection.silhouette_detector import detect_silhouette_candidates
from core.ui.candidate_selector import SilhouetteCandidateSelector
from core.ui.contour_editor import ContourEditor
from core.ui.zone_editor import ZonePolygonEditor
from core.ui.feature_editor import FeatureRegionEditor
from core.geometry.target_coordinates import get_target_bounds, pixel_to_normalized, normalized_to_pixel
from core.models import Blueprint, AprilTagReference, FeatureRegion, Zone, VisualFeature, VisualFeatureSet


def verify_blueprint_contract(saved_bp: Blueprint, loaded_bp: Blueprint) -> None:
    """
    Validates that the reloaded blueprint matches the saved contract values.
    """
    print()
    print("=" * 80)
    print("BLUEPRINT CONTRACT VERIFICATION REPORT")
    print("=" * 80)
    
    # 1. Metadata Checks
    print("[1. METADATA CONTRACT]")
    print(f"  ID Match         : {saved_bp.blueprint_id == loaded_bp.blueprint_id} (ID: {loaded_bp.blueprint_id})")
    print(f"  Type Match       : {saved_bp.target_type == loaded_bp.target_type} (Type: {loaded_bp.target_type})")
    print(f"  Version Match    : {saved_bp.format_version == loaded_bp.format_version} (Ver: {loaded_bp.format_version})")
    print(f"  Timestamp Match  : {saved_bp.created_at == loaded_bp.created_at} ({loaded_bp.created_at})")
    
    # 2. Geometry Checks
    print("-" * 50)
    print("[2. GEOMETRY CONTRACT]")
    print(f"  ROI Bounds Match : {saved_bp.roi_bounds == loaded_bp.roi_bounds} (ROI: {loaded_bp.roi_bounds})")
    print(f"  Silhouette Match : {len(saved_bp.silhouette) == len(loaded_bp.silhouette)} (Vertices: {len(loaded_bp.silhouette)})")
    print(f"  Zones Count Match: {len(saved_bp.zones) == len(loaded_bp.zones)} (Zones: {len(loaded_bp.zones)})")
    for sz, lz in zip(saved_bp.zones, loaded_bp.zones):
        print(f"    - Zone '{lz.zone_id}': polygon match={np.array_equal(sz.polygon, lz.polygon)}, score={lz.score}")

    # 3. Calibration & AprilTag Checks
    print("-" * 50)
    print("[3. CALIBRATION & REGISTRATION REFERENCE CONTRACT]")
    print(f"  Scale Match      : {saved_bp.pixels_per_mm == loaded_bp.pixels_per_mm:.6f} px/mm")
    print(f"  AprilTags Count  : {len(saved_bp.april_tags) == len(loaded_bp.april_tags)} (Count: {len(loaded_bp.april_tags)})")
    for st, lt in zip(saved_bp.april_tags, loaded_bp.april_tags):
        print(f"    - Tag #{lt.tag_id}: center={lt.center}, corners match={np.array_equal(st.corners, lt.corners)}")

    # 4. Feature Regions & Descriptor Checks
    print("-" * 50)
    print("[4. FEATURE REGIONS & DESCRIPTOR CONTRACT]")
    print(f"  Regions Match    : {len(saved_bp.feature_regions) == len(loaded_bp.feature_regions)} (Count: {len(loaded_bp.feature_regions)})")
    for sr, lr in zip(saved_bp.feature_regions, loaded_bp.feature_regions):
        print(f"    - Region '{lr.id}': type={lr.region_type}, priority={lr.priority}, shape normalized={lr.polygon.shape}")
        
    saved_feats = saved_bp.features.features if saved_bp.features else ()
    loaded_feats = loaded_bp.features.features if loaded_bp.features else ()
    print(f"  Features Count   : {len(saved_feats) == len(loaded_feats)} (Features count: {len(loaded_feats)})")
    
    if loaded_feats:
        sources = loaded_bp.features.quality_metrics.get("sources", {})
        print(f"    - Source breakdown: {sources}")
        # Check descriptor type/shape
        sample_desc = loaded_feats[0].descriptor
        print(f"    - Sample descriptor size: {sample_desc.shape} (dtype: {sample_desc.dtype})")
    
    print("=" * 80)


def main():
    image_path = "test_images/outdoor target.jpeg"
    
    # Initialize Qt
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)

    # ---------------------------------------------------------
    # STEP 1: Load Image & Run Scale Calibration / AprilTag Detection
    # ---------------------------------------------------------
    print(f"Loading source target image: {image_path}")
    image_data = load_image(image_path)
    
    print("Detecting reference AprilTags...")
    detector = AprilTagDetector(families="tag36h11", nthreads=4)
    apriltags = detector.detect(image_data)
    
    print(f"Detected {apriltags.count} AprilTags.")
    
    print("Calibrating target physical scale...")
    scale_result = calibrate_scale(apriltags, tag_size_mm=50.0)
    print(f"Scale calibration: {scale_result.pixels_per_mm:.6f} px/mm")

    # ---------------------------------------------------------
    # STEP 2: Select ROI
    # ---------------------------------------------------------
    print("Opening ROI Selection Dialog...")
    roi_selector = ROISelector(image_data)
    result = roi_selector.exec()
    if result != ROISelector.DialogCode.Accepted:
        print("Cancelled.")
        return
    roi = roi_selector.roi
    print(f"Selected ROI: {roi.x}, {roi.y}, {roi.width}, {roi.height}")

    # ---------------------------------------------------------
    # STEP 3: Auto Silhouette Detection & Candidate Selection
    # ---------------------------------------------------------
    print("Running automatic silhouette candidate generation...")
    detection_result = detect_silhouette_candidates(image_data, roi)
    
    print("Opening Silhouette Candidate Selector...")
    candidate_selector = SilhouetteCandidateSelector(
        image_data=image_data,
        roi=roi,
        candidates=detection_result.candidates,
        white_contour=detection_result.white_mask_contour
    )
    if candidate_selector.exec() != SilhouetteCandidateSelector.DialogCode.Accepted:
        print("Cancelled.")
        return
    initial_contour = candidate_selector.selected_contour

    # ---------------------------------------------------------
    # STEP 4: Human Silhouette Contour Correction
    # ---------------------------------------------------------
    print("Opening Silhouette Contour Editor...")
    contour_editor = ContourEditor(image_data, roi, initial_contour)
    if contour_editor.exec() != ContourEditor.DialogCode.Accepted:
        print("Cancelled.")
        return
    final_contour = contour_editor.final_contour
    print(f"Final Silhouette Contour has {len(final_contour)} vertices.")

    # ---------------------------------------------------------
    # STEP 5: Scoring Zones (Auto-snaps on creation!)
    # ---------------------------------------------------------
    print("Opening Scoring Zone Editor (Snaps automatically on placing 4 vertices)...")
    zone_editor = ZonePolygonEditor(
        image_data=image_data,
        roi=roi,
        initial_zones=[],
        silhouette_contour=final_contour
    )
    if zone_editor.exec() != ZonePolygonEditor.DialogCode.Accepted:
        print("Cancelled.")
        return
    final_zones = zone_editor.final_zones

    # ---------------------------------------------------------
    # STEP 6: Landmark & Feature Regions (Zooms automatically!)
    # ---------------------------------------------------------
    print("Opening Feature Region Editor (Zooms to fit target ROI automatically)...")
    feature_editor = FeatureRegionEditor(
        image_data=image_data,
        roi=roi,
        initial_regions=[],
        zones=final_zones,
        silhouette_contour=final_contour
    )
    if feature_editor.exec() != FeatureRegionEditor.DialogCode.Accepted:
        print("Cancelled.")
        return
    final_regions_roi = feature_editor.final_regions
    generated_features = feature_editor.generated_features

    # ---------------------------------------------------------
    # STEP 7: Compile and Serialize Blueprint Reference Package
    # ---------------------------------------------------------
    print("Assembling Blueprint Package contract...")
    
    # 1. Map AprilTagReference objects (absolute coordinates in reference image space)
    ref_tags = []
    for tag in apriltags.detections:
        ref_tags.append(AprilTagReference(
            tag_id=tag.tag_id,
            corners=tag.corners,
            center=tag.center
        ))
        
    # 2. Map FeatureRegions to normalized target coordinates relative to silhouette bounds
    target_bounds = get_target_bounds(roi, final_contour)
    normalized_regions = []
    for r in final_regions_roi:
        poly_norm = pixel_to_normalized(r.polygon, target_bounds)
        normalized_regions.append(FeatureRegion(
            id=r.id,
            polygon=poly_norm,
            region_type=r.region_type,
            priority=r.priority,
            min_features=r.min_features,
            max_features=r.max_features,
            metadata=r.metadata.copy() if r.metadata else {}
        ))
        
    # Build complete Blueprint contract object
    blueprint = Blueprint(
        blueprint_id="bp_outdoor_target_001",
        target_type="outdoor_paper",
        name="Outdoor Target Reference",
        format_version="1.0.0",
        created_at=datetime.utcnow().isoformat() + "Z",
        roi_bounds=(roi.x, roi.y, roi.width, roi.height),
        pixels_per_mm=scale_result.pixels_per_mm,
        mm_per_pixel=scale_result.millimeters_per_pixel,
        tag_size_mm=scale_result.reference_tag_size_mm,
        april_tags=ref_tags,
        silhouette=final_contour,
        zones=final_zones,
        feature_regions=normalized_regions,
        features=generated_features
    )
    
    package_dir = Path("blueprints/bp_outdoor_target_001")
    print(f"Saving blueprint package to '{package_dir}'...")
    blueprint.save_package(package_dir, image_data.image)
    print("Blueprint package successfully serialized.")

    # ---------------------------------------------------------
    # STEP 8: Reload Blueprint Package & Verify Contract
    # ---------------------------------------------------------
    print(f"Reloading blueprint package from '{package_dir}'...")
    loaded_blueprint, loaded_image = Blueprint.load_package(package_dir)
    print("Blueprint package successfully reconstituted.")
    
    # Run structural and value assertions
    verify_blueprint_contract(blueprint, loaded_blueprint)

    # ---------------------------------------------------------
    # STEP 9: Visualize Reconstituted Target Blueprint
    # ---------------------------------------------------------
    # Crop the ROI to display loaded result
    cropped_bgr = loaded_image[roi.y:roi.y+roi.height, roi.x:roi.x+roi.width]
    cropped_rgb = cv2.cvtColor(cropped_bgr, cv2.COLOR_BGR2RGB)

    # Figure 1: Reconstituted Target Geometry (Silhouette & Solid Zones)
    fig1, ax1 = plt.subplots(figsize=(8, 8))
    ax1.imshow(cropped_rgb)
    ax1.set_title(f"Contract Reconstitution - Mode 1: Target Geometry")
    
    # Silhouette (load coordinates are ROI-local, plot directly)
    sil_pts = loaded_blueprint.silhouette - np.array([[[roi.x, roi.y]]], dtype=np.int32)
    sil_draw = np.vstack([sil_pts.reshape(-1, 2), sil_pts.reshape(-1, 2)[0]])
    ax1.plot(sil_draw[:, 0], sil_draw[:, 1], color="blue", linestyle="--", linewidth=2.0, label="Target Silhouette")

    zone_colors = plt.colormaps["tab10"]
    for idx, zone in enumerate(loaded_blueprint.zones):
        pts = zone.polygon.reshape(-1, 2)
        pts_draw = np.vstack([pts, pts[0]])
        c = zone_colors(idx)
        ax1.plot(pts_draw[:, 0], pts_draw[:, 1], color=c, linewidth=2.0, label=f"Zone {zone.zone_id}")
        ax1.fill(pts[:, 0], pts[:, 1], color=c, alpha=0.10)
    ax1.legend(loc="upper right")
    ax1.axis("off")
    fig1.tight_layout()

    # Figure 2: Reconstituted Features & Normalized Regions (Mapped back to pixel space)
    fig2, ax2 = plt.subplots(figsize=(8, 8))
    ax2.imshow(cropped_rgb)
    ax2.set_title("Contract Reconstitution - Mode 2: Features & Mapped Regions")
    
    # Plot feature regions mapped back from normalized coordinates to pixels using target_bounds
    loaded_bounds = get_target_bounds(roi, loaded_blueprint.silhouette)
    for idx, region in enumerate(loaded_blueprint.feature_regions):
        # Convert normalized coordinates to ROI-local pixels
        poly_roi = normalized_to_pixel(region.polygon, loaded_bounds)
        pts_draw = np.vstack([poly_roi.reshape(-1, 2), poly_roi.reshape(-1, 2)[0]])
        
        if region.region_type == "stable":
            color = "purple"
            style = "--"
        elif region.region_type == "optional":
            color = "dodgerblue"
            style = "-."
        else:
            color = "orange"
            style = ":"
            
        ax2.plot(pts_draw[:, 0], pts_draw[:, 1], color=color, linestyle=style, linewidth=2.0, 
                label=f"Region {region.id} ({region.region_type})" if idx < 5 else "")

    # Plot features by source
    if loaded_blueprint.features:
        for f in loaded_blueprint.features.features:
            # Coordinates are ROI-local
            if f.source == "silhouette_boundary":
                c = "deepskyblue"
            elif f.source == "zone_boundary":
                c = "orange"
            else:
                c = "green"
            ax2.scatter(f.x, f.y, color=c, s=12, edgecolors="black", linewidths=0.5)

    ax2.legend(loc="upper right")
    ax2.axis("off")
    fig2.tight_layout()

    print("Displaying reloaded contract blueprints. Close Matplotlib windows to exit.")
    plt.show()


if __name__ == "__main__":
    main()
