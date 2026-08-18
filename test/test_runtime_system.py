from __future__ import annotations

import sys
import os
from pathlib import Path
import numpy as np
import cv2

# Add project root to sys.path
sys.path.append(str(Path(__file__).parent.parent.resolve()))

from runtime.models import RuntimeBlueprint, RuntimeStatus, Bullet, ShotResult
from runtime.blueprint_loader import BlueprintLoader
from runtime.coarse_registration import CoarseRegistration
from runtime.feature_registration import FeatureRegistration
from runtime.calibration import Calibration
from runtime.pre_shot_tracker import PreShotTracker
from runtime.bullet_pipeline import BulletPipeline
from runtime.engine import RuntimeEngine


def run_unit_tests():
    print()
    print("=" * 80)
    print("RUNNING RUNTIME MODULE UNIT TESTS")
    print("=" * 80)

    # 1. Check Blueprint Loader & validation
    package_dir = Path("blueprints/bp_outdoor_target_001")
    if not package_dir.exists():
        print("[SKIP] Reusable blueprint package 'blueprints/bp_outdoor_target_001' not found.")
        print("       Please run test/test_final_blueprint_contract.py first to generate it.")
        return
        
    print("TEST 1: Loading Blueprint Package...")
    blueprint, ref_image = BlueprintLoader.load_blueprint(package_dir)
    print(f"  Blueprint loaded successfully: ID={blueprint.blueprint_id}")
    print(f"  Reference image size: {ref_image.shape}")
    print(f"  AprilTags stored: {[t.tag_id for t in blueprint.april_tags]}")
    print(f"  Zones stored: {[z.zone_id for z in blueprint.zones]}")
    print(f"  Features loaded: {len(blueprint.features)}")
    
    # 2. Simulate perspective change for current frame
    print("\nTEST 2: Coarse AprilTag Homography...")
    # Target transform: 4-degree rotation + small offset
    theta = np.radians(4.0)
    cos_t, sin_t = np.cos(theta), np.sin(theta)
    H_sim = np.array([
        [cos_t, -sin_t, 35.0],
        [sin_t, cos_t, -15.0],
        [0.0, 0.0, 1.0]
    ], dtype=np.float32)
    
    h, w = ref_image.shape[:2]
    sim_frame = cv2.warpPerspective(ref_image, H_sim, (w, h))
    
    # Simulate detected tags corners by warping reference corners
    observed_tags = []
    for ref_tag in blueprint.april_tags:
        pts = ref_tag.corners.reshape(-1, 1, 2)
        warped_pts = cv2.perspectiveTransform(pts, H_sim).reshape(4, 2)
        observed_tags.append({
            "tag_id": ref_tag.tag_id,
            "corners": warped_pts
        })
        
    H_coarse, confidence = CoarseRegistration.compute_coarse_homography(blueprint, observed_tags)
    print(f"  H_coarse computed: confidence={confidence:.3f}")
    print("  Comparing simulated H and estimated H_coarse:")
    print(f"    Simulated H:\n{H_sim}")
    print(f"    Estimated H:\n{H_coarse}")
    
    # Assert estimated homography is very close to simulated homography
    assert H_coarse is not None, "Failed to compute coarse homography"
    assert np.allclose(H_coarse[:2], H_sim[:2], atol=1e-1), "Estimated homography deviates from ground truth"
    print("  [PASS] Coarse registration estimated homography matches simulated ground truth.")

    # 3. Constrained local ORB search refinement
    print("\nTEST 3: Constrained Feature Refinement...")
    search_region = CoarseRegistration.get_runtime_search_region(blueprint, H_coarse, sim_frame.shape)
    feat_reg = FeatureRegistration()
    H_refined, conf_refined = feat_reg.refine_registration(blueprint, sim_frame, H_coarse, search_region)
    print(f"  H_refined computed: confidence={conf_refined:.3f}")
    assert H_refined is not None, "Failed to refine homography using local descriptors"
    print("  [PASS] Refined homography computed successfully.")

    # 4. Geometry and physical scale warping
    print("\nTEST 4: Geometry Warping & Scale Calibration...")
    scale = Calibration.compute_current_scale(blueprint, H_refined)
    geometry = Calibration.generate_runtime_geometry(blueprint, H_refined, sim_frame.shape)
    
    print(f"  Blueprint scale: {blueprint.pixels_per_mm:.5f} px/mm")
    print(f"  Runtime scale  : {scale:.5f} px/mm")
    print(f"  Silhouette points warped: {len(geometry.silhouette)}")
    print(f"  Zones warped count      : {len(geometry.zones)}")
    assert len(geometry.zones) == len(blueprint.zones), "Warped zones count mismatch"
    print("  [PASS] Calibration scale adjustment and geometry warping validated.")

    # 5. Good features tracking (LK Optical Flow)
    print("\nTEST 5: Fast Pre-shot Tracking...")
    tracker = PreShotTracker()
    init_ok = tracker.initialize(sim_frame, geometry.search_region)
    print(f"  Tracker initialization: {init_ok}")
    assert init_ok, "Tracker failed to initialize"
    
    # Simulate a small frame shift (e.g. 3px shift)
    H_shift = np.array([
        [1.0, 0.0, 3.0],
        [0.0, 1.0, 2.0],
        [0.0, 0.0, 1.0]
    ], dtype=np.float32)
    next_frame = cv2.warpPerspective(sim_frame, H_shift, (w, h))
    
    H_track, conf_track = tracker.update(next_frame, H_refined)
    print(f"  Tracked H_current: confidence={conf_track:.3f}")
    assert H_track is not None, "Frame tracking failed"
    # Expected tracking shift
    expected_H = H_shift @ H_refined
    print("  Comparing expected tracked H and estimated H:")
    print(f"    Expected H:\n{expected_H}")
    print(f"    Tracked H:\n{H_track}")
    assert np.allclose(H_track[:2], expected_H[:2], atol=2e-1), "Tracked homography deviates from expected shift"
    print("  [PASS] LK optical flow incremental tracking matches shift.")

    # 6. Bullet detection and scoring assignment
    print("\nTEST 6: Bullet Detection & Scoring Assignment...")
    # Warp target geometry to sim_frame to find zone centroids
    # Draw a mock bullet hole (dark circle) inside the center zone (Zone 1 or similar)
    center_zone = next((z for z in geometry.zones if z[0] == "Zone 1"), geometry.zones[0])
    # Compute centroid of this zone in frame coordinates
    pts = center_zone[1].reshape(-1, 2)
    cx = int(np.mean(pts[:, 0]))
    cy = int(np.mean(pts[:, 1]))
    
    # Draw dark bullet hole (radius 8px) on sim_frame
    cv2.circle(sim_frame, (cx, cy), 8, (15, 15, 15), -1)
    
    bullet_pipe = BulletPipeline()
    result = bullet_pipe.process_shot_frame(
        shot_id=1,
        frame=sim_frame,
        geometry=geometry,
        pixels_per_mm=scale,
        registration_confidence=conf_refined
    )
    
    print(f"  Processed shot frame #1:")
    print(f"    Bullets detected: {len(result.bullets)}")
    for b in result.bullets:
        print(f"      - Hit at pixel ({b.center_px[0]:.1f}, {b.center_px[1]:.1f})")
        print(f"        Measured Diameter: {b.diameter_px:.1f} px | {b.diameter_mm:.2f} mm")
        print(f"        Assigned Zone    : {b.zone_id} (Score: {b.score})")
        
    assert len(result.bullets) >= 1, "Failed to detect bullet hole"
    closest_bullet = min(result.bullets, key=lambda b: np.sqrt((b.center_px[0] - cx)**2 + (b.center_px[1] - cy)**2))
    assert closest_bullet.zone_id == center_zone[0], f"Bullet closest to center assigned to {closest_bullet.zone_id} instead of {center_zone[0]}"
    print("  [PASS] Bullet detection, measurement, and scoring pipeline matches zone attributes.")

    print("\n" + "=" * 80)
    print("ALL RUNTIME SYSTEM MODULE TESTS COMPLETED SUCCESSFULLY!")
    print("=" * 80)


def run_pipeline_test():
    print()
    print("=" * 80)
    print("RUNNING RUNTIME ENGINE STATE MACHINE PIPELINE TEST")
    print("=" * 80)
    
    package_dir = Path("blueprints/bp_outdoor_target_001")
    if not package_dir.exists():
        print("[SKIP] Blueprint package missing. Cannot run engine pipeline.")
        return
        
    engine = RuntimeEngine()
    load_ok = engine.load_blueprint_package(package_dir)
    print(f"Engine initialization: load={load_ok}, status={engine.state.status}")
    assert load_ok
    assert engine.state.status == RuntimeStatus.ACQUIRING
    
    # ---------------------------------------------------------
    # Frame 1: Target Acquisition
    # ---------------------------------------------------------
    print("\n--- FRAME 1: ACQUIRING TARGET ---")
    # Simulate observed tags (Ground truth identity H = identity, no shift)
    observed_tags = []
    for ref_tag in engine.blueprint.april_tags:
        observed_tags.append({
            "tag_id": ref_tag.tag_id,
            "corners": ref_tag.corners
        })
        
    state = engine.process_frame(engine.ref_image, observed_tags)
    print(f"  Frame 1 State: status={state.status}, confidence={state.registration_confidence:.3f}")
    assert state.status == RuntimeStatus.READY, "Engine failed to transition to READY state"
    
    # ---------------------------------------------------------
    # Frame 2: Pre-shot tracking (Small movement)
    # ---------------------------------------------------------
    print("\n--- FRAME 2: FAST PRE-SHOT TRACKING ---")
    # Simulate a small shift: shift reference image 2px right, 2px down
    h, w = engine.ref_image.shape[:2]
    H_shift = np.array([
        [1.0, 0.0, 2.0],
        [0.0, 1.0, 2.0],
        [0.0, 0.0, 1.0]
    ], dtype=np.float32)
    shifted_frame = cv2.warpPerspective(engine.ref_image, H_shift, (w, h))
    
    # Process frame on the fast path (no observed_tags provided)
    state = engine.process_frame(shifted_frame)
    print(f"  Frame 2 State: status={state.status}, confidence={state.registration_confidence:.3f}")
    assert state.status == RuntimeStatus.READY
    
    # ---------------------------------------------------------
    # Frame 3: Capture Shot & Score
    # ---------------------------------------------------------
    print("\n--- FRAME 3: SHOT FIRED & DETECTED ---")
    # Draw a bullet hole in the shifted target frame
    center_zone = next((z for z in state.geometry.zones if z[0] == "Zone 1"), state.geometry.zones[0])
    pts = center_zone[1].reshape(-1, 2)
    cx, cy = int(np.mean(pts[:, 0])), int(np.mean(pts[:, 1]))
    cv2.circle(shifted_frame, (cx, cy), 8, (15, 15, 15), -1)
    
    shot_result = engine.process_shot(shot_id=101, frame=shifted_frame)
    print(f"  Shot 101 Result:")
    print(f"    Status after shot: {engine.state.status}")
    print(f"    Bullets scored   : {len(shot_result.bullets)}")
    print(f"    Total Score      : {shot_result.total_score}")
    print(f"    Pixels per mm    : {shot_result.pixels_per_mm:.4f}")
    
    assert shot_result is not None
    assert len(shot_result.bullets) >= 1
    closest_bullet = min(shot_result.bullets, key=lambda b: np.sqrt((b.center_px[0] - cx)**2 + (b.center_px[1] - cy)**2))
    assert closest_bullet.zone_id == center_zone[0], f"Bullet closest to center assigned to {closest_bullet.zone_id} instead of {center_zone[0]}"
    
    # ---------------------------------------------------------
    # Frame 4: Tracking Failure & Recovery Path
    # ---------------------------------------------------------
    print("\n--- FRAME 4: TRACKING LOST & AUTOMATIC RECOVERY ---")
    # Simulate a sudden massive target movement (e.g. blank frame or huge displacement)
    blank_frame = np.zeros_like(engine.ref_image)
    
    # Fast tracker should fail
    state = engine.process_frame(blank_frame)
    print(f"  Frame 4 State: status={state.status} (Expected: CORRECTION_FAILED)")
    assert state.status == RuntimeStatus.CORRECTION_FAILED
    
    # Frame 5: Reacquisition arrives with detected tags
    print("\n--- FRAME 5: REACQUISITION ON NEXT FRAME ---")
    # Back to aligned frame
    state = engine.process_frame(engine.ref_image, observed_tags)
    print(f"  Frame 5 State: status={state.status} (Expected: READY)")
    assert state.status == RuntimeStatus.READY

    print("\n" + "=" * 80)
    print("RUNTIME ENGINE STATE MACHINE PIPELINE TEST PASSED!")
    print("=" * 80)


if __name__ == "__main__":
    run_unit_tests()
    run_pipeline_test()
