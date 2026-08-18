from __future__ import annotations

import time
import numpy as np
from pathlib import Path
from runtime.models import (
    RuntimeBlueprint,
    RuntimeState,
    RuntimeStatus,
    RuntimeGeometry,
    ShotResult,
)
from runtime.blueprint_loader import BlueprintLoader
from runtime.coarse_registration import CoarseRegistration
from runtime.feature_registration import FeatureRegistration
from runtime.calibration import Calibration
from runtime.pre_shot_tracker import PreShotTracker
from runtime.bullet_pipeline import BulletPipeline

class RuntimeEngine:
    """
    Main state machine orchestrating target registration, tracking,
    and bullet scoring pipelines.
    """
    def __init__(self) -> None:
        self.blueprint: RuntimeBlueprint | None = None
        self.ref_image: np.ndarray | None = None
        
        # Sub-modules
        self.feature_reg = FeatureRegistration()
        self.tracker = PreShotTracker()
        self.bullet_pipe = BulletPipeline()
        
        # Engine state
        self.state = RuntimeState(
            status=RuntimeStatus.NO_TARGET,
            homography=None,
            pixels_per_mm=None,
            geometry=None,
            registration_confidence=0.0,
            last_frame_timestamp=None,
            frame_id=0
        )

    def load_blueprint_package(self, package_dir: str | Path) -> bool:
        """
        Loads and validates a blueprint package directory.
        """
        try:
            self.blueprint, self.ref_image = BlueprintLoader.load_blueprint(package_dir)
            self.state = RuntimeState(
                status=RuntimeStatus.ACQUIRING,
                homography=None,
                pixels_per_mm=None,
                geometry=None,
                registration_confidence=0.0,
                last_frame_timestamp=None,
                frame_id=0
            )
            return True
        except Exception as e:
            print(f"Failed to load blueprint package: {e}")
            self.state.status = RuntimeStatus.ERROR
            return False

    def process_frame(
        self,
        frame: np.ndarray,
        observed_tags: list[dict] = None,
        timestamp: float = None
    ) -> RuntimeState:
        """
        Processes a camera frame to perform target acquisition or fast tracking.
        """
        if self.blueprint is None:
            self.state.status = RuntimeStatus.NO_TARGET
            return self.state
            
        self.state.frame_id += 1
        if timestamp is None:
            timestamp = time.time()
        self.state.last_frame_timestamp = timestamp
        
        # Force target acquisition if observed tags are provided or status is not READY/CALIBRATED
        if (observed_tags is not None and len(observed_tags) > 0) or self.state.status in (
            RuntimeStatus.ACQUIRING,
            RuntimeStatus.NO_TARGET,
            RuntimeStatus.CORRECTION_FAILED
        ):
            if observed_tags is None or len(observed_tags) == 0:
                self.state.status = RuntimeStatus.ACQUIRING
                self.state.registration_confidence = 0.0
                return self.state
                
            # 1. Coarse AprilTag registration
            H_coarse, confidence_coarse = CoarseRegistration.compute_coarse_homography(
                self.blueprint,
                observed_tags
            )
            
            if H_coarse is None:
                self.state.status = RuntimeStatus.ACQUIRING
                self.state.registration_confidence = 0.0
                return self.state
                
            self.state.status = RuntimeStatus.COARSE_REGISTERED
            
            # 2. Get coarse target search region
            search_region = CoarseRegistration.get_runtime_search_region(
                self.blueprint,
                H_coarse,
                frame.shape
            )
            
            # 3. Constrained ORB feature refinement
            H_refined, confidence_refined = self.feature_reg.refine_registration(
                self.blueprint,
                frame,
                H_coarse,
                search_region
            )
            
            # If refinement succeeds, establish baseline H_calibrated
            H_calibrated = H_refined if H_refined is not None else H_coarse
            confidence = confidence_refined if H_refined is not None else confidence_coarse
            
            # 4. Generate Calibrated State & Geometry
            self.state.homography = H_calibrated
            self.state.pixels_per_mm = Calibration.compute_current_scale(self.blueprint, H_calibrated)
            self.state.geometry = Calibration.generate_runtime_geometry(self.blueprint, H_calibrated, frame.shape)
            self.state.registration_confidence = confidence
            self.state.status = RuntimeStatus.CALIBRATED
            
            # 5. Initialize the Fast Pre-Shot Tracker
            tracking_init = self.tracker.initialize(frame, self.state.geometry.search_region)
            if tracking_init:
                self.state.status = RuntimeStatus.READY
            else:
                self.state.status = RuntimeStatus.CORRECTION_FAILED
                
            return self.state
            
        # ------------------------------------------------------------------
        # PATH B: Fast Pre-Shot Tracking (Optical Flow)
        # ------------------------------------------------------------------
        elif self.state.status == RuntimeStatus.READY:
            H_current, confidence_track = self.tracker.update(frame, self.state.homography)
            
            if H_current is not None and confidence_track > 0.4:
                # Update current state geometry using incremental track homography
                self.state.homography = H_current
                self.state.pixels_per_mm = Calibration.compute_current_scale(self.blueprint, H_current)
                self.state.geometry = Calibration.generate_runtime_geometry(self.blueprint, H_current, frame.shape)
                self.state.registration_confidence = confidence_track
            else:
                # Tracking failure -> trigger recovery reacquisition path next frame!
                print("Pre-shot tracking lost: triggering recovery reacquisition.")
                self.state.status = RuntimeStatus.CORRECTION_FAILED
                self.state.registration_confidence = 0.0
                
            return self.state
            
        return self.state

    def process_shot(self, shot_id: int, frame: np.ndarray) -> ShotResult | None:
        """
        Captures and scores a bullet shot. Must be in READY (calibrated/tracked) status.
        """
        if self.state.status not in (RuntimeStatus.READY, RuntimeStatus.CALIBRATED):
            print(f"Cannot process shot: engine is not calibrated (Current Status: {self.state.status})")
            return None
            
        self.state.status = RuntimeStatus.SHOT_PROCESSING
        
        result = self.bullet_pipe.process_shot_frame(
            shot_id=shot_id,
            frame=frame,
            geometry=self.state.geometry,
            pixels_per_mm=self.state.pixels_per_mm,
            registration_confidence=self.state.registration_confidence
        )
        
        # Transition status back to READY (pre-shot tracking resumes on next update)
        self.state.status = RuntimeStatus.READY
        return result
