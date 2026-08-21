from __future__ import annotations

import json
from pathlib import Path
import cv2
import numpy as np
from runtime.models import (
    RuntimeBlueprint,
    AprilTagRef,
    ZoneRef,
    FeatureRegionRef,
    VisualFeatureRef,
)

class BlueprintLoader:
    """
    Loads and validates a Blueprint Package from a package directory.
    Ensure that all required geometric structures and descriptors are intact.
    """
    @staticmethod
    def load_blueprint(package_dir: str | Path) -> tuple[RuntimeBlueprint, np.ndarray]:
        p_dir = Path(package_dir)
        json_path = p_dir / "blueprint.json"
        img_path = p_dir / "reference_image.jpg"
        
        if not json_path.exists():
            raise FileNotFoundError(f"Blueprint JSON file missing: {json_path}")
        if not img_path.exists():
            raise FileNotFoundError(f"Reference image file missing: {img_path}")
            
        # Load image
        ref_image = cv2.imread(str(img_path))
        if ref_image is None or ref_image.size == 0:
            raise ValueError(f"Failed to load or parse reference image from {img_path}")
            
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            
        # Validation checks
        required_keys = ["blueprint_id", "target_type", "roi_bounds", "scale", "april_tags", "silhouette", "zones", "feature_regions", "features"]
        for key in required_keys:
            if key not in data:
                raise ValueError(f"Invalid blueprint: missing required parameter '{key}'")
                
        # Parse AprilTags
        april_tags = []
        for tag in data["april_tags"]:
            april_tags.append(AprilTagRef(
                tag_id=int(tag["tag_id"]),
                corners=np.array(tag["corners"], dtype=np.float32),
                center=tuple(tag["center"])
            ))
            
        # Parse silhouette
        silhouette = np.array(data["silhouette"], dtype=np.int32) if data["silhouette"] is not None else None
        
        # Parse zones
        zones = []
        for z in data["zones"]:
            zones.append(ZoneRef(
                zone_id=z["zone_id"],
                polygon=np.array(z["polygon"], dtype=np.int32),
                score=float(z["score"]),
                name=z.get("name", "")
            ))
            
        # Parse feature regions
        feature_regions = []
        for r in data["feature_regions"]:
            feature_regions.append(FeatureRegionRef(
                id=r["id"],
                polygon=np.array(r["polygon"], dtype=np.float32),
                region_type=r["region_type"],
                priority=int(r["priority"])
            ))
            
        # Parse features
        features = []
        f_data = data["features"]
        for f in f_data.get("features", []):
            features.append(VisualFeatureRef(
                x=float(f["x"]),
                y=float(f["y"]),
                size=float(f["size"]),
                angle=float(f["angle"]),
                response=float(f["response"]),
                octave=int(f["octave"]),
                class_id=int(f["class_id"]),
                descriptor=np.array(f["descriptor"], dtype=np.uint8),
                source=f.get("source", "landmark_region"),
                region_id=f.get("region_id"),
                anchor_info=f.get("anchor_info")
            ))
            
        # Compile blueprint
        blueprint = RuntimeBlueprint(
            blueprint_id=data["blueprint_id"],
            target_type=data["target_type"],
            name=data.get("name", "Unnamed Blueprint"),
            format_version=data.get("format_version", "1.0.0"),
            created_at=data.get("created_at", ""),
            roi_bounds=tuple(data["roi_bounds"]),
            pixels_per_mm=float(data["scale"]["pixels_per_mm"]),
            mm_per_pixel=float(data["scale"]["mm_per_pixel"]),
            tag_size_mm=float(data["scale"]["tag_size_mm"]),
            target_width_mm=float(data["scale"].get("target_width_mm", 490.0)),
            target_height_mm=float(data["scale"].get("target_height_mm", 1220.0)),
            april_tags=april_tags,
            silhouette=silhouette,
            zones=zones,
            feature_regions=feature_regions,
            features=features
        )
        
        return blueprint, ref_image
