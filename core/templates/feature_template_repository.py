import json
from pathlib import Path
import numpy as np
from core.models import FeatureRegion, FeatureRegionTemplate

class FeatureTemplateRepository:
    """
    Manages loading and saving FeatureRegionTemplates to/from JSON.
    """
    def __init__(self, templates_dir: str | Path) -> None:
        self.templates_dir = Path(templates_dir)
        self.templates_dir.mkdir(parents=True, exist_ok=True)

    def save_template(self, template: FeatureRegionTemplate) -> None:
        """
        Saves a template to a JSON file.
        """
        filepath = self.templates_dir / f"{template.target_type}_v{template.version}.json"
        
        # Serialize regions
        serialized_regions = []
        for r in template.regions:
            # Convert polygon numpy array to nested list
            poly_list = r.polygon.tolist()
            serialized_regions.append({
                "id": r.id,
                "polygon": poly_list,
                "region_type": r.region_type,
                "priority": r.priority,
                "min_features": r.min_features,
                "max_features": r.max_features,
                "metadata": r.metadata
            })
            
        data = {
            "template_id": template.template_id,
            "target_type": template.target_type,
            "version": template.version,
            "regions": serialized_regions,
            "metadata": template.metadata
        }
        
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4)

    def load_template(self, target_type: str, version: int = 1) -> FeatureRegionTemplate | None:
        """
        Loads a template from the repository if it exists.
        """
        filepath = self.templates_dir / f"{target_type}_v{version}.json"
        if not filepath.exists():
            return None
            
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
            
        regions = []
        for r_data in data["regions"]:
            regions.append(FeatureRegion(
                id=r_data["id"],
                polygon=np.array(r_data["polygon"], dtype=np.float32), # Stored in normalized coords!
                region_type=r_data["region_type"],
                priority=r_data["priority"],
                min_features=r_data["min_features"],
                max_features=r_data["max_features"],
                metadata=r_data.get("metadata", {})
            ))
            
        return FeatureRegionTemplate(
            template_id=data["template_id"],
            target_type=data["target_type"],
            version=data["version"],
            regions=regions,
            metadata=data.get("metadata", {})
        )

    def list_available_templates(self) -> list[dict]:
        """
        Lists metadata of all available templates in the repository.
        """
        templates = []
        for file in self.templates_dir.glob("*.json"):
            try:
                with open(file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    templates.append({
                        "target_type": data["target_type"],
                        "version": data["version"],
                        "template_id": data["template_id"],
                        "filename": file.name
                    })
            except Exception as e:
                print(f"Error loading template header from {file}: {e}")
        return templates
