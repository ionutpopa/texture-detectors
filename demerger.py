import os
import json
import shutil
from tqdm import tqdm

# Load annotations
with open("fabric_clip_annotations.json") as f:
    annotations = json.load(f)

# Output base directory
output_base = "datasets_by_task"
os.makedirs(output_base, exist_ok=True)

# Define dataset paths
dataset_paths = {
    "fabric_class": os.path.join(output_base, "fabric_class"),
    "material_weight": os.path.join(output_base, "material_weight"),
    "finish": os.path.join(output_base, "finish")
}

# Create folders
for path in dataset_paths.values():
    os.makedirs(path, exist_ok=True)

# Copy images to proper folders
for ann in tqdm(annotations):
    image_path = ann["image_path"]

    # Each task's label
    labels = {
        "fabric_class": ann["fabric_class"],
        "material_weight": ann["material_weight"],
        "finish": ann["finish"]
    }

    for task, label in labels.items():
        target_dir = os.path.join(dataset_paths[task], label)
        os.makedirs(target_dir, exist_ok=True)

        target_path = os.path.join(target_dir, os.path.basename(image_path))
        if not os.path.exists(target_path):
            try:
                shutil.copy(image_path, target_path)
            except Exception as e:
                print(f"Failed to copy {image_path}: {e}")
