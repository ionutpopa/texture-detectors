import os
import json
from tqdm import tqdm
from PIL import Image
import torch
from transformers import CLIPProcessor, CLIPModel

device = "cuda" if torch.cuda.is_available() else "cpu"
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

dataset_path = "fabrics_dataset/fabric/train"  # adjust if needed

material_weights = ["lightweight", "medium weight", "heavyweight"]
finishes = ["matte", "shiny", "sheer", "textured", "smooth"]

annotations = []

for fabric_class in tqdm(os.listdir(dataset_path), desc="Processing classes"):
    class_path = os.path.join(dataset_path, fabric_class)
    if not os.path.isdir(class_path):
        continue

    image_files = [f for f in os.listdir(class_path) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.webp', '.JPG', '.JPEG', '.PNG'))]
    image_files = image_files[:1200]  # limit to 1200 images

    for image_name in image_files:
        print(f"Processing {image_name} in {fabric_class}")

        image_path = os.path.join(class_path, image_name)
        try:
            image = Image.open(image_path).convert("RGB")

            # Predict material weight
            weight_inputs = processor(
                text=[f"{mw} fabric" for mw in material_weights],
                images=image,
                return_tensors="pt",
                padding=True
            ).to(device)
            weight_outputs = model(**weight_inputs)
            weight_logits = weight_outputs.logits_per_image.softmax(dim=1)
            predicted_weight = material_weights[weight_logits.argmax().item()]

            # Predict finish
            finish_inputs = processor(
                text=[f"{fin} fabric" for fin in finishes],
                images=image,
                return_tensors="pt",
                padding=True
            ).to(device)
            finish_outputs = model(**finish_inputs)
            finish_logits = finish_outputs.logits_per_image.softmax(dim=1)
            predicted_finish = finishes[finish_logits.argmax().item()]

            # Save annotation
            annotations.append({
                "image_path": image_path,
                "fabric_class": fabric_class,
                "material_weight": predicted_weight,
                "finish": predicted_finish
            })

        except Exception as e:
            print(f"Error processing {image_path}: {e}")

output_file = "fabric_clip_annotations.json"
with open(output_file, "w") as f:
    json.dump(annotations, f, indent=2)

print(f"\n✅ Done! Annotations saved to: {output_file}")
