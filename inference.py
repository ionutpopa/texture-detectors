import argparse
import os
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import json

# === Model wrapper ===
class ResNetSingleClassifier(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        base = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        in_features = base.fc.in_features
        base.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(in_features, num_classes)
        )
        self.backbone = base

    def forward(self, x):
        return self.backbone(x)

# === Load label mappings ===
def load_class_names(task):
    label_path = f"{task}_label_map.json"
    if not os.path.exists(label_path):
        raise FileNotFoundError(f"Missing label file: {label_path}")
    with open(label_path) as f:
        return json.load(f)

# === Preprocessing ===
def load_and_preprocess_image(image_path):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    image = Image.open(image_path).convert("RGB")
    return transform(image).unsqueeze(0)

def predict(model_path, task, image_path):
    class_names = load_class_names(task)
    num_classes = len(class_names)

    model = ResNetSingleClassifier(num_classes)
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()

    image_tensor = load_and_preprocess_image(image_path)
    with torch.no_grad():
        outputs = model(image_tensor)
        predicted_idx = torch.argmax(outputs, dim=1).item()
        predicted_label = class_names[str(predicted_idx)]

    print(f"✅ Prediction for '{task}': {predicted_label}")
    return predicted_label

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Path to trained .pth model")
    parser.add_argument("--task", required=True, choices=["fabric_class", "material_weight", "finish"])
    parser.add_argument("--image", required=True, help="Path to image")
    args = parser.parse_args()

    predict(args.model, args.task, args.image)

if __name__ == "__main__":
    main()
