import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms, datasets
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import shutil

def clean_hidden_folders(dataset_dir):
    for root, dirs, files in os.walk(dataset_dir):
        for d in dirs:
            if d.startswith('.'):
                full_path = os.path.join(root, d)
                print(f"Removing {full_path}")
                shutil.rmtree(full_path)


def get_dataloaders(dataset_dir, batch_size=32, input_size=224):
    transform = transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
    ])
    clean_hidden_folders(dataset_dir)
    full_ds = datasets.ImageFolder(root=dataset_dir, transform=transform)
    class_names = full_ds.classes

    val_size = int(0.2 * len(full_ds))
    train_size = len(full_ds) - val_size
    train_ds, val_ds = random_split(full_ds, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, len(class_names), class_names

class ResNetMultiClassifier(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.backbone = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(in_features, num_classes)
        )

    def forward(self, x):
        return self.backbone(x)

def train(model, loader, device, epochs=5):
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for inputs, labels in tqdm(loader, desc=f"Epoch {epoch+1}"):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}, Loss: {total_loss / len(loader):.4f}")

def evaluate(model, val_loader, device, class_names, task):
    model.eval()
    y_true, y_pred = [], []

    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            y_pred.extend(preds)
            y_true.extend(labels.numpy())

    # Save classification report
    report = classification_report(y_true, y_pred, target_names=class_names, digits=3)
    with open(f"{task}_report.txt", "w") as f:
        f.write(report)
    print(f"✅ Classification report saved to {task}_report.txt")

    # Save confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.title(f"Confusion Matrix for {task}")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(f"{task}_confusion_matrix.png")
    print(f"✅ Confusion matrix saved to {task}_confusion_matrix.png")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=["fabric_class", "material_weight", "finish"], required=True)
    parser.add_argument("--dataset_base", default="datasets_by_task")
    parser.add_argument("--epochs", type=int, default=5)
    args = parser.parse_args()

    dataset_dir = os.path.join(args.dataset_base, args.task)
    if not os.path.exists(dataset_dir):
        raise ValueError(f"Dataset not found: {dataset_dir}")

    print(f"📦 Loading dataset for: {args.task}")
    train_loader, val_loader, num_classes, class_names = get_dataloaders(dataset_dir)

    print(f"🧠 Training ResNet50 for {args.task} with {num_classes} classes")
    model = ResNetMultiClassifier(num_classes)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train(model, train_loader, device, epochs=args.epochs)

    print(f"🧪 Evaluating model on validation set...")
    evaluate(model, val_loader, device, class_names, args.task)

    torch.save(model.state_dict(), f"{args.task}_resnet50.pth")
    print(f"✅ Model saved to {args.task}_resnet50.pth")

if __name__ == "__main__":
    main()
