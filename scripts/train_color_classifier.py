#!/usr/bin/env python3
"""
Color Classifier Training Script
Trains MobileNetV3 on vehicle color dataset.

Datasets:
- UFPR-VCR: https://web.inf.ufpr.br/vri/databases/ufpr-vcr/
- VCoR: Vehicle Color Recognition Dataset

Usage:
    pip install torch torchvision timm albumentations
    python train_color_classifier.py --data_dir /path/to/dataset --epochs 50
"""
import os
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
from PIL import Image
import json

COLOR_CLASSES = [
    "black", "white", "gray", "silver",
    "red", "blue", "green", "yellow",
    "orange", "brown", "beige", "unknown"
]


class VehicleColorDataset(Dataset):
    """
    Expected dataset structure:
    data_dir/
      train/
        black/img1.jpg ...
        white/img1.jpg ...
        ...
      val/
        black/...
        ...
    """
    def __init__(self, root: str, split: str = "train", transform=None):
        self.samples = []
        self.transform = transform
        split_dir = os.path.join(root, split)
        for label_idx, cls in enumerate(COLOR_CLASSES):
            cls_dir = os.path.join(split_dir, cls)
            if not os.path.isdir(cls_dir):
                continue
            for fname in os.listdir(cls_dir):
                if fname.lower().endswith((".jpg", ".jpeg", ".png")):
                    self.samples.append((os.path.join(cls_dir, fname), label_idx))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label


def train(data_dir: str, output_dir: str, epochs: int = 50, batch_size: int = 32):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on: {device}")

    # Transforms with illumination augmentation
    train_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.05),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    train_ds = VehicleColorDataset(data_dir, "train", train_transform)
    val_ds = VehicleColorDataset(data_dir, "val", val_transform)
    print(f"Train: {len(train_ds)} | Val: {len(val_ds)}")

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=4)

    # MobileNetV3 with custom head
    model = models.mobilenet_v3_small(pretrained=True)
    model.classifier[3] = nn.Linear(model.classifier[3].in_features, len(COLOR_CLASSES))
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_acc = 0.0
    for epoch in range(epochs):
        # Train
        model.train()
        train_loss = 0.0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            out = model(imgs)
            loss = criterion(out, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        scheduler.step()

        # Validate
        model.eval()
        correct = total = 0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                out = model(imgs)
                _, pred = out.max(1)
                correct += (pred == labels).sum().item()
                total += labels.size(0)

        acc = correct / total * 100
        print(f"Epoch {epoch+1}/{epochs} | Loss: {train_loss/len(train_loader):.4f} | Val Acc: {acc:.2f}%")

        if acc > best_acc:
            best_acc = acc
            os.makedirs(output_dir, exist_ok=True)
            # Export to ONNX
            dummy = torch.randn(1, 3, 224, 224).to(device)
            onnx_path = os.path.join(output_dir, "mobilenetv3_color.onnx")
            torch.onnx.export(
                model, dummy, onnx_path,
                input_names=["input"], output_names=["output"],
                dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
                opset_version=11,
            )
            print(f"  ✅ Best model saved: {onnx_path} (acc={acc:.2f}%)")

    print(f"\nTraining complete. Best accuracy: {best_acc:.2f}%")
    print(f"Model: {os.path.join(output_dir, 'mobilenetv3_color.onnx')}")
    print(f"Copy to: backend/models/color_classifier/mobilenetv3_color.onnx")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True, help="Path to dataset with train/val splits")
    parser.add_argument("--output_dir", default="./trained_models", help="Output directory")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=32)
    args = parser.parse_args()
    train(args.data_dir, args.output_dir, args.epochs, args.batch_size)
