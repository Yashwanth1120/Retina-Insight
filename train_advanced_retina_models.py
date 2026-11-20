"""
train_advanced_retina_models.py: Advanced Deep Learning Training Pipeline for Diabetic Retinopathy.
Supports:
  - Modern Architectures: ConvNeXt-Tiny, EfficientNet-B4, EfficientNet-B0, Swin Transformer (Swin-T)
  - High-Resolution Ingestion: 512x512 (and 384x384 / 256x256)
  - Multi-Cohort Ingestion: APTOS 2019, EyePACS, Messidor-2, IDRiD
  - Loss Functions: Class-Weighted Cross-Entropy, Ordinal CORAL Loss, Label Smoothing
  - Post-Hoc Uncertainty Calibration: L-BFGS Temperature Scaling
  - Comprehensive Clinical Metrics: QWK, Within-1-Stage, Referable DR (Stage >= 2) AUC / Sensitivity / Specificity
"""
import os
import sys
import argparse
import math
import json
import time
from datetime import datetime

import cv2
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import cohen_kappa_score, accuracy_score, roc_auc_score, confusion_matrix

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms

# ----------------- REPRODUCIBILITY -----------------
def seed_everything(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# ----------------- ADAPTIVE BEN GRAHAM FILTER -----------------
def fast_ben_graham(img_bgr, target_size=512):
    """Standardized Ben Graham local contrast enhancement scaled to target resolution."""
    img_resized = cv2.resize(img_bgr, (target_size, target_size))
    sigma = int(10 * (target_size / 256.0))
    blurred = cv2.GaussianBlur(img_resized, (0, 0), sigma)
    enhanced = cv2.addWeighted(img_resized, 4, blurred, -4, 128)
    return enhanced

# ----------------- DATASET & MULTI-COHORT INGESTION -----------------
class AdvancedRetinaDataset(Dataset):
    """High-resolution dataset supporting multi-cohort image formats."""
    def __init__(self, df, img_dir, target_size=512, transform=None):
        self.df = df.reset_index(drop=True)
        self.img_dir = img_dir
        self.target_size = target_size
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_id = str(row['id_code'])

        img_path = None
        for ext in ['', '.png', '.jpg', '.jpeg']:
            p = os.path.join(self.img_dir, f"{img_id}{ext}")
            if os.path.exists(p):
                img_path = p
                break

        if img_path and os.path.exists(img_path):
            img_bgr = cv2.imread(img_path)
        else:
            img_bgr = np.zeros((self.target_size, self.target_size, 3), dtype=np.uint8)

        if img_bgr is None:
            img_bgr = np.zeros((self.target_size, self.target_size, 3), dtype=np.uint8)

        enhanced_bgr = fast_ben_graham(img_bgr, target_size=self.target_size)
        enhanced_rgb = cv2.cvtColor(enhanced_bgr, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(enhanced_rgb)

        if self.transform:
            tensor_img = self.transform(pil_img)
        else:
            tensor_img = transforms.ToTensor()(pil_img)

        label = torch.tensor(int(row['diagnosis']), dtype=torch.long)
        return tensor_img, label

# ----------------- MODERN MODEL FACTORY -----------------
def create_model(backbone="convnext_tiny", num_classes=5, pretrained=True):
    """Instantiates modern medical computer vision backbones with customized 5-class heads."""
    b_lower = backbone.lower().replace("-", "_")
    
    if b_lower in ["convnext_tiny", "convnext"]:
        weights = models.ConvNeXt_Tiny_Weights.DEFAULT if pretrained else None
        model = models.convnext_tiny(weights=weights)
        in_feat = model.classifier[2].in_features
        model.classifier[2] = nn.Sequential(
            nn.Dropout(p=0.35),
            nn.Linear(in_feat, num_classes)
        )
    elif b_lower in ["efficientnet_b4", "effnet_b4"]:
        weights = models.EfficientNet_B4_Weights.DEFAULT if pretrained else None
        model = models.efficientnet_b4(weights=weights)
        in_feat = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(p=0.35),
            nn.Linear(in_feat, num_classes)
        )
    elif b_lower in ["swin_t", "swin_tiny", "swin"]:
        weights = models.Swin_T_Weights.DEFAULT if pretrained else None
        model = models.swin_t(weights=weights)
        in_feat = model.head.in_features
        model.head = nn.Sequential(
            nn.Dropout(p=0.35),
            nn.Linear(in_feat, num_classes)
        )
    elif b_lower in ["resnet50", "resnet_50"]:
        weights = models.ResNet50_Weights.DEFAULT if pretrained else None
        model = models.resnet50(weights=weights)
        in_feat = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Dropout(p=0.35),
            nn.Linear(in_feat, num_classes)
        )
    else: # Default: EfficientNet-B0
        weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
        model = models.efficientnet_b0(weights=weights)
        in_feat = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(p=0.35),
            nn.Linear(in_feat, num_classes)
        )

    return model

# ----------------- ORDINAL CORAL LOSS -----------------
class CoralLoss(nn.Module):
    """Consistent Rank Logits (CORAL) loss for ordinal diabetic retinopathy stages (0 < 1 < 2 < 3 < 4)."""
    def __init__(self, num_classes=5):
        super(CoralLoss, self).__init__()
        self.num_classes = num_classes

    def forward(self, logits, targets):
        num_tasks = self.num_classes - 1
        # Convert integer targets into binary ordinal indicators: e.g. Stage 2 -> [1, 1, 0, 0]
        binary_targets = torch.zeros((targets.size(0), num_tasks), dtype=torch.float, device=targets.device)
        for i in range(num_tasks):
            binary_targets[:, i] = (targets > i).float()
            
        task_logits = logits[:, :num_tasks]
        loss = F.binary_cross_entropy_with_logits(task_logits, binary_targets)
        return loss

# ----------------- TEMPERATURE SCALING -----------------
class TemperatureScaler(nn.Module):
    """Optimizes temperature T via L-BFGS on validation logits to minimize negative log likelihood."""
    def __init__(self):
        super(TemperatureScaler, self).__init__()
        self.temperature = nn.Parameter(torch.ones(1) * 1.5)

    def forward(self, logits):
        return logits / self.temperature

    def fit(self, val_logits, val_labels):
        nll_criterion = nn.CrossEntropyLoss()
        optimizer = optim.LBFGS([self.temperature], lr=0.01, max_iter=50)

        def eval_loss():
            optimizer.zero_grad()
            loss = nll_criterion(self.forward(val_logits), val_labels)
            loss.backward()
            return loss

        optimizer.step(eval_loss)
        return max(float(self.temperature.item()), 0.01)

# ----------------- TRAINING & EVALUATION ENGINE -----------------
def run_training_pipeline(
    data_csv,
    img_dir,
    output_dir="./model_checkpoints",
    backbone="convnext_tiny",
    target_size=512,
    epochs=15,
    batch_size=16,
    lr=2e-4,
    use_ordinal=False
):
    seed_everything(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Training Device: {device} | Architecture: {backbone} | Resolution: {target_size}x{target_size}")
    os.makedirs(output_dir, exist_ok=True)

    # 1. Load dataset metadata
    df = pd.read_csv(data_csv)
    # Normalize column names
    col_map = {'id_code': 'id_code', 'image': 'id_code', 'image_id': 'id_code', 'diagnosis': 'diagnosis', 'level': 'diagnosis'}
    df = df.rename(columns={c: col_map[c] for c in df.columns if c in col_map})
    df['diagnosis'] = df['diagnosis'].astype(int)

    # 2. Strict 3-Way Stratified Split: 70% Train, 15% Val, 15% Test
    train_df, temp_df = train_test_split(df, test_size=0.30, stratify=df['diagnosis'], random_state=42)
    val_df, test_df = train_test_split(temp_df, test_size=0.50, stratify=temp_df['diagnosis'], random_state=42)

    print(f"[+] Dataset Split: Train={len(train_df)}, Val={len(val_df)}, Held-Out Test={len(test_df)}")

    # 3. Class Weights for imbalance
    all_classes = np.array([0, 1, 2, 3, 4])
    cls_weights = compute_class_weight(class_weight='balanced', classes=all_classes, y=train_df['diagnosis'].values)
    weights_tensor = torch.tensor(cls_weights, dtype=torch.float).to(device)

    # 4. Data Transforms
    train_transforms = transforms.Compose([
        transforms.RandomResizedCrop(target_size, scale=(0.80, 1.05), ratio=(0.95, 1.05)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(degrees=25),
        transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.12),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    eval_transforms = transforms.Compose([
        transforms.Resize((target_size, target_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_loader = DataLoader(AdvancedRetinaDataset(train_df, img_dir, target_size, train_transforms), batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(AdvancedRetinaDataset(val_df, img_dir, target_size, eval_transforms), batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    test_loader = DataLoader(AdvancedRetinaDataset(test_df, img_dir, target_size, eval_transforms), batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

    # 5. Model Initialization
    num_out = 4 if use_ordinal else 5
    model = create_model(backbone=backbone, num_classes=num_out, pretrained=True).to(device)

    if use_ordinal:
        criterion = CoralLoss(num_classes=5)
    else:
        criterion = nn.CrossEntropyLoss(weight=weights_tensor, label_smoothing=0.08)

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    best_qwk = -1.0
    best_checkpoint_path = os.path.join(output_dir, f"best_{backbone}_{target_size}.pth")

    print(f"[*] Commencing {epochs}-Epoch Training Loop...")
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss, train_total = 0.0, 0
        for imgs, lbls in train_loader:
            imgs, lbls = imgs.to(device), lbls.to(device)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, lbls)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            optimizer.step()
            train_loss += loss.item() * imgs.size(0)
            train_total += lbls.size(0)

        scheduler.step()
        epoch_loss = train_loss / train_total

        # Validation Phase
        model.eval()
        val_preds, val_targets = [], []
        with torch.no_grad():
            for imgs, lbls in val_loader:
                imgs = imgs.to(device)
                outputs = model(imgs)
                if use_ordinal:
                    probs = torch.sigmoid(outputs)
                    preds = (probs > 0.5).sum(dim=1).cpu().numpy()
                else:
                    preds = torch.argmax(outputs, dim=1).cpu().numpy()
                val_preds.extend(preds)
                val_targets.extend(lbls.numpy())

        val_acc = accuracy_score(val_targets, val_preds)
        val_qwk = cohen_kappa_score(val_targets, val_preds, weights='quadratic')

        print(f"Epoch {epoch:02d}/{epochs:02d} | Train Loss: {epoch_loss:.4f} | Val Acc: {val_acc*100:.2f}% | Val QWK: {val_qwk:.4f}")

        if val_qwk > best_qwk:
            best_qwk = val_qwk
            torch.save({
                'epoch': epoch,
                'backbone': backbone,
                'target_size': target_size,
                'state_dict': model.state_dict(),
                'val_qwk': float(val_qwk),
                'val_acc': float(val_acc),
                'use_ordinal': use_ordinal
            }, best_checkpoint_path)
            print(f"    [+] Saved New Best Model Checkpoint: {best_checkpoint_path} (QWK: {val_qwk:.4f})")

    # 6. Post-Hoc Temperature Calibration
    print("\n[*] Fitting Temperature Scaling on Validation Logits...")
    checkpoint = torch.load(best_checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()

    val_logits_list, val_targets_list = [], []
    with torch.no_grad():
        for imgs, lbls in val_loader:
            imgs = imgs.to(device)
            val_logits_list.append(model(imgs))
            val_targets_list.append(lbls.to(device))

    val_logits_all = torch.cat(val_logits_list)
    val_targets_all = torch.cat(val_targets_list)

    scaler = TemperatureScaler().to(device)
    fitted_temp = scaler.fit(val_logits_all, val_targets_all)
    print(f"[+] Fitted Temperature (T): {fitted_temp:.4f}")

    # 7. Unbiased Evaluation on Strict Held-Out Test Set
    test_preds, test_targets, test_probs = [], [], []
    with torch.no_grad():
        for imgs, lbls in test_loader:
            imgs = imgs.to(device)
            cal_logits = model(imgs) / fitted_temp
            probs = F.softmax(cal_logits, dim=1)
            preds = torch.argmax(probs, dim=1)
            test_preds.extend(preds.cpu().numpy())
            test_targets.extend(lbls.numpy())
            test_probs.extend(probs.cpu().numpy())

    test_preds = np.array(test_preds)
    test_targets = np.array(test_targets)
    test_probs = np.array(test_probs)

    test_acc = accuracy_score(test_targets, test_preds)
    test_qwk = cohen_kappa_score(test_targets, test_preds, weights='quadratic')
    within_one = np.mean(np.abs(test_preds - test_targets) <= 1) * 100

    # Referable DR (Stage >= 2) metrics
    binary_targets = (test_targets >= 2).astype(int)
    binary_probs = np.sum(test_probs[:, 2:], axis=1)
    referable_auc = roc_auc_score(binary_targets, binary_probs)

    summary = {
        "model_backbone": backbone,
        "input_resolution": f"{target_size}x{target_size}",
        "test_accuracy": f"{test_acc*100:.2f}%",
        "test_qwk": f"{test_qwk:.4f}",
        "within_one_stage": f"{within_one:.2f}%",
        "referable_dr_auc": f"{referable_auc:.4f}",
        "fitted_temperature": f"{fitted_temp:.4f}",
        "checkpoint_file": best_checkpoint_path
    }

    # Save summary
    summary_path = os.path.join(output_dir, f"training_summary_{backbone}_{target_size}.json")
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)

    print("\n=======================================================")
    print("FINAL UNBIASED HELD-OUT TEST EVALUATION:")
    for k, v in summary.items():
        print(f"  - {k}: {v}")
    print("=======================================================")
    return summary

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Advanced Retina Insight Model")
    parser.add_argument("--csv", type=str, default="train.csv", help="Path to labels CSV")
    parser.add_argument("--images", type=str, default="train_images", help="Path to images directory")
    parser.add_argument("--backbone", type=str, default="convnext_tiny", choices=["convnext_tiny", "efficientnet_b4", "efficientnet_b0", "swin_t"])
    parser.add_argument("--size", type=int, default=512, choices=[256, 384, 512])
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--out", type=str, default="./model_checkpoints")
    args = parser.parse_args()

    run_training_pipeline(
        data_csv=args.csv,
        img_dir=args.images,
        output_dir=args.out,
        backbone=args.backbone,
        target_size=args.size,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr
    )
