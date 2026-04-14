import os
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from torch.amp import autocast
from sklearn.metrics import f1_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

from dataset import CholecTrack20Dataset
from model import build_model
from PIL import Image, ImageFile

# ============================================================
# POSTAVKE
# ============================================================
DATASET_ROOT   = r"way to the folder with cholectrack20 dataset"
TEST_CSV       = os.path.join(DATASET_ROOT, "test_sequences.csv")
CHECKPOINT_DIR = os.path.join(DATASET_ROOT, "checkpoints")
MODEL_PATH     = os.path.join(CHECKPOINT_DIR, "best_cholectrack20_finetuned.pth")

NUM_CLASSES = 7
BATCH_SIZE  = 4

PHASE_NAMES = [
    "Preparation",
    "CalotTriangleDissection",
    "ClippingCutting",
    "GallbladderDissection",
    "GallbladderPackaging",
    "CleaningCoagulation",
    "GallbladderRetraction"
]

# ============================================================
# EVALUACIJA
# ============================================================
def evaluate(model, loader, device):
    model.eval()
    all_preds  = []
    all_labels = []

    with torch.no_grad():
        for clips, labels in loader:
            clips  = clips.to(device)
            labels = labels.to(device)

            with autocast('cuda'):
                outputs = model(clips)

            _, predicted = outputs.max(1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    return np.array(all_labels), np.array(all_preds)

# ============================================================
# GLAVNI DIO
# ============================================================
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Koristi se: {device}\n")

    # Dataset
    test_dataset = CholecTrack20Dataset(TEST_CSV, mode="val")
    test_loader  = DataLoader(
        test_dataset, batch_size=BATCH_SIZE,
        shuffle=False, num_workers=8, pin_memory=True
    )
    print(f"[INFO] Test sekvenci: {len(test_dataset)}\n")

    # Model
    model, in_features = build_model(num_classes=NUM_CLASSES, pretrained=False)
    model.head = nn.Sequential(
        nn.Dropout(p=0.5),
        nn.Linear(in_features, NUM_CLASSES)
    )
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))
    model = model.to(device)
    print(f"[INFO] Model učitan: {MODEL_PATH}\n")

    # Evaluacija
    labels, preds = evaluate(model, test_loader, device)

    # Metrike
    accuracy = 100.0 * (preds == labels).sum() / len(labels)
    f1_macro = f1_score(labels, preds, average='macro', zero_division=0)
    f1_per_class = f1_score(labels, preds, average=None, zero_division=0)

    print("=" * 60)
    print("REZULTATI NA TESTING SKUPU")
    print("=" * 60)
    print(f"\nFrame Accuracy: {accuracy:.2f}%")
    print(f"F1 Score (macro): {f1_macro:.4f}")

    print("\nF1 Score po fazama:")
    for i, name in enumerate(PHASE_NAMES):
        print(f"  Phase {i} ({name}): {f1_per_class[i]:.4f}")

    print("\nDetaljni izvještaj:")
    print(classification_report(labels, preds, target_names=PHASE_NAMES, zero_division=0))

    # Confusion Matrix
    cm = confusion_matrix(labels, preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=PHASE_NAMES,
                yticklabels=PHASE_NAMES)
    plt.title('Confusion Matrix - Testing skup')
    plt.ylabel('Stvarna faza')
    plt.xlabel('Predviđena faza')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(DATASET_ROOT, "confusion_matrix.png"))
    plt.show()
    print(f"\n[INFO] Confusion matrix spremljena!")
    print("\n" + "=" * 60)
print("EVALUACIJA ZAVRSENA")
print("=" * 60)

if __name__ == "__main__":
    main()