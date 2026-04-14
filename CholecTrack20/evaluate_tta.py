import os
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch.amp import autocast
from sklearn.metrics import f1_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image, ImageFile
import pandas as pd
from torchvision import transforms

from model import build_model

# Dozvoli učitavanje truncated slika
ImageFile.LOAD_TRUNCATED_IMAGES = True

# ============================================================
# POSTAVKE
# ============================================================
DATASET_ROOT   = r"way to the folder with cholectrack20 dataset"
TEST_CSV       = os.path.join(DATASET_ROOT, "test_sequences.csv")
CHECKPOINT_DIR = os.path.join(DATASET_ROOT, "checkpoints")
MODEL_PATH     = os.path.join(CHECKPOINT_DIR, "best_cholectrack20_finetuned.pth")

NUM_CLASSES = 7
BATCH_SIZE  = 4
TTA_STEPS   = 5

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
# TTA TRANSFORMACIJE
# ============================================================
def get_tta_transforms():
    base = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    aug1 = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    aug2 = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomCrop((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    aug3 = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=1.0),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    aug4 = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomCrop((224, 224)),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    return [base, aug1, aug2, aug3, aug4]

# ============================================================
# ROBUSNO UCITAVANJE SLIKA
# ============================================================
def load_image_robust(path):
    candidates = []

    if os.path.exists(path):
        candidates.append(path)

    base_path = os.path.splitext(path)[0]
    for ext in [".png", ".jpg", ".jpeg"]:
        candidate = base_path + ext
        if candidate not in candidates and os.path.exists(candidate):
            candidates.append(candidate)

    if not candidates:
        raise FileNotFoundError(f"[ERROR] Slika nije pronađena: {path}")

    last_error = None
    for candidate in candidates:
        try:
            with Image.open(candidate) as img:
                img = img.convert("RGB")
            return img
        except Exception as e:
            last_error = e
            print(f"[ERROR] Ne mogu učitati sliku: {candidate}")
            print(f"[ERROR] Razlog: {e}")

    raise OSError(
        f"[ERROR] Nijedna varijanta slike nije uspješno učitana za path: {path}\n"
        f"Zadnja greška: {last_error}"
    )

# ============================================================
# TTA DATASET
# ============================================================
class TTADataset(Dataset):
    def __init__(self, csv_file, transform):
        self.data = pd.read_csv(csv_file)
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        image_paths = row["image_paths"].split("|")
        frames = []

        for path in image_paths:
            img = load_image_robust(path)
            img = self.transform(img)
            frames.append(img)

        clip = torch.stack(frames, dim=0)   # (T, C, H, W)
        clip = clip.permute(1, 0, 2, 3)     # (C, T, H, W)
        label = torch.tensor(int(row["label"]), dtype=torch.long)

        return clip, label

# ============================================================
# EVALUACIJA S TTA
# ============================================================
def evaluate_tta(model, csv_file, device):
    tta_transforms = get_tta_transforms()
    all_probs = None
    all_labels = None

    for t_idx, transform in enumerate(tta_transforms):
        print(f"[INFO] TTA korak {t_idx + 1}/{len(tta_transforms)}...")

        dataset = TTADataset(csv_file, transform)
        loader = DataLoader(
            dataset,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=8,   # stabilnije za debug i oštećene slike
            pin_memory=(device.type == "cuda")
        )

        step_probs = []
        step_labels = []

        model.eval()
        with torch.no_grad():
            for clips, labels in loader:
                clips = clips.to(device, non_blocking=True)

                if device.type == "cuda":
                    with autocast("cuda"):
                        outputs = model(clips)
                else:
                    outputs = model(clips)

                probs = torch.softmax(outputs, dim=1)
                step_probs.extend(probs.cpu().numpy())
                step_labels.extend(labels.numpy())

        step_probs = np.array(step_probs)

        if all_probs is None:
            all_probs = step_probs
            all_labels = np.array(step_labels)
        else:
            all_probs += step_probs

    all_probs /= len(tta_transforms)
    preds = all_probs.argmax(axis=1)

    return all_labels, preds

# ============================================================
# GLAVNI DIO
# ============================================================
def main():
    print("Ulazim u main...")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Koristi se: {device}\n")

    model, in_features = build_model(num_classes=NUM_CLASSES, pretrained=False)
    model.head = nn.Sequential(
        nn.Dropout(p=0.5),
        nn.Linear(in_features, NUM_CLASSES)
    )

    model.load_state_dict(torch.load(
        MODEL_PATH,
        map_location=device,
        weights_only=True
    ))
    model = model.to(device)

    print(f"[INFO] Model učitan: {MODEL_PATH}\n")
    print(f"[INFO] Pokrećem TTA evaluaciju ({TTA_STEPS} koraka)...\n")

    labels, preds = evaluate_tta(model, TEST_CSV, device)

    accuracy = 100.0 * (preds == labels).sum() / len(labels)
    f1_macro = f1_score(labels, preds, average="macro", zero_division=0)
    f1_per_class = f1_score(labels, preds, average=None, zero_division=0)

    print("\n" + "=" * 60)
    print("REZULTATI S TTA")
    print("=" * 60)
    print(f"\nFrame Accuracy: {accuracy:.2f}%")
    print(f"F1 Score (macro): {f1_macro:.4f}")

    print("\nF1 Score po fazama:")
    for i, name in enumerate(PHASE_NAMES):
        print(f"  Phase {i} ({name}): {f1_per_class[i]:.4f}")

    print("\nDetaljni izvještaj:")
    print(classification_report(
        labels,
        preds,
        target_names=PHASE_NAMES,
        zero_division=0
    ))

    cm = confusion_matrix(labels, preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=PHASE_NAMES,
        yticklabels=PHASE_NAMES
    )
    plt.title('Confusion Matrix - TTA')
    plt.ylabel('Stvarna faza')
    plt.xlabel('Predviđena faza')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()

    save_path = os.path.join(DATASET_ROOT, "confusion_matrix_tta.png")
    plt.savefig(save_path)
    plt.show()

    print(f"\n[INFO] Confusion matrix spremljena: {save_path}")
    print("\n" + "=" * 60)
    print("EVALUACIJA ZAVRSENA")
    print("=" * 60)

if __name__ == "__main__":
    main()