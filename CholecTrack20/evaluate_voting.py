import os
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from torch.amp import autocast
from sklearn.metrics import f1_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict

from dataset import CholecTrack20Dataset
from model import build_model
import pandas as pd

# ============================================================
# POSTAVKE
# ============================================================
DATASET_ROOT   = r"way to the folder with cholectrack20 dataset"
TEST_CSV       = os.path.join(DATASET_ROOT, "test_sequences.csv")
TEST_PAIRS_CSV = os.path.join(DATASET_ROOT, "test_frame_phase_pairs.csv")
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
# EVALUACIJA S VOTINGOM
# ============================================================
def evaluate_with_voting(model, loader, test_seq_df, device):
    model.eval()

    # Prikupi predikcije po sekvenci
    all_seq_preds  = []
    all_seq_labels = []

    with torch.no_grad():
        for clips, labels in loader:
            clips  = clips.to(device)
            labels = labels.to(device)

            with autocast('cuda'):
                outputs = model(clips)

            _, predicted = outputs.max(1)
            all_seq_preds.extend(predicted.cpu().numpy())
            all_seq_labels.extend(labels.cpu().numpy())

    # Voting po frameu
    # Za svaki frame skupi sve predikcije sekvenci koje ga sadrže
    frame_votes = defaultdict(lambda: defaultdict(int))
    frame_true  = {}

    for idx, (pred, true_label) in enumerate(zip(all_seq_preds, all_seq_labels)):
        row = test_seq_df.iloc[idx]
        case = row["case"]
        frame_ids = [int(f) for f in row["frame_ids"].split("|")] \
            if "frame_ids" in row else []

        # Dodaj glas za svaki frame u sekvenci
        for frame_id in frame_ids:
            key = f"{case}_{frame_id}"
            frame_votes[key][pred] += 1
            frame_true[key] = true_label

    # Uzmi najčešću predikciju po frameu
    voted_preds  = []
    voted_labels = []

    for key, votes in frame_votes.items():
        best_pred = max(votes, key=votes.get)
        voted_preds.append(best_pred)
        voted_labels.append(frame_true[key])

    return np.array(voted_labels), np.array(voted_preds)


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
    test_seq_df = pd.read_csv(TEST_CSV)
    print(f"[INFO] Test sekvenci: {len(test_dataset)}\n")

    # Model
    model, in_features = build_model(num_classes=NUM_CLASSES, pretrained=False)
    model.head = nn.Sequential(
        nn.Dropout(p=0.5),
        nn.Linear(in_features, NUM_CLASSES)
    )
    model.load_state_dict(torch.load(
        MODEL_PATH, map_location=device, weights_only=True
    ))
    model = model.to(device)
    print(f"[INFO] Model učitan: {MODEL_PATH}\n")

    # Evaluacija s votingom
    labels, preds = evaluate_with_voting(
        model, test_loader, test_seq_df, device
    )

    # Metrike
    accuracy = 100.0 * (preds == labels).sum() / len(labels)
    f1_macro = f1_score(labels, preds, average='macro', zero_division=0)
    f1_per_class = f1_score(labels, preds, average=None, zero_division=0)

    print("=" * 60)
    print("REZULTATI S MAJORITY VOTING")
    print("=" * 60)
    print(f"\nFrame Accuracy: {accuracy:.2f}%")
    print(f"F1 Score (macro): {f1_macro:.4f}")

    print("\nF1 Score po fazama:")
    for i, name in enumerate(PHASE_NAMES):
        print(f"  Phase {i} ({name}): {f1_per_class[i]:.4f}")

    print("\nDetaljni izvještaj:")
    print(classification_report(
        labels, preds,
        target_names=PHASE_NAMES,
        zero_division=0
    ))

    # Confusion Matrix
    cm = confusion_matrix(labels, preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=PHASE_NAMES,
                yticklabels=PHASE_NAMES)
    plt.title('Confusion Matrix - Majority Voting')
    plt.ylabel('Stvarna faza')
    plt.xlabel('Predviđena faza')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(DATASET_ROOT, "confusion_matrix_voting.png"))
    plt.show()

    print("\n" + "=" * 60)
    print("EVALUACIJA ZAVRŠENA")
    print("=" * 60)


if __name__ == "__main__":
    main()