import os
from collections import Counter

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler
from tqdm import tqdm
from sklearn.metrics import f1_score

from dataset import CholecTrack20Dataset
from model import build_model


# ============================================================
# SETTINGS
# ============================================================
DATASET_ROOT = r"way to the folder with cholec80 dataset"

TRAIN_CSV = os.path.join(DATASET_ROOT, "pretrain_train_sequences.csv")
VAL_CSV   = os.path.join(DATASET_ROOT, "pretrain_val_sequences.csv")

CHECKPOINT_DIR = os.path.join(DATASET_ROOT, "checkpoints")
BEST_MODEL_PATH = os.path.join(CHECKPOINT_DIR, "best_cholec80_pretrain.pth")

NUM_CLASSES = 7
BATCH_SIZE = 4
EPOCHS = 15
LR = 1e-4
NUM_WORKERS = 4

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================
# HELPERS
# ============================================================

def build_class_weights(label_counts, num_classes=7):
    total = sum(label_counts.values())
    weights = []

    for cls in range(num_classes):
        count = label_counts.get(cls, 1)
        weights.append(total / count)

    weights = torch.tensor(weights, dtype=torch.float32)
    weights = weights / weights.mean()
    return weights


def run_epoch(model, loader, criterion, optimizer, scaler, device):
    model.train()
    total_loss = 0.0
    all_preds = []
    all_labels = []

    progress_bar = tqdm(
        total=len(loader),
        desc="Train",
        dynamic_ncols=True,
        mininterval=1.0,
        leave=False
    )

    for batch_idx, (clips, labels) in enumerate(loader, start=1):
        clips = clips.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad()

        if device.type == "cuda":
            with autocast("cuda"):
                outputs = model(clips)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(clips)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        total_loss += loss.item()

        preds = outputs.argmax(dim=1)
        all_preds.extend(preds.detach().cpu().numpy())
        all_labels.extend(labels.detach().cpu().numpy())

        progress_bar.update(1)

        # update displayed loss every 10 batches or at the end
        if batch_idx % 10 == 0 or batch_idx == len(loader):
            avg_loss_so_far = total_loss / batch_idx
            progress_bar.set_postfix(loss=f"{avg_loss_so_far:.4f}")

    progress_bar.close()

    avg_loss = total_loss / len(loader)
    f1_macro = f1_score(all_labels, all_preds, average="macro", zero_division=0)

    return avg_loss, f1_macro


@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_labels = []

    progress_bar = tqdm(
        total=len(loader),
        desc="Val  ",
        dynamic_ncols=True,
        mininterval=1.0,
        leave=False
    )

    for batch_idx, (clips, labels) in enumerate(loader, start=1):
        clips = clips.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        if device.type == "cuda":
            with autocast("cuda"):
                outputs = model(clips)
                loss = criterion(outputs, labels)
        else:
            outputs = model(clips)
            loss = criterion(outputs, labels)

        total_loss += loss.item()

        preds = outputs.argmax(dim=1)
        all_preds.extend(preds.detach().cpu().numpy())
        all_labels.extend(labels.detach().cpu().numpy())

        progress_bar.update(1)

        if batch_idx % 10 == 0 or batch_idx == len(loader):
            avg_loss_so_far = total_loss / batch_idx
            progress_bar.set_postfix(loss=f"{avg_loss_so_far:.4f}")

    progress_bar.close()

    avg_loss = total_loss / len(loader)
    f1_macro = f1_score(all_labels, all_preds, average="macro", zero_division=0)

    return avg_loss, f1_macro


# ============================================================
# MAIN
# ============================================================
def main():
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    print(f"[INFO] Device: {DEVICE}")

    train_dataset = CholecTrack20Dataset(TRAIN_CSV, mode="train")
    val_dataset   = CholecTrack20Dataset(VAL_CSV, mode="val")

    train_df = pd.read_csv(TRAIN_CSV)
    label_counts = Counter(train_df["label"].tolist())
    class_weights = build_class_weights(label_counts, num_classes=NUM_CLASSES).to(DEVICE)

    print(f"[INFO] Train sequences: {len(train_dataset)}")
    print(f"[INFO] Val sequences:   {len(val_dataset)}")
    print(f"[INFO] Label counts:    {label_counts}")
    print(f"[INFO] Class weights:   {class_weights.detach().cpu().tolist()}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=(DEVICE.type == "cuda"),
        persistent_workers=(NUM_WORKERS > 0)
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=(DEVICE.type == "cuda"),
        persistent_workers=(NUM_WORKERS > 0)
    )

    model, in_features = build_model(num_classes=NUM_CLASSES, pretrained=True)
    model.head = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(in_features, NUM_CLASSES)
    )
    model = model.to(DEVICE)

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    scaler = GradScaler(enabled=(DEVICE.type == "cuda"))
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=2
    )

    best_val_f1 = -1.0

    for epoch in range(EPOCHS):
        print(f"\n[INFO] Epoch {epoch + 1}/{EPOCHS}")

        train_loss, train_f1 = run_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            scaler=scaler,
            device=DEVICE
        )

        val_loss, val_f1 = validate(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=DEVICE
        )
        scheduler.step(val_f1)
        print(
            f"[INFO] train_loss={train_loss:.4f} | train_f1={train_f1:.4f} | "
            f"val_loss={val_loss:.4f} | val_f1={val_f1:.4f}"
        )

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save(model.state_dict(), BEST_MODEL_PATH)
            print(f"[INFO] New best model saved: {BEST_MODEL_PATH}")

    print("\n[INFO] Training finished.")
    print(f"[INFO] Best val_f1: {best_val_f1:.4f}")


if __name__ == "__main__":
    main()