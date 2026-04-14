import os
from collections import Counter

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler
from sklearn.metrics import f1_score
from tqdm import tqdm

from dataset import CholecTrack20Dataset
from model import build_model


# ============================================================
# CONFIG
# ============================================================
DATASET_ROOT = r"way to the folder with cholectrack20 dataset"

TRAIN_CSV = os.path.join(DATASET_ROOT, "train_sequences.csv")
VAL_CSV   = os.path.join(DATASET_ROOT, "val_sequences.csv")
CHECKPOINT_DIR = os.path.join(DATASET_ROOT, "checkpoints")

# Cholec80 pretrained weights
PRETRAIN_WEIGHTS_PATH = r"way to the folder with best pretrain model from checkpoints folder within project folder with cholec 80 dataset"

# Save paths for CholecTrack20 fine-tuning
BEST_MODEL_PATH = os.path.join(CHECKPOINT_DIR, "best_cholectrack20_finetuned.pth")
FULL_CHECKPOINT_PATH = os.path.join(CHECKPOINT_DIR, "best_cholectrack20_finetuned_full.pth")

NUM_CLASSES = 7
BATCH_SIZE = 16
NUM_WORKERS = 8

# Training schedule
HEAD_ONLY_EPOCHS = 15
PARTIAL_UNFREEZE_EPOCHS = 0
TOTAL_EPOCHS = HEAD_ONLY_EPOCHS + PARTIAL_UNFREEZE_EPOCHS

# Learning rates
HEAD_LR = 5e-6
BACKBONE_LR = 3e-6
WEIGHT_DECAY = 1e-2

PATIENCE = 6

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

os.makedirs(CHECKPOINT_DIR, exist_ok=True)


# ============================================================
# HELPERS
# ============================================================
def build_class_weights_from_csv(csv_path, num_classes=7):
    df = pd.read_csv(csv_path)
    label_counts = Counter(df["label"].tolist())

    total = sum(label_counts.values())
    weights = []

    for cls in range(num_classes):
        count = label_counts.get(cls, 1)
        weights.append(total / count)

    weights = torch.tensor(weights, dtype=torch.float32)
    weights = weights / weights.mean()

    return label_counts, weights


def set_trainable_layers(model, stage="head_only"):
    """
    stage:
      - 'head_only'        : only head trainable
      - 'partial_unfreeze' : head + last Swin stage + norm
      - 'full_unfreeze'    : all trainable
    """
    for param in model.parameters():
        param.requires_grad = False

    if stage == "head_only":
        for param in model.head.parameters():
            param.requires_grad = True

    elif stage == "partial_unfreeze":
        for param in model.head.parameters():
            param.requires_grad = True

        # Unfreeze last feature stage and normalization if they exist
        for name, param in model.named_parameters():
            if name.startswith("features.6") or name.startswith("norm"):
                param.requires_grad = True

    elif stage == "full_unfreeze":
        for param in model.parameters():
            param.requires_grad = True

    else:
        raise ValueError(f"Unknown stage: {stage}")


def build_optimizer(model, head_lr, backbone_lr, weight_decay):
    head_params = []
    backbone_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if name.startswith("head"):
            head_params.append(param)
        else:
            backbone_params.append(param)

    param_groups = []
    if backbone_params:
        param_groups.append({
            "params": backbone_params,
            "lr": backbone_lr,
            "weight_decay": weight_decay
        })
    if head_params:
        param_groups.append({
            "params": head_params,
            "lr": head_lr,
            "weight_decay": weight_decay
        })

    optimizer = torch.optim.AdamW(param_groups)
    return optimizer


def save_full_checkpoint(path, epoch, model, optimizer, scheduler, scaler, best_val_f1, stage):
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
        "scaler_state_dict": scaler.state_dict() if scaler is not None else None,
        "best_val_f1": best_val_f1,
        "stage": stage,
    }, path)


def print_trainable_summary(model):
    trainable = []
    frozen = 0
    trainable_count = 0

    for name, param in model.named_parameters():
        if param.requires_grad:
            trainable.append(name)
            trainable_count += param.numel()
        else:
            frozen += param.numel()

    print(f"[INFO] Trainable parameter tensors: {len(trainable)}")
    print(f"[INFO] Trainable parameters: {trainable_count:,}")
    print(f"[INFO] Frozen parameters:    {frozen:,}")

    print("[INFO] First trainable layers:")
    for name in trainable[:20]:
        print(f"  - {name}")


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

        optimizer.zero_grad(set_to_none=True)

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
    print(f"[INFO] Device: {DEVICE}")
    print(f"[INFO] Train CSV: {TRAIN_CSV}")
    print(f"[INFO] Val CSV:   {VAL_CSV}")

    if not os.path.exists(PRETRAIN_WEIGHTS_PATH):
        raise FileNotFoundError(
            f"Cholec80 pretrained weights not found:\n{PRETRAIN_WEIGHTS_PATH}"
        )

    train_dataset = CholecTrack20Dataset(TRAIN_CSV, mode="train")
    val_dataset   = CholecTrack20Dataset(VAL_CSV, mode="val")

    label_counts, class_weights = build_class_weights_from_csv(TRAIN_CSV, num_classes=NUM_CLASSES)
    class_weights = class_weights.to(DEVICE)

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

    # Build model exactly like in pretraining
    model, in_features = build_model(num_classes=NUM_CLASSES, pretrained=True)
    model.head = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(in_features, NUM_CLASSES)
    )

    # Load Cholec80 pretrained weights
    state_dict = torch.load(PRETRAIN_WEIGHTS_PATH, map_location=DEVICE, weights_only=True)
    model.load_state_dict(state_dict)
    print(f"[INFO] Loaded Cholec80 pretrained weights from: {PRETRAIN_WEIGHTS_PATH}")

    model = model.to(DEVICE)

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    scaler = GradScaler(enabled=(DEVICE.type == "cuda"))

    best_val_f1 = -1.0
    no_improve_epochs = 0
    current_stage = "head_only"

    # -----------------------------
    # Stage 1: head only
    # -----------------------------
    set_trainable_layers(model, stage="head_only")
    print("\n[INFO] Stage 1: HEAD ONLY fine-tuning")
    print_trainable_summary(model)

    optimizer = build_optimizer(
        model,
        head_lr=HEAD_LR,
        backbone_lr=BACKBONE_LR,
        weight_decay=WEIGHT_DECAY
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=2
    )

    for epoch in range(1, HEAD_ONLY_EPOCHS + 1):
        print(f"\n[INFO] Epoch {epoch}/{TOTAL_EPOCHS} | Stage: {current_stage}")

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

        lrs = [group["lr"] for group in optimizer.param_groups]
        print(
            f"[INFO] lrs={lrs} | "
            f"train_loss={train_loss:.4f} | train_f1={train_f1:.4f} | "
            f"val_loss={val_loss:.4f} | val_f1={val_f1:.4f}"
        )

        save_full_checkpoint(
            path=FULL_CHECKPOINT_PATH,
            epoch=epoch,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            best_val_f1=best_val_f1,
            stage=current_stage
        )

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            no_improve_epochs = 0
            torch.save(model.state_dict(), BEST_MODEL_PATH)

            save_full_checkpoint(
                path=FULL_CHECKPOINT_PATH,
                epoch=epoch,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                best_val_f1=best_val_f1,
                stage=current_stage
            )

            print(f"[INFO] New best model saved: {BEST_MODEL_PATH}")
        else:
            no_improve_epochs += 1
            print(f"[INFO] No improvement for {no_improve_epochs} epoch(s).")

    # -----------------------------
    # Stage 2: partial unfreeze
    # -----------------------------
    current_stage = "partial_unfreeze"
    set_trainable_layers(model, stage="partial_unfreeze")
    print("\n[INFO] Stage 2: PARTIAL UNFREEZE fine-tuning")
    print_trainable_summary(model)

    optimizer = build_optimizer(
        model,
        head_lr=5e-5,
        backbone_lr=BACKBONE_LR,
        weight_decay=WEIGHT_DECAY
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=2
    )

    for epoch in range(HEAD_ONLY_EPOCHS + 1, TOTAL_EPOCHS + 1):
        print(f"\n[INFO] Epoch {epoch}/{TOTAL_EPOCHS} | Stage: {current_stage}")

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

        lrs = [group["lr"] for group in optimizer.param_groups]
        print(
            f"[INFO] lrs={lrs} | "
            f"train_loss={train_loss:.4f} | train_f1={train_f1:.4f} | "
            f"val_loss={val_loss:.4f} | val_f1={val_f1:.4f}"
        )

        save_full_checkpoint(
            path=FULL_CHECKPOINT_PATH,
            epoch=epoch,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            best_val_f1=best_val_f1,
            stage=current_stage
        )

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            no_improve_epochs = 0
            torch.save(model.state_dict(), BEST_MODEL_PATH)

            save_full_checkpoint(
                path=FULL_CHECKPOINT_PATH,
                epoch=epoch,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                best_val_f1=best_val_f1,
                stage=current_stage
            )

            print(f"[INFO] New best model saved: {BEST_MODEL_PATH}")
        else:
            no_improve_epochs += 1
            print(f"[INFO] No improvement for {no_improve_epochs} epoch(s).")

        if no_improve_epochs >= PATIENCE:
            print(f"[INFO] Early stopping triggered after {PATIENCE} epochs without improvement.")
            break

    print("\n[INFO] Fine-tuning finished.")
    print(f"[INFO] Best val_f1: {best_val_f1:.4f}")
    print(f"[INFO] Best weights: {BEST_MODEL_PATH}")
    print(f"[INFO] Full checkpoint: {FULL_CHECKPOINT_PATH}")


if __name__ == "__main__":
    main()