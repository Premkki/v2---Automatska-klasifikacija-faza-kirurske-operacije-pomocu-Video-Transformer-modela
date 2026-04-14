import pandas as pd
import os
from collections import Counter

# ============================================================
# SETTINGS
# ============================================================
DATASET_ROOT = r"way to the folder with cholec80 dataset"

PRETRAIN_CSV = os.path.join(DATASET_ROOT, "pretraining_frame_phase_pairs.csv")

SEQUENCE_LENGTH = 16
STRIDE = 8

PRETRAIN_SEQ_CSV = os.path.join(DATASET_ROOT, "pretraining_sequences.csv")


# ============================================================
# CREATE SEQUENCES
# ============================================================
def create_sequences(input_csv, output_csv, seq_len=16, stride=8):
    df = pd.read_csv(input_csv)
    df = df.sort_values(by=["case", "frame_id"]).reset_index(drop=True)

    rows = []
    mixed_count = 0

    for case_name, group in df.groupby("case"):
        group = group.sort_values("frame_id").reset_index(drop=True)
        total = len(group)

        for start in range(0, total - seq_len + 1, stride):
            end = start + seq_len
            window = group.iloc[start:end]

            phases_in_window = window["phase"].tolist()
            image_paths = window["image_path"].tolist()
            frame_ids = window["frame_id"].tolist()

            center_idx = seq_len // 2
            label = phases_in_window[center_idx]

            phase_counts = Counter(phases_in_window)
            dominant_phase, dominant_count = phase_counts.most_common(1)[0]
            purity = dominant_count / seq_len
            unique_phases = len(phase_counts)

            rows.append({
                "case": case_name,
                "start_frame": frame_ids[0],
                "end_frame": frame_ids[-1],
                "center_frame": frame_ids[center_idx],
                "label": label,                     # center-frame label
                "dominant_phase": dominant_phase,
                "purity": purity,                   # e.g. 1.0 = fully pure
                "pure_sequence": 1 if unique_phases == 1 else 0,
                "image_paths": "|".join(image_paths),
                "frame_ids": "|".join(map(str, frame_ids)),
            })

            if unique_phases > 1:
                mixed_count += 1

    seq_df = pd.DataFrame(rows)
    seq_df = seq_df[seq_df["purity"] >= 0.75].reset_index(drop=True)
    seq_df.to_csv(output_csv, index=False, encoding="utf-8")

    print(f"\n{'='*60}")
    print(f"Izlaz: {output_csv}")
    print(f"Ukupno sekvenci:        {len(seq_df)}")
    print(f"Čiste sekvence:         {(seq_df['pure_sequence'] == 1).sum()}")
    print(f"Mješovite sekvence:     {mixed_count}")
    print(f"\nDistribucija labela:")
    for phase, count in sorted(Counter(seq_df['label']).items()):
        print(f"  Phase {phase}: {count} sekvenci")
    print(f"{'='*60}\n")

    return seq_df


# ============================================================
# RUN
# ============================================================
if __name__ == "__main__":
    print("[INFO] Kreiram PRETRAIN sekvence...")
    create_sequences(PRETRAIN_CSV, PRETRAIN_SEQ_CSV, SEQUENCE_LENGTH, STRIDE)
