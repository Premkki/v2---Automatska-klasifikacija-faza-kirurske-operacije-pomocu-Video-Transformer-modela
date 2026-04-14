import os
import pandas as pd
from sklearn.model_selection import train_test_split

DATASET_ROOT = r"way to the folder with cholec80 dataset"

INPUT_CSV = os.path.join(DATASET_ROOT, "pretraining_sequences.csv")
TRAIN_OUT = os.path.join(DATASET_ROOT, "pretrain_train_sequences.csv")
VAL_OUT   = os.path.join(DATASET_ROOT, "pretrain_val_sequences.csv")

VAL_RATIO = 0.10
RANDOM_STATE = 42

def main():
    df = pd.read_csv(INPUT_CSV)

    cases = sorted(df["case"].unique())
    train_cases, val_cases = train_test_split(
        cases,
        test_size=VAL_RATIO,
        random_state=RANDOM_STATE
    )

    train_df = df[df["case"].isin(train_cases)].reset_index(drop=True)
    val_df   = df[df["case"].isin(val_cases)].reset_index(drop=True)

    train_df.to_csv(TRAIN_OUT, index=False, encoding="utf-8")
    val_df.to_csv(VAL_OUT, index=False, encoding="utf-8")

    print(f"[INFO] Total sequences: {len(df)}")
    print(f"[INFO] Train cases:      {len(train_cases)}")
    print(f"[INFO] Val cases:        {len(val_cases)}")
    print(f"[INFO] Train sequences:  {len(train_df)}")
    print(f"[INFO] Val sequences:    {len(val_df)}")
    print(f"[INFO] Saved:")
    print(TRAIN_OUT)
    print(VAL_OUT)

if __name__ == "__main__":
        main()