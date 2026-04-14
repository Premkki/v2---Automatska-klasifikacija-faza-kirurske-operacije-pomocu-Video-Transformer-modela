import os
import json
import csv
from collections import Counter

DATASET_ROOT = r"way to the folder with cholectrack20 dataset"
TESTING_DIR  = os.path.join(DATASET_ROOT, "testing")
OUTPUT_CSV   = os.path.join(DATASET_ROOT, "test_frame_phase_pairs.csv")

def load_json_annotations(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data.get("annotations", {})

def main():
    samples = []

    for case_name in sorted(os.listdir(TESTING_DIR)):
        case_folder = os.path.join(TESTING_DIR, case_name)

        if not os.path.isdir(case_folder):
            continue

        frames_folder = os.path.join(case_folder, "frames")
        if not os.path.isdir(frames_folder):
            print(f"[UPOZORENJE] Nema frames foldera: {case_name}")
            continue

        # Pronađi JSON
        json_file = None
        for f in os.listdir(case_folder):
            if f.endswith(".json"):
                json_file = os.path.join(case_folder, f)
                break

        if json_file is None:
            print(f"[UPOZORENJE] Nema JSON datoteke: {case_name}")
            continue

        annotations = load_json_annotations(json_file)
        case_samples = []

        for frame_id, ann_list in annotations.items():
            if not ann_list:
                continue

            # Isto kao training/validation
            image_name = f"{int(frame_id):06d}.png"
            image_path = os.path.join(frames_folder, image_name)

            if not os.path.exists(image_path):
                continue

            phase = int(ann_list[0]["phase"])

            case_samples.append({
                "split": "testing",
                "case": case_name,
                "frame_id": int(frame_id),
                "image_path": image_path,
                "phase": phase
            })

        case_samples.sort(key=lambda x: x["frame_id"])
        samples.extend(case_samples)
        print(f"[INFO] {case_name}: {len(case_samples)} uzoraka")

    # Spremi CSV
    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["split", "case", "frame_id", "image_path", "phase"]
        )
        writer.writeheader()
        writer.writerows(samples)

    print(f"\n[INFO] Ukupno uzoraka: {len(samples)}")
    print(f"[INFO] Distribucija faza:")
    counter = Counter(s["phase"] for s in samples)
    for phase, count in sorted(counter.items()):
        print(f"  Phase {phase}: {count}")
    print(f"\n[INFO] Spremljeno u: {OUTPUT_CSV}")

if __name__ == "__main__":
    main()