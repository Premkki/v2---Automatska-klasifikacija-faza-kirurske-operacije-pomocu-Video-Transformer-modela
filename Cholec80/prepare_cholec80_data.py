import os
import csv
from collections import Counter

DATASET_ROOT    = r"way to the folder with cholec80 dataset"
ANNOTATIONS_DIR = os.path.join(DATASET_ROOT, "cholec80", "phase_annotations")
FRAMES_DIR      = os.path.join(DATASET_ROOT, "frames")

PHASE_MAP = {
    "Preparation":              0,
    "CalotTriangleDissection":  1,
    "ClippingCutting":          2,
    "GallbladderDissection":    3,
    "GallbladderPackaging":     4,
    "CleaningCoagulation":      5,
    "GallbladderRetraction":    6
}

def get_split(video_name):
    return "pretraining"

def parse_txt_annotation(txt_path):
    frame_phase = {}

    with open(txt_path, "r") as f:
        lines = f.readlines()

    for line in lines[1:]:
        line = line.strip()
        if not line:
            continue

        parts = line.split("\t")
        if len(parts) != 2:
            continue

        frame_id = int(parts[0])
        phase_name = parts[1].strip()

        if phase_name in PHASE_MAP:
            frame_phase[frame_id] = PHASE_MAP[phase_name]

    return frame_phase

def main():
    all_samples = {"pretraining": []}

    for txt_file in sorted(os.listdir(ANNOTATIONS_DIR)):
        if not txt_file.endswith("-phase.txt"):
            continue

        video_name = txt_file.replace("-phase.txt", "")
        txt_path   = os.path.join(ANNOTATIONS_DIR, txt_file)
        frames_dir = os.path.join(FRAMES_DIR, video_name)
        split      = get_split(video_name)

        if not os.path.isdir(frames_dir):
            print(f"[WARNING] No frames folder: {video_name}")
            continue

        frame_phase = parse_txt_annotation(txt_path)
        samples = []

        for frame_id, phase in frame_phase.items():
            image_name = f"{frame_id:06d}.jpg"
            image_path = os.path.join(frames_dir, image_name)

            if not os.path.exists(image_path):
                continue

            samples.append({
                "split": split,
                "case": video_name,
                "frame_id": frame_id,
                "image_path": image_path,
                "phase": phase
            })

        samples.sort(key=lambda x: x["frame_id"])
        all_samples[split].extend(samples)
        print(f"[INFO] {split} | {video_name}: {len(samples)} samples")

    output_csv = os.path.join(DATASET_ROOT, "pretraining_frame_phase_pairs.csv")

    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["split", "case", "frame_id", "image_path", "phase"]
        )
        writer.writeheader()
        writer.writerows(all_samples["pretraining"])

    print(f"\n[INFO] pretraining: {len(all_samples['pretraining'])} total samples")
    counter = Counter(s["phase"] for s in all_samples["pretraining"])
    for phase, count in sorted(counter.items()):
        print(f"  Phase {phase}: {count}")

    print("\n[INFO] Done!")

if __name__ == "__main__":
    main()
