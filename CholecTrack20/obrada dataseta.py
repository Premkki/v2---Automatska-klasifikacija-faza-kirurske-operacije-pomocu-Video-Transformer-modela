import os
import json
import csv
from collections import Counter

# =========================================================
# 1. POSTAVI PUTANJU DO CHOLECTRACK20 DATASETA
# =========================================================
# Primjer:
DATASET_ROOT = r"way to the folder with cholectrack20 dataset"


# =========================================================
# 2. POMOĆNE FUNKCIJE
# =========================================================
def find_json_file(case_folder):
    """
    Pronađe prvi .json file unutar case foldera.
    """
    for file_name in os.listdir(case_folder):
        if file_name.endswith(".json"):
            return os.path.join(case_folder, file_name)
    return None


def load_json_annotations(json_path):
    """
    Učita JSON i vrati annotations dio.
    """
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if "annotations" not in data:
        raise KeyError(f"JSON nema ključ 'annotations': {json_path}")

    return data["annotations"]


def frame_id_to_filename(frame_id):
    """
    Pretvara broj framea u ime slike formata 000026.png
    """
    return f"{int(frame_id):06d}.png"


def load_frame_phase_pairs(case_folder, split_name):
    """
    Za jedan case folder učita sve parove:
    (frame_id, image_path, phase)

    Vraća listu dict objekata.
    """
    frames_folder = os.path.join(case_folder, "frames")
    if not os.path.isdir(frames_folder):
        print(f"[UPOZORENJE] Nema frames foldera: {frames_folder}")
        return []

    json_file = find_json_file(case_folder)
    if json_file is None:
        print(f"[UPOZORENJE] Nema JSON datoteke u: {case_folder}")
        return []

    annotations = load_json_annotations(json_file)

    case_name = os.path.basename(case_folder)
    samples = []

    for frame_id, ann_list in annotations.items():
        if not ann_list:
            continue

        image_name = frame_id_to_filename(frame_id)
        image_path = os.path.join(frames_folder, image_name)

        if not os.path.exists(image_path):
            # Ako slika ne postoji, preskoči
            continue

        # Uzmi fazu iz prve anotacije
        phase = ann_list[0]["phase"]

        samples.append({
            "split": split_name,
            "case": case_name,
            "frame_id": int(frame_id),
            "image_path": image_path,
            "phase": int(phase)
        })

    # sortiraj po frame_id
    samples.sort(key=lambda x: x["frame_id"])
    return samples


def collect_split_samples(split_folder, split_name):
    """
    Prođe kroz sve case foldere unutar jednog splita (training/validation)
    i prikupi sve uzorke.
    """
    all_samples = []

    if not os.path.isdir(split_folder):
        print(f"[UPOZORENJE] Split folder ne postoji: {split_folder}")
        return all_samples

    for item in os.listdir(split_folder):
        case_folder = os.path.join(split_folder, item)

        # preskoči manifest datoteke i sve što nije folder
        if not os.path.isdir(case_folder):
            continue

        case_samples = load_frame_phase_pairs(case_folder, split_name)
        all_samples.extend(case_samples)

        print(f"[INFO] {split_name} | {item} | broj uzoraka: {len(case_samples)}")

    return all_samples


def save_samples_to_csv(samples, output_csv):
    """
    Spremi uzorke u CSV.
    """
    fieldnames = ["split", "case", "frame_id", "image_path", "phase"]

    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(samples)


def print_phase_distribution(samples, title):
    """
    Ispis distribucije faza.
    """
    counter = Counter(sample["phase"] for sample in samples)

    print(f"\n=== Distribucija faza: {title} ===")
    for phase_id in sorted(counter.keys()):
        print(f"Phase {phase_id}: {counter[phase_id]}")
    print(f"Ukupno uzoraka: {len(samples)}\n")


# =========================================================
# 3. GLAVNI DIO
# =========================================================
def main():
    training_folder = os.path.join(DATASET_ROOT, "training")
    validation_folder = os.path.join(DATASET_ROOT, "validation")

    print("[INFO] Učitavanje training uzoraka...")
    train_samples = collect_split_samples(training_folder, "training")

    print("[INFO] Učitavanje validation uzoraka...")
    val_samples = collect_split_samples(validation_folder, "validation")

    all_samples = train_samples + val_samples

    print("\n==============================")
    print(f"Ukupno training uzoraka:   {len(train_samples)}")
    print(f"Ukupno validation uzoraka: {len(val_samples)}")
    print(f"Ukupno svih uzoraka:       {len(all_samples)}")
    print("==============================\n")

    # Ispiši prvih nekoliko uzoraka
    print("Primjer nekoliko training uzoraka:")
    for sample in train_samples[:5]:
        print(sample)

    print_phase_distribution(train_samples, "TRAINING")
    print_phase_distribution(val_samples, "VALIDATION")

    # Spremi CSV
    output_train_csv = os.path.join(DATASET_ROOT, "train_frame_phase_pairs.csv")
    output_val_csv = os.path.join(DATASET_ROOT, "val_frame_phase_pairs.csv")
    output_all_csv = os.path.join(DATASET_ROOT, "all_frame_phase_pairs.csv")

    save_samples_to_csv(train_samples, output_train_csv)
    save_samples_to_csv(val_samples, output_val_csv)
    save_samples_to_csv(all_samples, output_all_csv)

    print("[INFO] CSV datoteke spremljene:")
    print(output_train_csv)
    print(output_val_csv)
    print(output_all_csv)


if __name__ == "__main__":
    main()