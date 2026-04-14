import os
import pandas as pd
from PIL import Image, ImageFile
import torch
from torch.utils.data import Dataset
from torchvision import transforms

# Opcionalno: dozvoli PIL-u da pokuša učitati djelomično oštećene slike
# Ako želiš strogu provjeru, stavi False
ImageFile.LOAD_TRUNCATED_IMAGES = True


PHASE_NAMES = {
    0: "Preparation",
    1: "CalotTriangleDissection",
    2: "ClippingCutting",
    3: "GallbladderDissection",
    4: "GallbladderPackaging",
    5: "CleaningCoagulation",
    6: "GallbladderRetraction"
}


class CholecTrack20Dataset(Dataset):
    def __init__(self, csv_file, mode="train"):
        """
        csv_file : putanja do train_sequences.csv / val_sequences.csv / test_sequences.csv
        mode     : 'train' ili 'val'
        """
        self.data = pd.read_csv(csv_file)
        self.mode = mode
        self.transform = self._build_transforms()

    def _build_transforms(self):
        if self.mode == "train":
            return transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.RandomCrop((224, 224)),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(
                    brightness=0.2,
                    contrast=0.2,
                    saturation=0.2
                ),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ])
        else:
            return transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ])

    def _load_image(self, path):
        """
        Pokušaj učitati sliku iz točne putanje.
        Ako putanja nema valjanu ekstenziju ili je kriva, probaj .png i .jpg.
        """
        candidates = []

        if os.path.exists(path):
            candidates.append(path)

        base_path = os.path.splitext(path)[0]
        for ext in [".png", ".jpg", ".jpeg"]:
            candidate = base_path + ext
            if candidate not in candidates and os.path.exists(candidate):
                candidates.append(candidate)

        if not candidates:
            raise FileNotFoundError(f"[ERROR] Image not found: {path}")

        last_error = None
        for candidate in candidates:
            try:
                with Image.open(candidate) as img:
                    img = img.convert("RGB")
                return img
            except Exception as e:
                last_error = e
                print(f"[ERROR] Failed to load image: {candidate}")
                print(f"[ERROR] Reason: {e}")

        raise OSError(f"[ERROR] All image candidates failed for path: {path}\nLast error: {last_error}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]

        image_paths = row["image_paths"].split("|")
        frames = []

        for path in image_paths:
            img = self._load_image(path)
            img = self.transform(img)
            frames.append(img)

        clip = torch.stack(frames, dim=0)   # (T, C, H, W)
        clip = clip.permute(1, 0, 2, 3)     # (C, T, H, W)

        label = torch.tensor(int(row["label"]), dtype=torch.long)

        return clip, label


if __name__ == "__main__":
    DATASET_ROOT = r"way to the folder with cholectrack20 dataset"

    test_dataset = CholecTrack20Dataset(
        csv_file=os.path.join(DATASET_ROOT, "test_sequences.csv"),
        mode="val"
    )

    print(f"Broj sekvenci: {len(test_dataset)}")
    clip, label = test_dataset[0]
    print(f"Clip shape: {clip.shape}")
    print(f"Label: {label.item()} ({PHASE_NAMES[label.item()]})")