import os
import cv2
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from torchvision import transforms
from collections import deque
from model import build_model

# ============================================================
# POSTAVKE
# ============================================================
DATASET_ROOT   = r"way to the folder with cholectrack20 dataset"
CHECKPOINT_DIR = os.path.join(DATASET_ROOT, "checkpoints")
MODEL_PATH     = os.path.join(CHECKPOINT_DIR, "best_model_ever.pth")

# Odaberi video za vizualizaciju (jedan od testing videa)
VIDEO_PATH = r"way to the folder with cholectrack20 dataset, choose one video"

NUM_CLASSES  = 7
SEQUENCE_LEN = 16

PHASE_NAMES = [
    "Preparation",
    "CalotTriangleDissection",
    "ClippingCutting",
    "GallbladderDissection",
    "GallbladderPackaging",
    "CleaningCoagulation",
    "GallbladderRetraction"
]

PHASE_COLORS = [
    (255, 165, 0),    # narančasta
    (0, 255, 0),      # zelena
    (255, 0, 0),      # crvena
    (0, 0, 255),      # plava
    (255, 255, 0),    # žuta
    (0, 255, 255),    # cijan
    (255, 0, 255),    # magenta
]

# ============================================================
# PREPROCESSING
# ============================================================
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# ============================================================
# GLAVNI DIO
# ============================================================
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Koristi se: {device}")

    # Učitaj model
    model, in_features = build_model(num_classes=NUM_CLASSES, pretrained=False)
    model.head = nn.Sequential(
        nn.Dropout(p=0.5),
        nn.Linear(in_features, NUM_CLASSES)
    )
    model.load_state_dict(torch.load(
        MODEL_PATH, map_location=device, weights_only=True
    ))
    model = model.to(device)
    model.eval()
    print(f"[INFO] Model učitan!")

    # Otvori video
    cap = cv2.VideoCapture(VIDEO_PATH)
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"[INFO] Video: {VIDEO_PATH}")
    print(f"[INFO] FPS: {video_fps}, Ukupno frameova: {total_frames}")
    print(f"[INFO] Pritisni 'q' za izlaz, 'space' za pauzu\n")

    # Buffer za sekvence (uzorkujemo svaki 25. frame = 1fps)
    frame_buffer = deque(maxlen=SEQUENCE_LEN)
    frame_count  = 0
    current_pred = "Čekanje..."
    current_color = (255, 255, 255)
    current_conf  = 0.0
    paused = False

    while cap.isOpened():
        if not paused:
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1

            # Uzorkuj svaki 25. frame (1fps)
            if frame_count % 25 == 0:
                # Pretvori frame u PIL i primijeni transformacije
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_img   = Image.fromarray(rgb_frame)
                tensor    = transform(pil_img)
                frame_buffer.append(tensor)

                # Kad imamo dovoljno frameova, napravi predikciju
                if len(frame_buffer) == SEQUENCE_LEN:
                    with torch.no_grad():
                        clip = torch.stack(list(frame_buffer), dim=0)
                        clip = clip.permute(1, 0, 2, 3).unsqueeze(0)
                        clip = clip.to(device)

                        outputs = model(clip)
                        probs   = torch.softmax(outputs, dim=1)
                        conf, pred_idx = probs.max(1)

                        current_pred  = PHASE_NAMES[pred_idx.item()]
                        current_color = PHASE_COLORS[pred_idx.item()]
                        current_conf  = conf.item() * 100

        # Prikaz na ekranu
        display_frame = frame.copy()
        h, w = display_frame.shape[:2]

        # Pozadinski pravokutnik za tekst
        cv2.rectangle(display_frame, (0, 0), (w, 80), (0, 0, 0), -1)

        # Naziv faze
        cv2.putText(
            display_frame,
            f"Faza: {current_pred}",
            (10, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            current_color,
            2
        )

        # Pouzdanost
        cv2.putText(
            display_frame,
            f"Pouzdanost: {current_conf:.1f}%",
            (10, 65),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            1
        )

        # Progress bar
        progress = int((frame_count / total_frames) * w)
        cv2.rectangle(display_frame, (0, h-10), (progress, h), (0, 255, 0), -1)

        # Info o frameu
        cv2.putText(
            display_frame,
            f"Frame: {frame_count}/{total_frames}",
            (w - 200, h - 15),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1
        )

        cv2.imshow("Kirurske faze - Video Swin Transformer", display_frame)

        # Kontrole
        wait_ms = max(1, int(1000 / video_fps))
        key = cv2.waitKey(wait_ms) & 0xFF
        if key == ord('q'):
            break
        elif key == ord(' '):
            paused = not paused

    cap.release()
    cv2.destroyAllWindows()
    print("[INFO] Vizualizacija završena!")


if __name__ == "__main__":
    main()