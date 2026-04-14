import cv2
import os
import json

# ============================================================
# POSTAVKE
# ============================================================
DATASET_ROOT  = r"way to the folder with cholectrack20 dataset"
TESTING_DIR   = os.path.join(DATASET_ROOT, "testing")

# ============================================================
# EKSTRAKCIJA FRAMEOVA
# ============================================================
def extract_frames(video_path, output_dir, fps=1):
    """
    Izdvaja frameove iz videozapisa pri zadanoj stopi uzorkovanja.
    fps=1 znaci 1 frame po sekundi
    """
    os.makedirs(output_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)

    video_fps = cap.get(cv2.CAP_PROP_FPS)
    frame_interval = int(video_fps / fps)

    frame_idx = 0
    saved_idx = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % frame_interval == 0:
            filename = os.path.join(output_dir, f"{saved_idx * frame_interval + 1:06d}.png")
            cv2.imwrite(filename, frame)
            saved_idx += 1

        frame_idx += 1

    cap.release()
    print(f"  Spremljeno {saved_idx} frameova iz {os.path.basename(video_path)}")
    return saved_idx


# ============================================================
# GLAVNI DIO
# ============================================================
def main():
    print("[INFO] Ekstrakcija frameova iz testing videa...")

    for case_name in sorted(os.listdir(TESTING_DIR)):
        case_folder = os.path.join(TESTING_DIR, case_name)

        if not os.path.isdir(case_folder):
            continue

        # Pronađi video file
        video_file = None
        for f in os.listdir(case_folder):
            if f.endswith(".mp4"):
                video_file = os.path.join(case_folder, f)
                break

        if video_file is None:
            print(f"  [UPOZORENJE] Nema videa u: {case_name}")
            continue

        # Output folder za frameove
        frames_dir = os.path.join(case_folder, "frames")

        print(f"\n[INFO] Obradjujem: {case_name}")
        extract_frames(video_file, frames_dir, fps=1)

    print("\n[INFO] Ekstrakcija zavrsena!")


if __name__ == "__main__":
    main()