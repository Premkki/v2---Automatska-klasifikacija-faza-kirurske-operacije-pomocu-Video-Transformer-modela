import cv2
import os

DATASET_ROOT = r"way to the folder with cholec80 dataset"
VIDEOS_DIR   = os.path.join(DATASET_ROOT, "cholec80", "videos")
FRAMES_DIR   = os.path.join(DATASET_ROOT, "frames")

def extract_frames(video_path, output_dir, fps=1):
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
            filename = os.path.join(output_dir, f"{frame_idx:06d}.jpg")
            cv2.imwrite(filename, frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
            saved_idx += 1

        frame_idx += 1

    cap.release()
    print(f"  Saved {saved_idx} frames from {os.path.basename(video_path)}")

def main():
    print("[INFO] Extracting frames from Cholec80 videos...")

    for video_file in sorted(os.listdir(VIDEOS_DIR)):
        if not video_file.endswith(".mp4"):
            continue

        video_name = video_file.replace(".mp4", "")
        video_path = os.path.join(VIDEOS_DIR, video_file)
        output_dir = os.path.join(FRAMES_DIR, video_name)

        print(f"\n[INFO] Processing: {video_name}")
        extract_frames(video_path, output_dir, fps=1)

    print("\n[INFO] Done!")

if __name__ == "__main__":
    main()