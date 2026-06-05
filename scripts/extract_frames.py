import cv2
import os

# =========================
# PARAMETERS
# =========================

DATA_SPLIT = "train"  # change to "val" or "test"
VIDEO_NAME = "train_11.mp4"
VIDEO_PREFIX = "train_11"

FRAME_OFFSET = 0

# =========================
# PATH SETUP
# =========================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

VIDEO_PATH = os.path.join(BASE_DIR, "videos", DATA_SPLIT, VIDEO_NAME)
OUTPUT_DIR = os.path.join(BASE_DIR, "frames_full", DATA_SPLIT)

# =========================
# SETUP
# =========================

os.makedirs(OUTPUT_DIR, exist_ok=True)

cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    print(f"[ERROR] Cannot open video: {VIDEO_PATH}")
    exit()

frame_id = 0
saved_count = 0

# =========================
# MAIN LOOP
# =========================

while True:
    ret, frame = cap.read()
    if not ret:
        break

    adjusted_id = frame_id + FRAME_OFFSET

    if adjusted_id >= 0:
        filename = f"{VIDEO_PREFIX}_{adjusted_id:06d}.jpg"
        path = os.path.join(OUTPUT_DIR, filename)
        cv2.imwrite(path, frame)
        saved_count += 1

    frame_id += 1


print("VIDEO_PATH:", VIDEO_PATH)
cap.release()

print(f"Frames read: {frame_id}")
print(f"Frames saved: {saved_count}")
print(f"Output dir: {OUTPUT_DIR}")