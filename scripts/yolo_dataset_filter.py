import os
import shutil

# =========================
# PARAMETERS
# =========================

DATA_SPLIT = "train"   # train / val / test
FRAME_STEP = 5         # sampling

# =========================
# PATH SETUP
# =========================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

IMAGES_SRC = os.path.join(BASE_DIR, "frames_full", DATA_SPLIT)
LABELS_SRC = os.path.join(BASE_DIR, "labels_full", DATA_SPLIT)

IMAGES_DST = os.path.join(BASE_DIR, "dataset", "images", DATA_SPLIT)
LABELS_DST = os.path.join(BASE_DIR, "dataset", "labels", DATA_SPLIT)

# =========================
# SETUP
# =========================

os.makedirs(IMAGES_DST, exist_ok=True)
os.makedirs(LABELS_DST, exist_ok=True)

images = sorted([f for f in os.listdir(IMAGES_SRC) if f.endswith(".jpg")])

copied = 0

# =========================
# MAIN LOOP
# =========================

for i, img_name in enumerate(images):
    if i % FRAME_STEP != 0:
        continue

    # remove file extension
    name = os.path.splitext(img_name)[0]

    # extract numeric index from name
    index = name.split("_")[-1]

    label_name = f"frame_{index}.txt"
    dst_label_name = name + ".txt"

    shutil.copy(
        os.path.join(IMAGES_SRC, img_name),
        os.path.join(IMAGES_DST, img_name)
    )

    shutil.copy(
        os.path.join(LABELS_SRC, label_name),
        os.path.join(LABELS_DST, dst_label_name)

    )

    copied += 1

print(f"Copied pairs: {copied}")
print(f"Output: {IMAGES_DST}")