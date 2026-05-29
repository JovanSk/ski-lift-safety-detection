from ultralytics import YOLO
import cv2
import os
import numpy as np

# =========================
# PATH SETUP
# =========================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODEL_PATH = os.path.join(BASE_DIR, "models", "yolov8s_fine_tuned", "best.pt")
VIDEO_PATH = os.path.join(BASE_DIR, "videos", "sample", "sample_input.mp4")
OUTPUT_PATH = os.path.join(BASE_DIR, "output", "output.mp4")

# =========================
# CONFIGURATION
# =========================

IMG_SIZE = 768
CONF_THRESHOLD = 0.6

LINE = ((425, 680), (525, 550))
ALERT_CUTOFF_LINE = ((LINE[0][0] + 100, LINE[0][1]),
                      (LINE[1][0] + 100, LINE[1][1]))

ZONE_QUAD = [
    (355, 710),
    (470, 605),
    (470, 513),
    (355, 605)
]

FONT = cv2.FONT_HERSHEY_SIMPLEX

COLOR_ADULT = (255, 200, 100)
COLOR_CHILD = (0, 180, 0)
COLOR_ALERT = (0, 0, 255)

# =========================
# TRACKING CONFIG
# =========================

memory = []           # Stores active tracked objects across frames
label_memory = {}     # Stores label placement history per track ID, label_memory = {box, slot, lock}     

MAX_AGE = 3           # Frames to keep unmatched tracks alive
next_id = 0           # Track ID counter
DIST_THRESHOLD = 30
NEW_TRACK_THRESHOLD = 40

# =========================
# LABEL STABILITY CONFIG
# =========================

LABEL_SMOOTHING_ALPHA = 0.75
LABEL_LOCK_FRAMES = 4

# =========================
# ALERT CONFIG
# =========================

ALERT_FRAMES_TOTAL = 40
alert_timer = 0

# =========================
# UTILS
# =========================

def point_side(px, py, line):
    (x1, y1), (x2, y2) = line
    return (px - x1) * (y2 - y1) - (py - y1) * (x2 - x1)


def is_overlapping(a, b):
    return not (a[2] < b[0] or b[2] < a[0] or a[3] < b[1] or b[3] < a[1])



# ===============
# DRAW DETECTION
# ===============


def draw_detection(frame, box, label, center, det_id, placed_labels, alerted=False):
    x1, y1, x2, y2 = box
    cx, cy = center

    if alerted and label == "child":
        color = COLOR_ALERT
        text = "CHILD"
    elif label == "child":
        color = COLOR_CHILD
        text = "CHILD"
    else:
        color = COLOR_ADULT
        text = "ADULT"

    font_scale = 0.4
    thickness = 1

    
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 1)  # draw bbox


    #LABEL POSITIONING

    (tw, th), _ = cv2.getTextSize(text, FONT, font_scale, thickness)
    padding = 6

    
    # Initial candidate position for the label (on top of bbox)
    base_x1, base_y1 = x1, y1 - th - padding                               # top-left corner of the label box
    base_x2, base_y2 = base_x1 + tw + padding, y1                          # bottom-right corner of the label box

    candidate_positions = []
    for dy in [-5, -15, -25]:
        candidate_positions.append((base_x1, base_y1 + dy, base_x2, base_y2 + dy))

    prev_state = label_memory.get(det_id, {})    # retrieve from earlier frames previously stored label positioning for this tracking ID
    prev_box = prev_state.get("box")
    prev_slot = prev_state.get("slot")
    prev_lock = prev_state.get("lock", 0)

    chosen_box = None
    chosen_slot = None
    chosen_lock = 0

    # Prefer the previous slot if it is still available.
    # If it is locked, keep it for a few frames before allowing a change.

    if prev_slot is not None and 0 <= prev_slot < len(candidate_positions):  # prev_slot < len(candidate_positions) is currently always true, but will
                                                                             # be needed in case the number of candidate positions changes dynamically
        preferred = candidate_positions[prev_slot]
        if not any(is_overlapping(preferred, p) for p in placed_labels):     # check if preferred label position 
                                                                             # overlaps with any of placed labels
            chosen_box = preferred
            chosen_slot = prev_slot
            chosen_lock = max(prev_lock - 1, 0)

    # Otherwise choose the first free slot.
    if chosen_box is None:
        for idx, candidate in enumerate(candidate_positions):
            if not any(is_overlapping(candidate, p) for p in placed_labels):
                chosen_box = candidate
                chosen_slot = idx
                chosen_lock = LABEL_LOCK_FRAMES
                break

    # Fallback if all candidates are occupied.
    if chosen_box is None:
        chosen_box = candidate_positions[0]
        chosen_slot = 0
        chosen_lock = LABEL_LOCK_FRAMES

    # Apply a light smoothing to reduce label jitter when the slot changes.
    if prev_box is not None:
        smoothed_box = tuple(
            int(LABEL_SMOOTHING_ALPHA * prev_box[i] + (1 - LABEL_SMOOTHING_ALPHA) * chosen_box[i])
            for i in range(4)
        )

        if not any(is_overlapping(smoothed_box, p) for p in placed_labels):
            final_box = smoothed_box
        else:
            final_box = chosen_box
    else:
        final_box = chosen_box

    label_memory[det_id] = {
        "box": final_box,
        "slot": chosen_slot,
        "lock": chosen_lock,
    }

    placed_labels.append(final_box)

    lx1, ly1, lx2, ly2 = final_box    # final label box coordinates
    label_cx = (lx1 + lx2) // 2
    label_cy = (ly1 + ly2) // 2

    bbox_top_cx = (x1 + x2) // 2
    bbox_top_cy = y1


    if not (lx2 >= x1 and lx1 <= x2 and ly2 >= y1 and ly1 <= y2):
        cv2.line(frame, (bbox_top_cx, bbox_top_cy), (label_cx, label_cy), color, 1)  # draw leader line (bbox - label)

    cv2.rectangle(frame, (lx1, ly1), (lx2, ly2), color, -1)  # draw label background
    cv2.putText(frame, text, (lx1 + 5, ly2 - 4), FONT, font_scale, (255, 255, 255), thickness)  # render label text




# =========================
# MODEL
# =========================

model = YOLO(MODEL_PATH)

cap = cv2.VideoCapture(VIDEO_PATH)

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(
    OUTPUT_PATH,
    fourcc,
    25,
    (int(cap.get(3)), int(cap.get(4)))
)



# =========================
# MAIN LOOP
# =========================


while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame, imgsz=IMG_SIZE, conf=CONF_THRESHOLD)

    current_detections = []

    for result in results:
        for box in result.boxes:
            cls = int(box.cls[0])
            label = "adult" if cls == 0 else "child"

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cx, cy = (x1 + x2) // 2, y2

            current_detections.append({
                "box": (x1, y1, x2, y2),
                "label": label,
                "center": (cx, cy),
                "id": None,
                "side": point_side(cx, cy, LINE),
                "age": 0,
                "alerted": False,
                "label_history": []
            })

    updated_memory = []     # Stores updated tracks for the current frame



    # ===================================================================
    # MATCH (DISTANCE + UNIQUE): 
    # match (associate) current detections to tracks from previous frames
    # ===================================================================

    for mem in memory:
        best = None
        best_dist = 99999

        for det in current_detections:
            if det["id"] is not None:
                continue

            d = np.hypot(det["center"][0] - mem["center"][0],
                         det["center"][1] - mem["center"][1])

            if d < best_dist and d < DIST_THRESHOLD:
                best = det
                best_dist = d

        if best:
            best["id"] = mem["id"]   # new detection is the same object as an existing track

            # SMOOTHING:
            # Blend previous and current detection positions to reduce bbox flicker
            # and small coordinate fluctuations between frames
            
            alpha = 0.85

            best["center"] = (
                int(alpha * mem["center"][0] + (1 - alpha) * best["center"][0]),
                int(alpha * mem["center"][1] + (1 - alpha) * best["center"][1])
            )

            old = mem["box"]
            new = best["box"]

            best["box"] = (
                int(alpha * old[0] + (1 - alpha) * new[0]),
                int(alpha * old[1] + (1 - alpha) * new[1]),
                int(alpha * old[2] + (1 - alpha) * new[2]),
                int(alpha * old[3] + (1 - alpha) * new[3]),
            )

            # LABEL STABILIZATION:
            # stabilize child/adult classification using temporal window (last 5 frames)

            history = mem.get("label_history", [])
            history.append(best["label"])
            history = history[-5:]

            best["label"] = "child" if history.count("child") >= 3 else "adult"
            best["label_history"] = history

            best["alerted"] = mem.get("alerted", False)
            best["age"] = 0

            updated_memory.append(best)

        else:                               # failed to find this object in the current frame
            mem["age"] += 1
            if mem["age"] <= MAX_AGE:
                updated_memory.append(mem)

    # ===================
    # NEW DETECTIONS
    # ===================

    for det in current_detections:
        if det["id"] is None:

            # Do not create a new track if this detection is too close
            # to an already existing track.
            
            too_close = any(
                np.hypot(det["center"][0] - mem["center"][0],
                         det["center"][1] - mem["center"][1]) < NEW_TRACK_THRESHOLD
                for mem in updated_memory
            )

            if too_close:
                continue

            det["id"] = next_id     # Create a new tracking ID for this detection
            next_id += 1
            det["label_history"] = [det["label"]]
            updated_memory.append(det)


    # DEDUPLICATE
    unique = {}
    for m in updated_memory:
        unique[m["id"]] = m
    memory = list(unique.values())

    # Remove aged-out label memory entries
    active_ids = {m["id"] for m in memory}
    for aged_id in list(label_memory.keys()):
        if aged_id not in active_ids:
     
            del label_memory[aged_id]

    # =========================
    # ALERT
    # =========================


    for det in memory:
        # If the object is already past the cutoff line, do not allow a new alert.
        cutoff_side = point_side(det["center"][0], det["center"][1], ALERT_CUTOFF_LINE)

        if cutoff_side < 0:
            det["alert_disabled"] = True

        if det["label"] == "child" and det["side"] < 0:
            if not det.get("alerted", False) and not det.get("alert_disabled", False):
                alert_timer = ALERT_FRAMES_TOTAL
            det["alerted"] = True




    # =======
    # DRAW
    # =======

    overlay = frame.copy()
    pts = np.array(ZONE_QUAD, np.int32)

    cv2.fillPoly(overlay, [pts], (0, 0, 255))
    cv2.addWeighted(overlay, 0.1, frame, 0.9, 0, frame)
    cv2.polylines(frame, [pts], True, (0, 0, 150), 1)

    placed_labels = []
    seen_ids = set()

    for m in memory:
        if m["age"] > 1:
           continue

        #skip duplicate ID
        if m["id"] in seen_ids:
            continue

#        seen_ids.add(m["id"])

        # render detection
        draw_detection(
            frame,
            m["box"],
            m["label"],
            m["center"],
            m["id"],
            placed_labels,
            m.get("alerted", False)
        )

    # =========================
    # ALERT BOX
    # =========================

    if alert_timer > 0:
        alert_timer -= 1
        if (alert_timer // 14) % 2 == 0:
            text = "ALERT: CHILD IN ZONE"
            font_scale = 0.9
            thickness = 2

            (tw, th), _ = cv2.getTextSize(text, FONT, font_scale, thickness)

            x, y = 40, 120
            padding = 12

            cv2.rectangle(
                frame,
                (x, y - th - padding),
                (x + tw + padding, y + padding // 2),
                (0, 0, 200),
                -1
            )

            cv2.putText(frame, text, (x + 6, y + 2), FONT, font_scale, (0, 0, 0), thickness + 1)
            cv2.putText(frame, text, (x + 5, y), FONT, font_scale, (255, 255, 255), thickness)

    out.write(frame)
    cv2.imshow("Ski Lift Safety Monitor", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
out.release()
cv2.destroyAllWindows()
