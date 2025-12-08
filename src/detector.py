# detector_improved.py
import os
import cv2
import json
import time
import logging
import numpy as np
from pathlib import Path
from datetime import datetime
import requests

# --- PATH CORRECTION ---
# Automatically find the project's root directory.
# This script is in 'src', so the root is one level up.
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# ---------------- Configuration (Using corrected paths) ----------------
# I have placed your token and chat ID here directly.
# For better security, consider using a .env file or real environment variables.
BOT_TOKEN = "8291475086:AAFr2q1d4wYkapspV1YqtbAlklwcMOyPNAg"
CHAT_ID   = "7038274330"

NOTIFICATION_COOLDOWN = 10.0      # seconds (global fallback)
PER_LABEL_COOLDOWN = 15.0     # seconds per label to avoid spamming
SNAPSHOT_COOLDOWN = 5.0       # seconds between snapshots

# --- PATH CORRECTION ---
# Define paths to model files relative to the project root.
MODELS_DIR = PROJECT_ROOT / "models"
MODEL_CFG = MODELS_DIR / "yolov4-tiny.cfg"
MODEL_WEIGHTS = MODELS_DIR / "yolov4-tiny.weights"
COCO_NAMES = MODELS_DIR / "coco.names"
SNAPSHOT_DIR = PROJECT_ROOT / "Image_snapshot"


# ---------------- Logging ----------------
logging.basicConfig(
    filename= PROJECT_ROOT / "detector.log", # Log file in the root directory
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
console = logging.StreamHandler()
console.setLevel(logging.INFO)
logging.getLogger().addHandler(console)

# ---------------- Ensure model files exist ----------------
for p in (MODEL_CFG, MODEL_WEIGHTS, COCO_NAMES):
    if not p.exists():
        logging.critical(f"Required file missing: {p}")
        raise SystemExit(f"Required file missing: {p}")

# ---------------- Create snapshot folder ----------------
SNAPSHOT_DIR.mkdir(exist_ok=True)

# ---------------- Load class names ----------------
with open(COCO_NAMES, "r") as f:
    classes = [c.strip() for c in f.read().splitlines() if c.strip()]
if not classes:
    logging.critical("No class names loaded from coco.names")
    raise SystemExit("coco.names seems empty or invalid")

# ---------------- Load YOLO (with backend fallback) ----------------
# --- PATH CORRECTION ---
# Convert Path objects to strings for OpenCV
net = cv2.dnn.readNet(str(MODEL_WEIGHTS), str(MODEL_CFG))


# --- Backend selection: prefer CUDA if truly available, otherwise CPU (robust) ---
use_cuda = False
try:
    # check whether CUDA backend/target constants exist
    has_cuda_backend = hasattr(cv2.dnn, "DNN_BACKEND_CUDA")
    has_cuda_target = hasattr(cv2.dnn, "DNN_TARGET_CUDA") or hasattr(cv2.dnn, "DNN_TARGET_CUDA_FP16")

    # check if OpenCV knows any CUDA devices
    cuda_device_count = 0
    if hasattr(cv2, "cuda") and hasattr(cv2.cuda, "getCudaEnabledDeviceCount"):
        try:
            cuda_device_count = cv2.cuda.getCudaEnabledDeviceCount()
        except Exception:
            cuda_device_count = 0

    if has_cuda_backend and has_cuda_target and cuda_device_count > 0:
        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
        # prefer FP16 if available for better perf on some builds
        if hasattr(cv2.dnn, "DNN_TARGET_CUDA_FP16"):
            net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA_FP16)
        else:
            net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
        use_cuda = True
        logging.info("Using CUDA backend for OpenCV DNN (CUDA devices: %d)", cuda_device_count)
        print(f"Using CUDA backend for OpenCV DNN (CUDA devices: {cuda_device_count})")
    else:
        raise RuntimeError("CUDA backend/target not available or no CUDA devices found")
except Exception as e:
    # fallback to CPU
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_DEFAULT)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
    logging.info("Falling back to CPU backend for OpenCV DNN (%s)", e)
    print("Falling back to CPU backend for OpenCV DNN:", e)

# Detection thresholds
CONF_THRESHOLD = 0.4
NMS_THRESHOLD = 0.4
INPUT_SIZE = (416, 416)

# Restricted zone (example) - tune these to your camera
ZONE_X1, ZONE_Y1, ZONE_X2, ZONE_Y2 = 100, 100, 540, 380

# ---------------- Utilities ----------------
def send_telegram_photo(caption: str, image) -> bool:
    """Send photo to Telegram; returns True on success."""
    if not BOT_TOKEN or not CHAT_ID:
        logging.warning("Telegram BOT_TOKEN/CHAT_ID not set. Skipping send.")
        return False
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendPhoto"
    _, buffer = cv2.imencode('.jpg', image)
    files = {"photo": ("image.jpg", buffer.tobytes(), "image/jpeg")}
    payload = {"chat_id": CHAT_ID, "caption": caption}
    try:
        r = requests.post(url, data=payload, files=files, timeout=10)
        logging.info("Telegram response: %s %s", r.status_code, r.text[:200])
        return r.status_code == 200
    except Exception as e:
        logging.exception("Failed to send Telegram photo: %s", e)
        return False

def detect_objects(frame):
    """Runs YOLO inference on frame and returns annotated frame, counts, detections list."""
    h, w = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1/255.0, INPUT_SIZE, swapRB=True, crop=False)
    net.setInput(blob)
    outputs = net.forward(net.getUnconnectedOutLayersNames())

    boxes, confidences, class_ids = [], [], []
    for output in outputs:
        for det in output:
            scores = det[5:]
            class_id = int(np.argmax(scores))
            conf = float(scores[class_id])
            if conf > CONF_THRESHOLD:
                cx = int(det[0] * w); cy = int(det[1] * h)
                bw = int(det[2] * w); bh = int(det[3] * h)
                x = int(cx - bw/2); y = int(cy - bh/2)
                boxes.append([x, y, bw, bh])
                confidences.append(conf)
                class_ids.append(class_id)
    indices = []
    if boxes:
        idxs = cv2.dnn.NMSBoxes(boxes, confidences, CONF_THRESHOLD, NMS_THRESHOLD)
        if isinstance(idxs, (list, tuple, np.ndarray)):
            indices = idxs.flatten() if hasattr(idxs, "flatten") else idxs
        else:
            # older OpenCV may return tuple-of-lists
            try:
                indices = np.array(idxs).flatten()
            except Exception:
                indices = []
    object_counts = {}
    detections = []
    annotated = frame.copy()
    for i in indices:
        i = int(i)
        x, y, bw, bh = boxes[i]
        label = classes[class_ids[i]] if 0 <= class_ids[i] < len(classes) else str(class_ids[i])
        object_counts[label] = object_counts.get(label, 0) + 1
        detections.append({"label": label, "box": (int(x), int(y), int(bw), int(bh)), "conf": float(confidences[i])})
        cv2.rectangle(annotated, (x, y), (x + bw, y + bh), (0, 255, 0), 2)
        cv2.putText(annotated, f"{label} {confidences[i]:.2f}", (x, y - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    return annotated, object_counts, detections

# ---------------- Run camera loop ----------------
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

if not cap.isOpened():
    logging.critical("Camera could not be opened")
    raise SystemExit("Camera error")

last_snapshot_time = 0.0
last_notification_time_global = 0.0
last_notification_per_label = {}  # label -> last sent timestamp
objects_seen_prev = set()

fps_avg = 0.0
frame_count_for_fps = 0
t0 = time.time()

logging.info("Starting detector, press 'q' in the window to quit.")

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            logging.warning("Failed to read frame from capture")
            time.sleep(0.1)
            continue

        processed, counts, detections = detect_objects(frame)
        cur_time = time.time()

        # draw restricted zone
        cv2.rectangle(processed, (ZONE_X1, ZONE_Y1), (ZONE_X2, ZONE_Y2), (255, 0, 0), 2)
        cv2.putText(processed, "Restricted Zone", (ZONE_X1, ZONE_Y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

        # Snapshot when new object types appear (cooldown)
        objects_current = set(counts.keys())
        newly = objects_current - objects_seen_prev
        if newly and (cur_time - last_snapshot_time) > SNAPSHOT_COOLDOWN:
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            filename = SNAPSHOT_DIR / f"snapshot_{timestamp}.jpg"
            # annotate new objects with red box
            snap_img = frame.copy()
            for det in detections:
                x, y, bw, bh = det['box']
                if det['label'] in newly:
                    cv2.rectangle(snap_img, (x, y), (x + bw, y + bh), (0, 0, 255), 3)
                    cv2.putText(snap_img, f"NEW: {det['label']}", (x, y - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.imwrite(str(filename), snap_img)
            # also save JSON metadata
            meta = {"time": timestamp, "new_labels": list(newly), "detections": detections}
            with open(SNAPSHOT_DIR / f"snapshot_{timestamp}.json", "w") as jf:
                json.dump(meta, jf, indent=2)
            logging.info("Snapshot saved: %s", filename)
            last_snapshot_time = cur_time
        objects_seen_prev = objects_current

        # Notification logic: check center point in restricted zone, use per-label cooldown
        for det in detections:
            x, y, bw, bh = det['box']
            cx, cy = x + bw // 2, y + bh // 2
            if ZONE_X1 < cx < ZONE_X2 and ZONE_Y1 < cy < ZONE_Y2:
                label = det['label']
                last_for_label = last_notification_per_label.get(label, 0.0)
                if (cur_time - last_for_label) >= PER_LABEL_COOLDOWN and (cur_time - last_notification_time_global) >= 1.0:
                    caption = f"🚨 ALERT: '{label}' detected in restricted zone at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
                    alert_img = frame.copy()
                    cv2.rectangle(alert_img, (ZONE_X1, ZONE_Y1), (ZONE_X2, ZONE_Y2), (255, 0, 0), 2)
                    cv2.rectangle(alert_img, (x, y), (x + bw, y + bh), (0, 0, 255), 3)
                    cv2.putText(alert_img, f"ALERT: {label}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    sent = send_telegram_photo(caption, alert_img)
                    if sent:
                        last_notification_per_label[label] = cur_time
                        last_notification_time_global = cur_time
                        logging.info("Notification sent for label=%s", label)
                    else:
                        logging.warning("Notification failed for label=%s", label)
                # break? we allow multiple labels in same frame to be checked (no break)

        # overlay counts and FPS
        y_offset = 25
        for name, c in counts.items():
            cv2.putText(processed, f"{name}: {c}", (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            y_offset += 30

        # compute simple FPS (averaged)
        frame_count_for_fps += 1
        if frame_count_for_fps >= 6:
            t1 = time.time()
            fps_avg = frame_count_for_fps / (t1 - t0 + 1e-6)
            frame_count_for_fps = 0
            t0 = t1
        cv2.putText(processed, f"FPS: {fps_avg:.1f}", (480, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        cv2.imshow("YOLO Live Security Feed", processed)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            logging.info("Quit requested by user")
            break

except KeyboardInterrupt:
    logging.info("Interrupted by user")

finally:
    cap.release()
    cv2.destroyAllWindows()
    logging.info("Clean exit")