# Run AFTER training
# Controls:
#   Q / ESC -- quit

import cv2
import numpy as np
from pathlib import Path
from tensorflow.keras.models import load_model

import uno_config as cfg


def load():
    if not Path(cfg.MODEL_PATH).exists():
        raise FileNotFoundError(f"Model not found at {cfg.MODEL_PATH}\nRun uno_train.py first.")
    if not Path(cfg.LABELS_PATH).exists():
        raise FileNotFoundError(f"Labels not found at {cfg.LABELS_PATH}\nRun uno_train.py first.")

    model  = load_model(cfg.MODEL_PATH)
    with open(cfg.LABELS_PATH) as f:
        labels = [line.strip() for line in f if line.strip()]

    print(f"[INFO] Model loaded -- {len(labels)} classes")
    return model, labels


def preprocess(frame_bgr):
    img = cv2.resize(frame_bgr, cfg.IMG_SIZE, interpolation=cv2.INTER_LANCZOS4)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype("float32") / 255.0
    return np.expand_dims(img, 0)


def run():
    model, labels = load()

    cap = cv2.VideoCapture(cfg.CAMERA_ID)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera {cfg.CAMERA_ID}. "
                           f"Change CAMERA_ID in uno_config.py")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    pred_buffer = []
    SMOOTH_N    = 5
    last_printed = None

    print("[INFO] Camera open. Press Q to quit")
    print("[INFO] Place a card in front of the camera...\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        probs   = model.predict(preprocess(frame), verbose=0)[0]
        top_idx = int(np.argmax(probs))

        # Smooth predictions via majority vote over last SMOOTH_N frames
        pred_buffer.append(top_idx)
        if len(pred_buffer) > SMOOTH_N:
            pred_buffer.pop(0)
        smoothed = max(set(pred_buffer), key=pred_buffer.count)

        label      = labels[smoothed] if smoothed < len(labels) else "unknown"
        confidence = float(probs[smoothed])

        # Only print when confident and when the label changes
        if confidence >= cfg.CONFIDENCE_THRESHOLD and label != last_printed:
            print(label)
            last_printed = label

        cv2.imshow("UNO Camera (Q to quit)", frame)
        if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    run()
