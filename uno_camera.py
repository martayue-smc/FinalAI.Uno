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
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    print("[INFO] Camera open.")
    print("[INFO] Press SPACE to recognise card, Q to quit\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        cv2.imshow("UNO Camera (SPACE to scan, Q to quit)", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q') or key == 27:
            break

        elif key == ord(' '):
            # take 5 frames and majority vote for stability
            frames = []
            for _ in range(5):
                ret, f = cap.read()
                if ret:
                    frames.append(f)

            predictions = []
            for f in frames:
                probs = model.predict(preprocess(f), verbose=0)[0]
                predictions.append(int(np.argmax(probs)))

            # majority vote
            smoothed = max(set(predictions), key=predictions.count)
            label = labels[smoothed] if smoothed < len(labels) else "unknown"
            confidence = float(model.predict(preprocess(frames[-1]), verbose=0)[0][smoothed])

            print(f"Recognised: {label} (confidence: {confidence:.0%})")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    run()
