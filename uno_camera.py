
# Run AFTER training:
#   python uno_camera.py
# Controls:
#   Q / ESC -- quit
#   S       -- save a snapshot

import cv2
import numpy as np
import time
from pathlib import Path

from tensorflow.keras.models import load_model

import uno_config as cfg


def load():
    if not Path(cfg.MODEL_PATH).exists():
        raise FileNotFoundError(f"Model not found: {cfg.MODEL_PATH}\nRun uno_train.py first.")
    if not Path(cfg.LABELS_PATH).exists():
        raise FileNotFoundError(f"Labels not found: {cfg.LABELS_PATH}\nRun uno_train.py first.")

    model = load_model(cfg.MODEL_PATH)
    with open(cfg.LABELS_PATH) as f:
        labels = [line.strip() for line in f if line.strip()]

    print(f"[INFO] Model loaded -- {len(labels)} classes")
    return model, labels


def preprocess(roi_bgr):
    img = cv2.resize(roi_bgr, cfg.IMG_SIZE, interpolation=cv2.INTER_LANCZOS4)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype("float32") / 255.0
    return np.expand_dims(img, 0)


def card_color(label):
    l = label.lower()
    if l.startswith("red"):    return (50, 50, 210)
    if l.startswith("blue"):   return (210, 100, 30)
    if l.startswith("green"):  return (50, 180, 50)
    if l.startswith("yellow"): return (30, 200, 230)
    return (180, 80, 180)  # wild


def draw(frame, label, confidence, top3, fps, roi_tl, roi_br):
    h, w = frame.shape[:2]
    confident = confidence >= cfg.CONFIDENCE_THRESHOLD
    color = card_color(label) if confident else (100, 100, 100)

    # ROI box + corner accents
    cv2.rectangle(frame, roi_tl, roi_br, color, 2)
    arm = 18
    for (cx, cy), (dx, dy) in [
        (roi_tl, (1, 1)), (roi_br, (-1, -1)),
        ((roi_br[0], roi_tl[1]), (-1, 1)),
        ((roi_tl[0], roi_br[1]), (1, -1)),
    ]:
        cv2.line(frame, (cx, cy), (cx + dx * arm, cy), color, 3)
        cv2.line(frame, (cx, cy), (cx, cy + dy * arm), color, 3)

    # Label above the box
    display = label.replace("_", " ").upper() if confident else "Align card in box..."
    (tw, th), _ = cv2.getTextSize(display, cv2.FONT_HERSHEY_DUPLEX, 0.85, 2)
    py = roi_tl[1] - 12
    overlay = frame.copy()
    cv2.rectangle(overlay, (roi_tl[0], py - th - 8), (roi_tl[0] + tw + 16, py + 4), color, -1)
    cv2.addWeighted(overlay, 0.72, frame, 0.28, 0, frame)
    cv2.putText(frame, display, (roi_tl[0] + 6, py - 2),
                cv2.FONT_HERSHEY_DUPLEX, 0.85, (255, 255, 255), 2, cv2.LINE_AA)

    # Confidence bar below the box
    bar_y = roi_br[1] + 10
    bar_w = roi_br[0] - roi_tl[0]
    cv2.rectangle(frame, (roi_tl[0], bar_y), (roi_br[0], bar_y + 8), (50, 50, 50), -1)
    cv2.rectangle(frame, (roi_tl[0], bar_y),
                  (roi_tl[0] + int(bar_w * confidence), bar_y + 8), color, -1)
    cv2.putText(frame, f"{confidence * 100:.1f}%", (roi_br[0] + 6, bar_y + 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.44, (180, 180, 180), 1, cv2.LINE_AA)

    # Top-3 predictions (bottom-left)
    for rank, (lbl, conf) in enumerate(reversed(top3)):
        c = color if rank == len(top3) - 1 else (140, 140, 140)
        cv2.putText(frame,
                    f"#{len(top3) - rank}  {lbl.replace('_', ' '):<22}  {conf * 100:.1f}%",
                    (10, h - 20 - rank * 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.46, c, 1, cv2.LINE_AA)

    # FPS + hints
    cv2.putText(frame, f"FPS {fps:.1f}", (w - 90, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.52, (160, 160, 160), 1, cv2.LINE_AA)
    cv2.putText(frame, "[Q] Quit  [S] Snapshot", (10, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.46, (120, 120, 120), 1, cv2.LINE_AA)
    return frame


def run():
    model, labels = load()

    cap = cv2.VideoCapture(cfg.CAMERA_ID)
    if not cap.isOpened():
        raise RuntimeError(
            f"Cannot open camera (ID={cfg.CAMERA_ID}). "
            "Change CAMERA_ID in uno_config.py."
        )
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    roi_tl, roi_br = cfg.ROI_TOP_LEFT, cfg.ROI_BOTTOM_RIGHT
    frame_times = []
    pred_buffer = []
    SMOOTH_N = 5
    snap_count = 0

    cv2.namedWindow("UNO Classifier", cv2.WINDOW_NORMAL)
    print("[INFO] Camera open. Press Q to quit, S to save snapshot.")

    while True:
        t0 = time.perf_counter()
        ret, frame = cap.read()
        if not ret:
            continue

        x1, y1 = roi_tl
        x2, y2 = roi_br
        roi = frame[y1:y2, x1:x2]

        probs = model.predict(preprocess(roi), verbose=0)[0]

        # Smooth via majority vote over last SMOOTH_N frames
        top_idx = int(np.argmax(probs))
        pred_buffer.append(top_idx)
        if len(pred_buffer) > SMOOTH_N:
            pred_buffer.pop(0)
        smoothed = max(set(pred_buffer), key=pred_buffer.count)

        label = labels[smoothed] if smoothed < len(labels) else "unknown"
        confidence = float(probs[smoothed])
        top3_idx = np.argsort(probs)[-3:][::-1]
        top3 = [(labels[i] if i < len(labels) else "?", float(probs[i]))
                for i in top3_idx]

        frame_times.append(time.perf_counter() - t0)
        if len(frame_times) > 30:
            frame_times.pop(0)
        fps = 1.0 / (sum(frame_times) / len(frame_times))

        frame = draw(frame, label, confidence, top3, fps, roi_tl, roi_br)
        cv2.imshow("UNO Classifier", frame)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            break
        elif key == ord("s"):
            fname = f"snapshot_{snap_count:03d}.png"
            cv2.imwrite(fname, frame)
            print(f"[INFO] Saved -> {fname}")
            snap_count += 1

    cap.release()
    cv2.destroyAllWindows()
    print("[INFO] Done.")


if __name__ == "__main__":
    run()