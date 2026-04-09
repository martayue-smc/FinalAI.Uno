import json
import numpy as np
import cv2
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report

import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.utils import to_categorical

import uno_config as cfg
from uno_model import build_model


# Load and prepare data directly from uno_cards.json
def load_data(json_path="uno_cards.json"):
    print(f"[INFO] Loading {json_path} ...")
    with open(json_path) as f:
        records = json.load(f)

    print(f"[INFO] {len(records)} records found.")

    target_h, target_w = cfg.IMG_SIZE   # e.g. (64, 64)

    images = []
    labels = []

    for rec in records:
        # Convert the nested list to a numpy array: shape (32, 32, 3)
        frame = np.array(rec["frame"], dtype=np.uint8)

        # Upsample to target size (lanczos gives sharper results for small->large)
        if frame.shape[:2] != (target_h, target_w):
            frame = cv2.resize(frame, (target_w, target_h), interpolation=cv2.INTER_LANCZOS4)

        images.append(frame)
        labels.append(rec["label"])

    X = np.array(images, dtype=np.float32) / 255.0   # normalise to [0, 1]

    # Encode string labels to integers
    le = LabelEncoder()
    y = le.fit_transform(labels)
    label_names = list(le.classes_)

    print(f"[INFO] Classes ({len(label_names)}): {label_names}")
    return X, y, label_names


# Simple numpy-based augmentation applied on-the-fly per batch
def augment_batch(X_batch):
    """
    Apply random augmentations to a batch of images.
    Keeps augmentation light because the source frames are only 32x32.
    """
    aug = []
    for img in X_batch:
        # Random horizontal flip (disabled -- 6 vs 9 matters)
        # Random brightness shift
        delta = np.random.uniform(-0.15, 0.15)
        img = np.clip(img + delta, 0.0, 1.0)
        # Random small rotation via OpenCV
        angle = np.random.uniform(-12, 12)
        h, w = img.shape[:2]
        M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
        img = cv2.warpAffine(img, M, (w, h))
        aug.append(img)
    return np.array(aug, dtype=np.float32)


def batch_generator(X, y_cat, batch_size, augment=False):
    """Yields (X_batch, y_batch) pairs, shuffling each epoch."""
    n = len(X)
    indices = np.arange(n)
    while True:
        np.random.shuffle(indices)
        for start in range(0, n, batch_size):
            idx = indices[start:start + batch_size]
            X_batch = X[idx]
            if augment:
                X_batch = augment_batch(X_batch)
            yield X_batch, y_cat[idx]


# Train
def train(json_path="uno_cards.json"):
    X, y, label_names = load_data(json_path)
    num_classes = len(label_names)

    # Save label list so uno_camera.py can load it
    with open(cfg.LABELS_PATH, "w") as f:
        for name in label_names:
            f.write(name + "\n")
    print(f"[INFO] Labels saved -> {cfg.LABELS_PATH}")

    # Train / val split (stratified so every class appears in both sets)
    X_train, X_val, y_train, y_val = train_test_split(
        X, y,
        test_size=cfg.VAL_SPLIT,
        random_state=42,
        stratify=y,
    )
    print(f"[INFO] Train: {len(X_train)}  Val: {len(X_val)}")

    y_train_cat = to_categorical(y_train, num_classes)
    y_val_cat   = to_categorical(y_val,   num_classes)

    # Build and compile model
    model = build_model(num_classes, cfg.IMG_SIZE)
    model.summary(line_length=80)
    model.compile(
        optimizer=Adam(cfg.LEARNING_RATE),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    # Steps per epoch
    steps_train = int(np.ceil(len(X_train) / cfg.BATCH_SIZE))
    steps_val   = int(np.ceil(len(X_val)   / cfg.BATCH_SIZE))

    train_gen = batch_generator(X_train, y_train_cat, cfg.BATCH_SIZE, augment=True)
    val_gen   = batch_generator(X_val,   y_val_cat,   cfg.BATCH_SIZE, augment=False)

    callbacks = [
        EarlyStopping(patience=10, restore_best_weights=True, verbose=1),
        ModelCheckpoint(cfg.MODEL_PATH, save_best_only=True, verbose=1),
        ReduceLROnPlateau(factor=0.5, patience=5, min_lr=1e-6, verbose=1),
    ]

    history = model.fit(
        train_gen,
        steps_per_epoch=steps_train,
        epochs=cfg.EPOCHS,
        validation_data=val_gen,
        validation_steps=steps_val,
        callbacks=callbacks,
    )

    # Classification report
    print("\n[INFO] Evaluating on validation set...")
    y_pred = np.argmax(model.predict(X_val, verbose=0), axis=1)
    print(classification_report(y_val, y_pred, target_names=label_names))

    # Training curves
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history["accuracy"],     label="Train")
    plt.plot(history.history["val_accuracy"], label="Val")
    plt.title("Accuracy"); plt.legend(); plt.grid(alpha=0.3)

    plt.subplot(1, 2, 2)
    plt.plot(history.history["loss"],     label="Train")
    plt.plot(history.history["val_loss"], label="Val")
    plt.title("Loss"); plt.legend(); plt.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig("training_curves.png", dpi=120)
    plt.show()
    print(f"[DONE] Model saved -> {cfg.MODEL_PATH}")


if __name__ == "__main__":
    train("uno_cards.json")