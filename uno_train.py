import json
import os
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


def load_data(dataset_dir=cfg.DATASET_DIR):
    print(f"[INFO] Loading images from '{dataset_dir}' ...")

    target_h, target_w = cfg.IMG_SIZE

    images = []
    labels = []

    class_folders = sorted([
        d for d in os.listdir(dataset_dir)
        if os.path.isdir(os.path.join(dataset_dir, d))
    ])

    print(f"[INFO] Found {len(class_folders)} classes: {class_folders}")

    for class_name in class_folders:
        class_path = os.path.join(dataset_dir, class_name)

        img_files = [
            f for f in os.listdir(class_path)
            if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".HEIC"))
        ]

        for fname in img_files:
            fpath = os.path.join(class_path, fname)
            img = cv2.imread(fpath)

            if img is None:
                continue

            img = cv2.resize(img, (target_w, target_h))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            images.append(img)
            labels.append(class_name)

    print(f"[INFO] Loaded {len(images)} images total.")

    X = np.array(images, dtype=np.float32) / 255.0

    le = LabelEncoder()
    y = le.fit_transform(labels)
    label_names = list(le.classes_)

    print(f"[INFO] Classes: {label_names}")

    return X, y, label_names


# Augmentation
def augment_batch(X_batch):
    aug = []
    for img in X_batch:
        h, w = img.shape[:2]

        # mild brightness
        delta = np.random.uniform(-0.1, 0.1)
        img = np.clip(img + delta, 0.0, 1.0)

        # mild rotation
        angle = np.random.uniform(-10, 10)
        M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
        img = cv2.warpAffine(img, M, (w, h))

        aug.append(img)
    return np.array(aug, dtype=np.float32)


def batch_generator(X, y_cat, batch_size, augment=False):
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


# TRAIN
def train(dataset_dir=cfg.DATASET_DIR):
    X, y, label_names = load_data(dataset_dir)
    num_classes = len(label_names)

    # Save label list so uno_camera.py can load it
    with open(cfg.LABELS_PATH, "w") as f:
        for name in label_names:
            f.write(name + "\n")
    print(f"[INFO] Labels saved -> {cfg.LABELS_PATH}")

    # Stratified train/val split
    X_train, X_val, y_train, y_val = train_test_split(
        X, y,
        test_size=cfg.VAL_SPLIT,
        random_state=42,
        stratify=y
    )
    print(f"[INFO] Train: {len(X_train)}  Val: {len(X_val)}")

    y_train_cat = to_categorical(y_train, num_classes)
    y_val_cat = to_categorical(y_val, num_classes)

    # Build and compile model
    model = build_model(num_classes, cfg.IMG_SIZE)
    model.summary(line_length=80)
    model.compile(
        optimizer=Adam(cfg.LEARNING_RATE),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    steps_train = int(np.ceil(len(X_train) / cfg.BATCH_SIZE))
    steps_val = int(np.ceil(len(X_val) / cfg.BATCH_SIZE))

    train_gen = batch_generator(X_train, y_train_cat, cfg.BATCH_SIZE, augment=True)
    val_gen = batch_generator(X_val, y_val_cat, cfg.BATCH_SIZE, augment=False)

    callbacks = [
        EarlyStopping(patience=20, restore_best_weights=True, verbose=1),
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
    present_labels = np.unique(y_val)
    present_names = [label_names[i] for i in present_labels]
    print(classification_report(y_val, y_pred, labels=present_labels, target_names=present_names))

    # Training curves
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history["accuracy"], label="Train")
    plt.plot(history.history["val_accuracy"], label="Val")
    plt.title("Accuracy");
    plt.legend();
    plt.grid(alpha=0.3)

    plt.subplot(1, 2, 2)
    plt.plot(history.history["loss"], label="Train")
    plt.plot(history.history["val_loss"], label="Val")
    plt.title("Loss");
    plt.legend();
    plt.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig("training_curves.png", dpi=120)
    plt.show()
    print(f"[DONE] Model saved -> {cfg.MODEL_PATH}")


if __name__ == "__main__":
    train()
