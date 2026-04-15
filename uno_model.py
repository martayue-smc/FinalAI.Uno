from tensorflow.keras import models, layers

def build_model(num_classes, img_size):
    h, w = img_size

    model = models.Sequential([
        layers.Conv2D(32, (3, 3), padding="same", activation="relu", input_shape=(h, w, 3)),
        layers.MaxPooling2D(),
        layers.Dropout(0.3),
        layers.Conv2D(64, (3, 3), padding="same", activation="relu"),
        layers.MaxPooling2D(),
        layers.Dropout(0.3),
        layers.Conv2D(128, (3, 3), padding="same", activation="relu"),
        layers.MaxPooling2D(),
        layers.Dropout(0.4),
        layers.Flatten(),
        layers.Dense(256, activation="relu"),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation="softmax")
    ])
    return model