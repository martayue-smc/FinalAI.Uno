from tensorflow.keras import layers, Model

def build_model(num_classes: int, img_size=(128, 128)) -> Model:
    inputs = layers.Input(shape=(*img_size, 3), name="input")

    # Block 1 -> Learns low-level features: edges, color regions
    x = layers.Conv2D(32, (3, 3), padding="same", name="conv1")(inputs)
    x = layers.BatchNormalization(name="bn1")(x)
    x = layers.Activation("relu", name="relu1")(x)
    x = layers.MaxPooling2D((2, 2), name="pool1")(x)

    # Block 2 -> Learns mid-level features: digit strokes, card borders
    x = layers.Conv2D(64, (3, 3), padding="same", name="conv2")(x)
    x = layers.BatchNormalization(name="bn2")(x)
    x = layers.Activation("relu", name="relu2")(x)
    x = layers.MaxPooling2D((2, 2), name="pool2")(x)

    # Block 3 -> Learns higher features: number shapes, Skip/Reverse symbols
    x = layers.Conv2D(128, (3, 3), padding="same", name="conv3")(x)
    x = layers.BatchNormalization(name="bn3")(x)
    x = layers.Activation("relu", name="relu3")(x)
    x = layers.MaxPooling2D((2, 2), name="pool3")(x)

    # Block 4 -> Combines features into full card representations
    x = layers.Conv2D(256, (3, 3), padding="same", name="conv4")(x)
    x = layers.BatchNormalization(name="bn4")(x)
    x = layers.Activation("relu", name="relu4")(x)
    x = layers.MaxPooling2D((2, 2), name="pool4")(x)

    # Classification head
    x = layers.GlobalAveragePooling2D(name="gap")(x)
    x = layers.Dense(128, activation="relu", name="dense1")(x)
    x = layers.Dropout(0.5, name="dropout")(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="predictions")(x)

    return Model(inputs=inputs, outputs=outputs, name="UNO_CNN")


if __name__ == "__main__":
    model = build_model(num_classes=54)
    model.summary(line_length=80)



