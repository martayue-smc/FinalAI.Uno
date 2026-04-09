import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "uno_model.keras")
LABELS_PATH = os.path.join(BASE_DIR, "uno_labels.txt")

# The frames in uno_cards.json are already 32x32 RGB.
# We upsample to 64x64 so the CNN has more spatial room to learn from.
IMG_SIZE = (64, 64)

# Training
BATCH_SIZE = 32
EPOCHS = 50
LEARNING_RATE = 1e-3
VAL_SPLIT = 0.2  # 20% of data used for validation

# Real-time camera
CAMERA_ID = 1
CONFIDENCE_THRESHOLD = 0.70
ROI_TOP_LEFT = (100, 60)
ROI_BOTTOM_RIGHT = (540, 420)