# uno_config.py  -- central settings, imported by all other scripts

# PATHS
DATASET_DIR = "Data"  # folder containing one sub-folder per class
MODEL_PATH = "uno_model.keras"  # saved after training
LABELS_PATH = "uno_labels.txt"  # one label per line, written by uno_train.py

# IMG SETTINGS
IMG_SIZE = (96, 96)   # (height, width) -- must match uno_model.py default

# HYPERPARAMETERS
EPOCHS = 100
BATCH_SIZE = 32
VAL_SPLIT = 0.2
LEARNING_RATE = 0.0005

# CAMERA SETTINGS
CAMERA_ID = 0
CONFIDENCE_THRESHOLD = 0.85  # minimum confidence to print/display a prediction
