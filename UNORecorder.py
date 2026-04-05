import cv2
import numpy as np
import json
import time

from support_functions import AsciiDecoder


CAMERA_INDEX = 0
SAVE_PATH = "uno_cards.json"
DISPLAY_SCALE = 2
FRAME_SIZE = (64, 64)

# ------------------ LABEL MAPS ------------------------------------------
COLOR_KEYS = {
    'r': 'red',
    'g': 'green',
    'b': 'blue',
    'y': 'yellow',
    'w': 'wild',
}

# Value keys per color type

VALUE_KEYS_NORMAL = {
    '0': '0', '1': '1', '2': '2', '3': '3', '4': '4',
    '5': '5', '6': '6', '7': '7', '8': '8', '9': '9',
    's': 'skip',
    'v': 'reverse',
    'd': 'draw_two',
}

VALUE_KEYS_WILD = {
    'w': 'wild', # plain wild
    'f': 'wild_draw_four', # draw 4
}

# -----------------------------------------------------------------


data = []
current_color = None
current_label = None
new_card = False

print("Two-step labeling:")
print("Step 1 (color): r=red | g=green | b=blue | y=yellow | w=wild")
print("Step 2 (value): 0-9=number | s=skip | v=reverse | d=draw_two")
print("       (wild only): w=wild | f=wild_draw_four")
print()
print(" c -> clear current label")
print(" q -> quit and save uno_cards.json")

cap = cv2.VideoCapture(CAMERA_INDEX)

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    display_image =cv2.resize(
        frame,
        None,
        fx=DISPLAY_SCALE,
        fy=DISPLAY_SCALE,
        interpolation=cv2.INTER_LINEAR_EXACT,
    )



# Overlay current state
if current_color is None:
    status = "Waiting for color key..."
elif current_label is None:
    status = f"Color: {current_color} | Waiting for value key..."
else:
    status = f"Recording: {current_label} ({len(data)} frames saved)"

cv2.putText(
    display_image, status,
    (12, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2, cv2.LINE_AA
)

cv2.imshow("UNO Recorder", display_image)



# Key handling
key = AsciiDecoder(cv2.waitKey(1))

if key == 'q':
    break

if key == 'c':
    current_color = None
    current_label = None
    new_card = False
    print("Label cleared.")
    continue

# Step 1: color key
if key in COLOR_KEYS and current_color is None:
    current_color = COLOR_KEYS[key]
    print(f" Color set: {current_color}, now press value key")
    continue

# Step 2: value key
if current_color is not None and current_label is None:
    valid_values = VALUE_KEYS_WILD if current_color == 'wild' else VALUE_KEYS_NORMAL

    if key in valid_values:
        value = valid_values[key]
        # Wild cards carry no color prefix
        if current_color == 'wild':
            current_label = value # -> e.g. "wild_draw_four"
        else:
            current_label = f"{current_color}_{value}" # -> e.g. "red_5"
        new_card = True
        print(f" Label set: {current_label} - recording frames (press 'c' to continue)")
    else:
        print(f" Invalid value key '{key}' for color '{current_color}'. Try again.")
    continue



# Record frame
if current_label is not None:
    small = cv2.resize(frame, FRAME_SIZE, interpolation=cv2.INTER_AREA)
    entry = {
        "time": time.time(),
        "label": current_label,
        "new_card": new_card, # True only on first frame after label is set
        "frame": small.tolist(),
    }
    data.append(entry)
    new_card = False # only the first frame is marked new



# Save JSON
print(f"\nSaving {SAVE_PATH}...")
with open(SAVE_PATH, "wb") as f:
    json.dump(data, f, indent=1)

print(f"Saved {len(data)} frames across all cards.")

cap.release()
cv2.destroyAllWindows()




