#converting yolo to json format
import os
import json
from pathlib import Path

LABELS_DIR = Path("predicted_package\labels")
OUTPUT_DIR = Path("json_labels_1")

NUM_KEYPOINTS = 5
MAX_FILES = 200

os.makedirs(OUTPUT_DIR, exist_ok=True)

txt_files = sorted(Path(LABELS_DIR).glob("*.txt"))[:MAX_FILES]

print(f"Found {len(txt_files)} txt files")

for txt_file in txt_files:
    annotations = []

    with open(txt_file, "r") as f:
        lines = f.readlines()

    for line in lines:
        values = line.strip().split()

        expected_len = 5 + (NUM_KEYPOINTS * 3)

        if len(values) < expected_len:
            print(f"Skipping invalid label in {txt_file.name}")
            continue

        class_id = int(values[0])

        bbox = {
            "x_center": float(values[1]),
            "y_center": float(values[2]),
            "width": float(values[3]),
            "height": float(values[4])
        }

        keypoints = []

        kp_values = values[5:]

        for i in range(NUM_KEYPOINTS):
            x = float(kp_values[i * 3])
            y = float(kp_values[i * 3 + 1])
            v = int(float(kp_values[i * 3 + 2]))

            keypoints.append({
                "x": x,
                "y": y,
                "visibility": v
            })

        annotations.append({
            "class_id": class_id,
            "bbox": bbox,
            "keypoints": keypoints
        })

    output_file = os.path.join(
        OUTPUT_DIR,
        txt_file.stem + ".json"
    )

    with open(output_file, "w") as jf:
        json.dump(annotations, jf, indent=4)

    print(f"Converted: {txt_file.name}")

print("Finished converting 55 files.")