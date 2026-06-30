#converting json to presdictions format applicable for label-studio
import json
from pathlib import Path

IMAGES_DIR = r"C:\Users\santp\Documents\eikern\predicted_package\images_1"
LABELS_DIR = LABELS_DIR = Path(__file__).parent / "json_labels_1"

# Define your 5 keypoint names in order (match your model's output order)
KEYPOINT_NAMES = ["Carina", "Left", "Right", "Centre2", "Centre1"]

output_tasks = []

image_files = sorted(list(Path(IMAGES_DIR).glob("*")))
print(f"Found {len(image_files)} images")  # add this line

for idx, image_path in enumerate(image_files):

    image_name = image_path.stem
    json_file = Path(LABELS_DIR) / f"{image_name}.json"

    if not json_file.exists():
        continue

    with open(json_file, "r") as f:
        anns = json.load(f)

    results = []

    for ann_idx, ann in enumerate(anns):

        bbox = ann["bbox"]
        keypoints = ann["keypoints"]  # FIX: was never read before

        x = (bbox["x_center"] - bbox["width"] / 2) * 100
        y = (bbox["y_center"] - bbox["height"] / 2) * 100
        w = bbox["width"] * 100
        h = bbox["height"] * 100

        bbox_id = f"bbox_{ann_idx}"

        # Bounding box
        results.append({
            "id": bbox_id,
            "type": "rectanglelabels",
            "from_name": "bbox",
            "to_name": "image",
            "original_width": 1000,
            "original_height": 1000,
            "image_rotation": 0,
            "value": {
                "x": x,
                "y": y,
                "width": w,
                "height": h,
                "rotation": 0,
                "rectanglelabels": ["Carina"]
            }
        })

        # FIX: Add keypoints, linked to their parent bbox via parentID
        for kp_idx, kp in enumerate(keypoints):
            results.append({
                "type": "keypointlabels",
                "from_name": "keypoints",
                "to_name": "image",
                "parentID": bbox_id,  # links keypoint to its bounding box
                "original_width": 1000,
                "original_height": 1000,
                "image_rotation": 0,
                "value": {
                    "x": kp["x"] * 100,   # FIX: YOLO normalized → LS percentage
                    "y": kp["y"] * 100,
                    "width": 0.5,          # dot size in Label Studio
                    "keypointlabels": [KEYPOINT_NAMES[kp_idx]]
                }
            })

    task = {
        "data": {
            "image": f"/data/local-files/?d=eikernd/{image_path.name}"
        },
        "predictions": [
            {
                "model_version": "yolo_pose",
                "score": 0.99,
                "result": results
            }
        ]
    }

    output_tasks.append(task)

with open("predictions.json", "w") as f:
    json.dump(output_tasks, f, indent=4)

print(f"Created predictions.json with {len(output_tasks)} tasks")