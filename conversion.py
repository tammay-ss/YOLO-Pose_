import json
from pathlib import Path

JSON_PATH = Path("project-17-at-2026-06-30-15-57-17587ff7.json")
OUTPUT_DIR = Path("labels_3/train_3")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

KEYPOINT_ORDER = ["Carina", "Left", "Right", "Centre1", "Centre2"]
CLASS_ID = 0
PADDING = 0.02

with open(JSON_PATH, "r") as f:
    data = json.load(f)

converted = 0
skipped = 0

def extract_keypoints(results):
    """Extract keypoints dict from a result list. Returns {} if none found."""
    keypoints = {}
    for r in results:
        if r.get("type") != "keypointlabels":
            continue
        label = r["value"]["keypointlabels"][0]
        x = r["value"]["x"] / 100.0
        y = r["value"]["y"] / 100.0
        keypoints[label] = (x, y)
    return keypoints


def find_keypoints(task):
    """
    Search all annotations (all of them, not just [0]),
    then fall back to predictions — but only if they have keypoints.
    Returns (keypoints_dict, source_str) or (None, None).
    """
    # 1. Search every annotation for keypoints
    for ann in task.get("annotations", []):
        kp = extract_keypoints(ann.get("result", []))
        if kp:
            return kp, "annotation"

    # 2. Search every annotation's embedded prediction for keypoints
    for ann in task.get("annotations", []):
        pred = ann.get("prediction", {})
        kp = extract_keypoints(pred.get("result", []))
        if kp:
            return kp, "prediction (embedded)"

    # 3. Search top-level predictions
    for pred in task.get("predictions", []):
        kp = extract_keypoints(pred.get("result", []))
        if kp:
            return kp, "prediction (top-level)"

    return None, None


for task in data:
    image_url = task["data"]["image"]
    image_name = Path(image_url.split("?d=")[-1]).name
    stem = Path(image_name).stem

    keypoints, source = find_keypoints(task)

    if not keypoints:
        print(f"SKIP — no keypoints found anywhere: {image_name}")
        skipped += 1
        continue

    # Build ordered keypoint list
    ordered_points = []
    for kp_name in KEYPOINT_ORDER:
        if kp_name not in keypoints:
            print(f"  WARNING: '{kp_name}' missing in {image_name}, using 0 0 0")
            ordered_points.append(None)
        else:
            ordered_points.append(keypoints[kp_name])

    valid_points = [p for p in ordered_points if p is not None]

    if not valid_points:
        print(f"SKIP — all keypoints missing: {image_name}")
        skipped += 1
        continue

    # Compute bbox from keypoint extents
    xs = [p[0] for p in valid_points]
    ys = [p[1] for p in valid_points]
    x_min = max(0.0, min(xs) - PADDING)
    y_min = max(0.0, min(ys) - PADDING)
    x_max = min(1.0, max(xs) + PADDING)
    y_max = min(1.0, max(ys) + PADDING)

    bbox_x = (x_min + x_max) / 2
    bbox_y = (y_min + y_max) / 2
    bbox_w = x_max - x_min
    bbox_h = y_max - y_min

    # Build YOLO line: class cx cy bw bh [x y v] x5
    yolo_line = [
        str(CLASS_ID),
        f"{bbox_x:.6f}",
        f"{bbox_y:.6f}",
        f"{bbox_w:.6f}",
        f"{bbox_h:.6f}",
    ]

    for point in ordered_points:
        if point is None:
            yolo_line.extend(["0.000000", "0.000000", "0"])
        else:
            yolo_line.extend([f"{point[0]:.6f}", f"{point[1]:.6f}", "2"])

    # Sanity check — must be exactly 20 columns
    assert len(yolo_line) == 20, f"BUG: {len(yolo_line)} cols for {image_name}"

    output_path = OUTPUT_DIR / (stem + ".txt")
    with open(output_path, "w") as out_file:
        out_file.write(" ".join(yolo_line) + "\n")

    converted += 1

print(f"\nDone.")
print(f"  Converted : {converted}")
print(f"  Skipped   : {skipped}")