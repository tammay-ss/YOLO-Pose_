# YOLO-Pose Carina Point Detection

This repo uses YOLO-Pose Estimation (Ultralytics) to predict Carina points from X-ray and angiography images.

## Dataset

- **`dataset_3`** — the final dataset used for training. Contains **313 labelled images**. This is small for pose estimation in general, but adequate for a medical imaging task.
- YOLOv26s and YOLOv26m were trained **separately** on this dataset.

## Models

| Model | Metrics & Training Notebook |
|-------|------------------------------|
| YOLOv26s | `yolo-26s_metrics/` |
| YOLOv26m | `yolo-26m_metrics/` |

Each folder contains the training loop notebook and the resulting metrics for that model.

**Current result:** based on the metrics, **YOLOv26s performs better** than YOLOv26m.

## Predictions on Unseen Data

Both models were run on **200 unseen images**, with predictions saved to:

- `predicted_images_s/` — YOLOv26s predictions
- `predicted_images_m/` — YOLOv26m predictions

| Model | Images Labelled (out of 200) |
|-------|-------------------------------|
| YOLOv26s | 115 |
| YOLOv26m | 75 |

The difference in count is due to each model's confidence threshold — lower-confidence predictions are not retained as labels.

## Scripts

| Script | Purpose |
|--------|---------|
| `conversion.py` | Converts JSON labels from Label Studio into YOLO pose estimation format |
| `conversion_2json.py` | Converts predicted labels (YOLO format) back into JSON |
| `convert2p.py` | Converts predicted + converted JSON labels into the prediction format accepted by Label Studio |
