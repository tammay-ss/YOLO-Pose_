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

### YOLOv26s-Pose

YOLOv26s-Pose achieved a **mAP@0.50 of 0.428**, **mAP@0.50–0.95 of approximately 0.39**, and a peak **F1-score of 0.45** with a recall of **0.77**. The training and validation losses showed consistent convergence, indicating stable learning. With approximately **83% of Carina instances correctly detected**, the model provides a competitive lightweight baseline for Carina keypoint detection.

### YOLOv26m-Pose

YOLOv26m-Pose achieved stronger overall performance, with a **mAP@0.50 of 0.469**, **mAP@0.50–0.95 of approximately 0.41**, and a peak **F1-score of 0.49** with a recall of **0.78**. The model demonstrated stable convergence throughout training, with progressively improving validation metrics. Although its confusion matrix showed a similar Carina detection rate of approximately **82%**, its higher mAP and F1-score indicate improved overall localization and detection performance.

### Model Comparison and Trade-off

YOLOv26m-Pose provides **higher detection and localization accuracy**, improving mAP@0.50 from **0.428 to 0.469** and F1-score from **0.45 to 0.49** compared with YOLOv26s-Pose. However, this improvement comes at the cost of a larger model and higher computational requirements. YOLOv26s-Pose therefore offers a better **accuracy–efficiency trade-off** for resource-constrained deployment, whereas **YOLOv26m-Pose is the preferred model when prediction accuracy is the primary objective**.
