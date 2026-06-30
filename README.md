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

## Workflow

The dataset began with a pool of roughly **2,000 unlabeled images**. To make manual review feasible at this scale, images were compiled into contact sheets of 10 images each, allowing rapid visual screening. From this pool, **292 images** were selected as relevant candidates, of which **150** had the anatomical structure necessary for Carina point labelling.

These 150 images were manually annotated in Label Studio, exported in COCO Keypoints format, and converted to YOLO Pose format to train an initial YOLO26s-pose model.

The trained model was then run on the remaining unlabeled images from the original pool. Its predictions were converted into Label Studio's import format and brought back in as pre-annotations, which were manually corrected and refined, then exported as updated annotations to retrain the model.

This annotate → train → predict → correct → retrain cycle was repeated across **two rounds**, with the labelled dataset converging at **313 images**, forming the final training set (`dataset_3`).


## Scripts

| Script | Purpose |
|--------|---------|
| `conversion.py` | Converts JSON labels from Label Studio into YOLO pose estimation format |
| `conversion_2json.py` | Converts predicted labels (YOLO format) back into JSON |
| `convert2p.py` | Converts predicted + converted JSON labels into the prediction format accepted by Label Studio |
| `contact_sheet.py` | Converts the pool of 2,000 images into contact sheets of 10 images each, for rapid visual screening |
| `rename_images.py` / `rename.py` | Renames images in sequential order |
| `image_selection.py` | Selects the chosen images from a given folder |
| `split.py` | Splits images into train and validation sets |


