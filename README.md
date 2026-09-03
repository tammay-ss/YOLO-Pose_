# YOLO-Pose Carina Point Detection

Automated detection of the **Carina** anatomical landmark and tracheal bifurcation points from chest X-ray and angiography images using Ultralytics YOLO-Pose.

---

## Dataset: `dataset_696`

`dataset_696` is the final curated dataset produced through a 5-round active learning pipeline:
- **Curation:** ~2,000 raw medical scans were screened via contact sheets (`contact_sheet.py`) and deduplicated via perceptual hashing (`dedup.py`) to shortlist 292 images. An initial 150 images were labelled in Label Studio, followed by 5 iterative rounds of model predictions, human correction (`convert2p.py`), and retraining to reach **696 clinically validated images**.
- **Data Partition:** 80% Train / 20% Validation split generated via `split.py`.

### Anatomical Keypoints (5 Landmarks)

```
        (4) Centre1 [Upper Trachea]
             |
        (3) Centre2 [Lower Trachea]
             |
        (0) Carina  [Tracheal Bifurcation]
           /       \
 (1) Left           (2) Right
[Left Bronchus]    [Right Bronchus]
```

- **0 — `Carina`**: Tracheal bifurcation apex (primary landmark).
- **1 — `Left`**: Left main bronchus lateral reference.
- **2 — `Right`**: Right main bronchus lateral reference.
- **3 — `Centre2`**: Lower intermediate tracheal midline reference.
- **4 — `Centre1`**: Upper tracheal midline reference.

### Label Format (`.txt`)

Each label file contains a 20-column space-delimited vector:

```text
class_id cx cy bw bh x1 y1 v1 x2 y2 v2 x3 y3 v3 x4 y4 v4 x5 y5 v5
```

- **Bounding Box (`class_id`, `cx`, `cy`, `bw`, `bh`)**: `class_id = 0` (Carina); center coordinates, width, and height normalized to `[0.0, 1.0]`.
- **Keypoints (`x1`–`x5`, `y1`–`y5`)**: Normalized landmark coordinates `[0.0, 1.0]`.
- **Visibility (`v1`–`v5`)**: `2` = visible, `0` = missing / occluded.

---

## Scripts & Pipeline

The codebase includes 11 scripts supporting the end-to-end active learning and training workflow:

| Script | Category | Functionality |
|:---|:---|:---|
| `contact_sheet.py` | Triage | Generates thumbnail contact sheets with auto-contrast for rapid visual screening of raw scans |
| `dedup.py` | Cleaning | Uses perceptual hashing (`pHash`) to remove duplicate and near-duplicate medical images |
| `conversion.py` | Annotation | Converts Label Studio JSON exports to YOLO-Pose `.txt` labels with auto-computed bounding boxes |
| `conversion_2json.py` | Annotation | Converts predicted YOLO-Pose `.txt` files into intermediate structured JSON |
| `convert2p.py` | Active Learning | Converts prediction JSON into Label Studio pre-annotations format (`predictions.json`) with linked keypoints |
| `image_selection.py` | Dataset | Shortlists candidate images and pairs generated `.txt` labels with raw images for new training batches |
| `labell.py` | Dataset | Matches downloaded Label Studio labels to image pools and segregates confirmed labelled images |
| `rename.py` | Maintenance | Strips random hash prefixes from Label Studio export filenames (`hash-imgno.txt` → `imgno.txt`) |
| `rename_images.py` | Maintenance | Cleans copy-paste duplicate names (`1(2).png`) and safely renumbers images/labels sequentially |
| `split.py` | Partitioning | Deterministically splits paired images and labels into Train/Val sets (80/20) with reproducible seed |
| `visualise.py` | QC / Audit | Overlays bounding boxes and 5 colored keypoints on images for visual quality inspection |

---

## Model Evaluation & Metric Comparisons

Two models (**YOLOv26s-Pose** and **YOLOv26m-Pose**) were evaluated on `dataset_696` across **3 experimental settings**:
1. **Baseline 640:** `imgsz = 640`, 100 epochs, default augmentations.
2. **Resolution 768:** `imgsz = 768` (+44% input resolution), 150 epochs.
3. **Augmentation 640:** `imgsz = 640`, 150 epochs, custom spatial augmentations (`degrees=5°`, `translate=0.05`, `scale=0.30`, `fliplr=0.5`).

### Performance Comparison

| Model | Setting | Input Size | Epochs | Pose mAP@50 | Pose mAP@50–95 | Pose Precision | Pose Recall | Box mAP@50 | Carina Detection (CM) |
|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **YOLOv26s-Pose** | **Baseline** | 640 | 100 | **0.428** | **~0.39** | 0.487 | 0.474 | 0.390 | **83%** |
| **YOLOv26m-Pose** | **Baseline** | 640 | 100 | **0.469** | **~0.41** | **0.512** | **0.526** | **0.417** | **82%** |
| **YOLOv26s-Pose** | **Resolution** | 768 | 150 | ~0.42 | ~0.39 | 0.514 | 0.500 | 0.371 | 47% |
| **YOLOv26m-Pose** | **Resolution** | 768 | 150 | ~0.44 | ~0.40 | 0.582 | 0.444 | 0.400 | 48% |
| **YOLOv26s-Pose** | **Augmentation** | 640 | 150 | ~0.41 | ~0.38 | 0.505 | 0.442 | 0.362 | 43% |
| **YOLOv26m-Pose** | **Augmentation** | 640 | 150 | ~0.40 | ~0.38 | 0.503 | 0.469 | 0.386 | 39% |

### Key Findings & Recommendations

- **Model Capacity:** YOLOv26m-Baseline achieved the highest validation performance (mAP@50 = **0.469**, mAP@50–95 = **~0.41**), improving landmark localization over YOLOv26s (mAP@50 = **0.428**).
- **Resolution Scaling:** Increasing resolution to 768 increased compute overhead without improving pose mAP (~0.41–0.44) and degraded Carina detection rates.
- **Augmentation:** Constrained spatial augmentations reduced validation metrics compared to default training settings.
- **Unseen Generalization:** On a separate 50-image unseen test set, **YOLOv26s generalized better** than YOLOv26m, offering a more robust and lightweight option for resource-constrained edge deployments.
- **Deployment Trade-off:**
  - Use **YOLOv26m-Pose (Baseline 640)** for centralized, high-accuracy diagnostic workstations.
  - Use **YOLOv26s-Pose (Baseline 640)** for edge/embedded medical devices requiring fast inference and strong generalization.

---

## Quick Start

### Installation

```bash
git clone https://github.com/tammay-ss/YOLO-Pose_.git
cd YOLO-Pose_
pip install ultralytics pillow imagehash tqdm pyyaml opencv-python
```

### Training

```python
from ultralytics import YOLO

# Train YOLOv26m-Pose Baseline
model = YOLO("yolo26m-pose.pt")
model.train(
    data="dataset_696/data.yaml",
    epochs=100,
    imgsz=640,
    batch=16,
    name="v26m_baseline_640"
)
```

### Inference & Label Studio Pre-annotations

```bash
# 1. Convert predicted YOLO labels to JSON
python conversion_2json.py

# 2. Format predictions for Label Studio import
python convert2p.py
```
