#split into train and val: 90/10 or 80/20 for the first time when labels had names: hash_imgno.png
# from pathlib import Path
# import random
# import shutil

# random.seed(42)

# images_dir = Path("labelled")
# labels_dir = Path("labels/train_1")

# output_images_train = Path("labelled/train")
# output_images_val = Path("labelled/val")

# output_labels_train = Path("labels/train")
# output_labels_val = Path("labels/val")

# output_images_train.mkdir(parents=True, exist_ok=True)
# output_images_val.mkdir(parents=True, exist_ok=True)

# output_labels_train.mkdir(parents=True, exist_ok=True)
# output_labels_val.mkdir(parents=True, exist_ok=True)

# label_files = list(labels_dir.glob("*.txt"))

# pairs = []

# for label_file in label_files:
#     stem = label_file.stem

#     # label format example:
#     # image-123.txt  -> corresponding image: 123.jpg/png/etc

#     img_number = stem.split("-")[-1]

#     image_file = None

#     for ext in [".jpg", ".jpeg", ".png", ".bmp"]:
#         candidate = images_dir / f"{img_number}{ext}"
#         if candidate.exists():
#             image_file = candidate
#             break

#     if image_file is not None:
#         pairs.append((image_file, label_file))
#     else:
#         print(f"Image not found for {label_file.name}")

# random.shuffle(pairs)

# split_idx = int(len(pairs) * 0.9)

# train_pairs = pairs[:split_idx]
# val_pairs = pairs[split_idx:]

# for image_file, label_file in train_pairs:
#     shutil.copy(image_file, output_images_train / image_file.name)
#     shutil.copy(label_file, output_labels_train / label_file.name)

# for image_file, label_file in val_pairs:
#     shutil.copy(image_file, output_images_val / image_file.name)
#     shutil.copy(label_file, output_labels_val / label_file.name)

# print(f"Total pairs: {len(pairs)}")
# print(f"Train pairs: {len(train_pairs)}")
# print(f"Val pairs: {len(val_pairs)}")

#when names are sequential
import os
import shutil
import random
from pathlib import Path

images_folder = Path("dataset_3/images")
labels_folder = Path("dataset_3/labels")

train_images = Path("dataset_3/images/train")
train_labels = Path("dataset_3/labels/train")
val_images = Path("dataset_3/images/val")
val_labels = Path("dataset_3/labels/val")

for folder in [train_images, train_labels, val_images, val_labels]:
    folder.mkdir(parents=True, exist_ok=True)

extensions = [".png", ".jpg", ".jpeg", ".bmp"]

# only add images that HAVE a corresponding label
paired = []
for ext in extensions:
    for img in images_folder.glob(f"*{ext}"):
        label = labels_folder / f"{img.stem}.txt"
        if label.exists():
            paired.append((img, label))

random.seed(42)
random.shuffle(paired)

split = int(0.8 * len(paired))
train_pairs = paired[:split]
val_pairs = paired[split:]

for img, label in train_pairs:
    shutil.copy2(img, train_images / img.name)
    shutil.copy2(label, train_labels / label.name)

for img, label in val_pairs:
    shutil.copy2(img, val_images / img.name)
    shutil.copy2(label, val_labels / label.name)

print(f"Total paired : {len(paired)}")
print(f"Train        : {len(train_pairs)}")
print(f"Val          : {len(val_pairs)}")