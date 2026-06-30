## Selecting the images and copying them into new folders to train
# import os
# import shutil
# from pathlib import Path

# # ── CONFIG ────────────────────────────────────────────────────────────────────
# BASE = Path(__file__).parent  # gets the folder where the script itself lives
# SOURCE_FOLDER = BASE / "images" / "train"
# DEST_FOLDER   = BASE / "images" / "unlabelled_tr"
# IMAGE_EXT     = ".png"                        

# FILE_NAMES = [
#     1002,1003,1011,1022,102,1040,1058,1069,1067,1088,1091,1113,1114,118,1204,1206,1213,124,1254,1255,1262,1260,1289,1293,1302,1315,1329,1342,1355,1356,1354,1371,1381,1382,1390,1388,1403,1428,1451,1450,145,1462,1482,1484,1485,1489,1488,1513,1529,1527,1555,974,978,932,86,843,845,755,72,720,659,467,466,464,391,1568,1590,1591,1627,1704,1711,1730,1917,1930,1950
# ]


# def copy_selected_images():
#     os.makedirs(DEST_FOLDER, exist_ok=True)

#     copied  = 0
#     missing = []

#     for name in FILE_NAMES:
#         filename = f"{name}{IMAGE_EXT}"
#         src = os.path.join(SOURCE_FOLDER, filename)

#         if os.path.isfile(src):
#             shutil.copy2(src, os.path.join(DEST_FOLDER, filename))
#             copied += 1
#         else:
#             missing.append(filename)

#     print(f"Done. Copied: {copied} / {len(FILE_NAMES)}")

#     if missing:
#         print(f"\nMissing files ({len(missing)}):")
#         for f in missing:
#             print(f"  {f}")

# if __name__ == "__main__":
#     copy_selected_images()

## Copying corresponding images from labels for further rounds of training

import os
import shutil
from pathlib import Path

labels_dir = Path("labels_3/train_3")
images_dir = Path("unlabelled_3")
output_images_dir = Path("dataset_t/images")
output_labels_dir = Path("dataset_t/labels")

os.makedirs(output_images_dir, exist_ok=True)
os.makedirs(output_labels_dir, exist_ok=True)

copied = 0
missing = []

for label_file in os.listdir(labels_dir):
    if not label_file.endswith(".txt"):
        continue

    name = os.path.splitext(label_file)[0]  # e.g., "42" from "42.txt"
    image_file = name + ".png"
    src_image = os.path.join(images_dir, image_file)
    src_label = os.path.join(labels_dir, label_file)

    if os.path.exists(src_image):
        shutil.copy2(src_image, os.path.join(output_images_dir, image_file))
        shutil.copy2(src_label, os.path.join(output_labels_dir, label_file))
        copied += 1
    else:
        missing.append(image_file)

print(f"Copied {copied} image-label pairs to dataset_2/")
if missing:
    print(f"Missing images for labels: {missing}")