#renaming of duplicate names that are with a (2) due to copy paste

# import os
# import re

# folder = "C:/Users/santp/Documents/eikern/images/test"

# # Collect all files with pattern like "1(2).jpg", "500(2).png", etc.
# duplicate_files = []

# for filename in os.listdir(folder):
#     name, ext = os.path.splitext(filename)
#     match = re.fullmatch(r'(\d+)\s*\(2\)', name)  # \s* handles optional space before (2)
#     if match:
#         original_num = int(match.group(1))
#         duplicate_files.append((filename, original_num, ext))

# duplicate_files.sort(key=lambda x: x[1])

# for filename, original_num, ext in duplicate_files:
#     new_num = original_num + 1000
#     new_filename = f"{new_num}{ext}"
    
#     src = os.path.join(folder, filename)
#     dst = os.path.join(folder, new_filename)
    
#     os.rename(src, dst)
#     print(f"Renamed: {filename} -> {new_filename}")

# print(f"\nDone! Renamed {len(duplicate_files)} files.")

#renaming all scattered images sequentially for better understanding
from pathlib import Path

images_folder = Path("dataset_3/images")
labels_folder = Path("dataset_3/labels")

extensions = [".png", ".jpg", ".jpeg", ".bmp"]

files = []

for ext in extensions:
    files.extend(images_folder.glob(f"*{ext}"))

files = sorted(files)

# STEP 1: rename temporarily (images + labels)
temp_files = []

for i, file in enumerate(files):
    temp_name = images_folder / f"temp_{i}{file.suffix}"
    file.rename(temp_name)
    temp_files.append(temp_name)

    # rename corresponding label if it exists
    label = labels_folder / f"{file.stem}.txt"
    if label.exists():
        temp_label = labels_folder / f"temp_{i}.txt"
        label.rename(temp_label)

# STEP 2: rename sequentially (images + labels)
for i, file in enumerate(temp_files, start=1):
    new_name = images_folder / f"{i}{file.suffix.lower()}"
    file.rename(new_name)

    temp_label = labels_folder / f"temp_{file.stem.split('_')[1]}.txt"
    if temp_label.exists():
        new_label = labels_folder / f"{i}.txt"
        temp_label.rename(new_label)

    print(f"{file.name} -> {new_name.name}")