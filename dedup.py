# to reduce from 3k by removing duplicates and blurs
from PIL import Image
import imagehash
from pathlib import Path
import shutil
from tqdm import tqdm

# Input images
input_dir = Path("images/train")

# Output deduplicated images
output_dir = Path("dedup")

output_dir.mkdir(exist_ok=True)

hashes = {}

THRESHOLD = 5

# collect all image files
files = []
extensions = ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tif"]

for ext in extensions:
    files.extend(input_dir.glob(ext))

print(f"Found {len(files)} images")

for file in tqdm(files):

    try:
        img = Image.open(file).convert("L")

        # perceptual hash
        h = imagehash.phash(img)

        duplicate = False

        for existing_hash in hashes:

            # hamming distance
            if h - existing_hash <= THRESHOLD:
                duplicate = True
                break

        if not duplicate:
            hashes[h] = file
            shutil.copy(file, output_dir / file.name)

    except Exception as e:
        print("Error:", file, e)

print(f"\nKept {len(hashes)} unique-ish images")