#renaming labels from has-n to n where n is the actual matching name with image

from pathlib import Path

images_dir = Path("unlabelled_3")
labels_dir = Path(r"labels_3\train_3")

if not images_dir.exists():
    print(f"Missing images folder: {images_dir}")
elif not labels_dir.exists():
    print(f"Missing labels folder: {labels_dir}")
else:
    image_map = {}

    for img in images_dir.iterdir():
        if img.suffix.lower() in [".png", ".jpg", ".jpeg"]:
            image_map[img.stem] = img

    renamed = 0
    skipped = 0

    for txt in labels_dir.glob("*.txt"):
        try:
            number = txt.stem.split("-")[-1]

            if number in image_map:
                new_path = labels_dir / f"{number}.txt"

                if new_path.exists():
                    print(f"Skipped (already exists): {new_path.name}")
                    skipped += 1
                    continue

                txt.rename(new_path)
                print(f"Renamed: {txt.name} -> {new_path.name}")
                renamed += 1

            else:
                print(f"No matching image for: {txt.name}")

        except Exception as e:
            print(f"Error with {txt.name}: {e}")

    print(f"\nRenamed: {renamed}, Skipped: {skipped}\n")

print("Done")