##copying labelled images out of the selected ones through use of labels downloaded
from pathlib import Path
import shutil

selected_dir = Path("images/selected")
labels_dir = Path("labels/train_1")
output_dir = Path("labelled")

output_dir.mkdir(exist_ok=True)

image_extensions = [".jpg", ".jpeg", ".png", ".bmp", ".webp"]

moved = 0

for txt_file in labels_dir.glob("*.txt"):
    
    number = txt_file.stem.split("-")[-1]
    
    found = False

    for ext in image_extensions:
        
        image_path = selected_dir / f"{number}{ext}"
        
        if image_path.exists():
            
            shutil.move(
                str(image_path),
                str(output_dir / image_path.name)
            )
            
            moved += 1
            found = True
            break

    if not found:
        print(f"Could not find image for: {txt_file.name}")

print(f"\nMoved {moved} labelled images.")