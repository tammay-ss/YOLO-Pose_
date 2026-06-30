#converting images into a sheet of images for better view to select images for training
from PIL import Image, ImageOps, ImageDraw, ImageFont
from pathlib import Path
import math

input_dir = Path("images/val")
output_dir = Path("contact_sheets")

output_dir.mkdir(exist_ok=True)

# supported formats
files = []

extensions = ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tif"]

for ext in extensions:
    files.extend(input_dir.glob(ext))

files = sorted(files)

thumb_size = 420

cols = 2
rows = 5

per_page = cols * rows

pages = math.ceil(len(files) / per_page)

print(f"Creating {pages} contact sheets")

try:
    font = ImageFont.truetype("arial.ttf", 20)
except:
    font = ImageFont.load_default()

for page in range(pages):

    sheet = Image.new(
        "RGB",
        (cols * thumb_size, rows * thumb_size),
        "white"
    )

    batch = files[
        page * per_page:(page + 1) * per_page
    ]

    for i, file in enumerate(batch):

        try:
            img = Image.open(file).convert("L")

            img.thumbnail((thumb_size - 20, thumb_size - 40))

            img = ImageOps.autocontrast(img)

            tile = Image.new(
                "RGB",
                (thumb_size, thumb_size),
                "white"
            )

            # center image
            x_offset = (thumb_size - img.width) // 2
            y_offset = 10

            tile.paste(img.convert("RGB"), (x_offset, y_offset))

            draw = ImageDraw.Draw(tile)

            # filename text
            text = file.stem

            draw.text(
                (10, thumb_size - 30),
                text,
                fill="red",
                font=font
            )

            x = (i % cols) * thumb_size
            y = (i // cols) * thumb_size

            sheet.paste(tile, (x, y))

        except Exception as e:
            print("Error:", file, e)

    sheet.save(
        output_dir / f"sheet_V_{page:03d}.jpg"
    )

print("Done")