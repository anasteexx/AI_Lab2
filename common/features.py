
from PIL import Image


def extract_features(file_path):
    with Image.open(file_path) as source:
        if source.format != "BMP":
            raise ValueError("Підтримуються лише BMP-зображення.")

        image = source.convert("L")

    width, height = image.size

    if width < 5 or height < 5:
        raise ValueError("Розмір зображення має бути не менше 5×5.")

    absolute = []
    bipolar = []

    for row in range(5):
        top = row * height // 5
        bottom = (row + 1) * height // 5

        for column in range(5):
            left = column * width // 5
            right = (column + 1) * width // 5

            region = image.crop((left, top, right, bottom))
            pixels = list(region.getdata())

            black_pixels = sum(pixel < 128 for pixel in pixels)

            absolute.append(black_pixels)
            bipolar.append(
                1 if black_pixels / len(pixels) >= 0.01 else -1
            )

    total = sum(absolute)

    normalized = [
        value / total if total else 0.0
        for value in absolute
    ]

    return absolute, normalized, bipolar
