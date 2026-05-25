import numpy as np
import torch
from PIL import Image

from ocr.config import IMAGE_HEIGHT, IMAGE_WIDTH, NORMALIZE_MEAN, NORMALIZE_STD


def crop_foreground(grayscale_image, threshold_offset=20, margin=2):
    image_array = np.asarray(grayscale_image, dtype=np.uint8)
    background_level = float(np.percentile(image_array, 95))
    threshold = max(0.0, background_level - threshold_offset)
    foreground = image_array < threshold

    if not foreground.any():
        return grayscale_image

    rows, columns = np.where(foreground)
    top = max(0, int(rows.min()) - margin)
    bottom = min(grayscale_image.height, int(rows.max()) + margin + 1)
    left = max(0, int(columns.min()) - margin)
    right = min(grayscale_image.width, int(columns.max()) + margin + 1)
    return grayscale_image.crop((left, top, right, bottom))


def preprocess_pil_image(image, return_width=False):
    grayscale = image.convert("L")
    grayscale = crop_foreground(grayscale)
    original_width, original_height = grayscale.size

    if original_width <= 0 or original_height <= 0:
        raise ValueError("Image has invalid size.")

    left_padding = 4
    usable_width = IMAGE_WIDTH - left_padding
    scale = min(usable_width / original_width, IMAGE_HEIGHT / original_height)
    resized_width = max(1, min(usable_width, int(round(original_width * scale))))
    resized_height = max(1, min(IMAGE_HEIGHT, int(round(original_height * scale))))
    resized = grayscale.resize((resized_width, resized_height), resample=Image.Resampling.BICUBIC)

    canvas = Image.new("L", (IMAGE_WIDTH, IMAGE_HEIGHT), color=255)
    top_offset = (IMAGE_HEIGHT - resized_height) // 2
    canvas.paste(resized, (left_padding, top_offset))

    image_array = np.asarray(canvas, dtype=np.float32) / 255.0
    image_array = (image_array - NORMALIZE_MEAN) / NORMALIZE_STD

    tensor = torch.from_numpy(image_array).unsqueeze(0)
    if return_width:
        return tensor, left_padding + resized_width
    return tensor


def preprocess_image_file(image_path, return_width=False):
    image = Image.open(image_path)
    return preprocess_pil_image(image, return_width=return_width)
