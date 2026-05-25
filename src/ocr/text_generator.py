import random
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageFont, ImageOps

from ocr.charset import DEFAULT_CHARSET
from ocr.config import FONTS_DIR, MAX_LABEL_LENGTH, WORDS_PATH


def load_word_list(words_path=WORDS_PATH):
    with Path(words_path).open("r", encoding="utf-8") as file:
        return [line.strip() for line in file if line.strip()]


def load_font_paths(fonts_dir=FONTS_DIR):
    fonts_dir = Path(fonts_dir)
    return sorted(
        path for path in fonts_dir.iterdir() if path.suffix.lower() in {".ttf", ".otf", ".ttc"}
    )


def apply_case_variant(word, rng):
    variant = rng.choices(
        population=["lower", "upper", "title", "mixed"],
        weights=[0.45, 0.15, 0.3, 0.1],
        k=1,
    )[0]
    if variant == "lower":
        return word.lower()
    if variant == "upper":
        return word.upper()
    if variant == "title":
        return word.title()
    characters = []
    for character in word:
        characters.append(character.upper() if rng.random() < 0.5 else character.lower())
    return "".join(characters)


def maybe_add_digits(word, rng):
    if rng.random() < 0.1:
        word = f"{rng.randint(0, 999)}{word}"
    if rng.random() < 0.1:
        word = f"{word}{rng.randint(0, 999)}"
    return word


def sample_label(rng, words, max_label_length=MAX_LABEL_LENGTH, charset=DEFAULT_CHARSET):
    for _ in range(100):
        word_count = rng.choices(population=[1, 2, 3, 4], weights=[0.55, 0.3, 0.1, 0.05], k=1)[0]
        parts = []
        for _ in range(word_count):
            word = apply_case_variant(rng.choice(words), rng)
            word = maybe_add_digits(word, rng)
            parts.append(word)
        text = " ".join(parts)
        if 1 <= len(text) <= max_label_length and charset.contains(text):
            return text

    fallback = apply_case_variant(rng.choice(words), rng)
    fallback = maybe_add_digits(fallback, rng)
    fallback = fallback[:max_label_length]
    fallback = "".join(character for character in fallback if character in charset.char_to_index)
    return fallback or "Hello"


def render_text_image(text, font_paths, rng):
    if not font_paths:
        raise RuntimeError(
            f"No fonts found in {FONTS_DIR}. Add open .ttf or .otf fonts before generating data."
        )

    font_path = rng.choice(font_paths)
    font_size = rng.randint(22, 30)
    font = ImageFont.truetype(str(font_path), font_size)

    temp_image = Image.new("L", (1, 1), color=255)
    temp_draw = ImageDraw.Draw(temp_image)
    left, top, right, bottom = temp_draw.textbbox((0, 0), text, font=font)
    text_width = right - left
    text_height = bottom - top

    margin_x = 16
    margin_y = 10
    offset_x = rng.randint(0, 8)
    offset_y = rng.randint(-3, 3)
    background = rng.randint(238, 255)
    text_color = rng.randint(0, 28)

    canvas_width = max(80, text_width + margin_x * 2 + 12)
    canvas_height = max(42, text_height + margin_y * 2 + 8)
    image = Image.new("L", (canvas_width, canvas_height), color=background)
    draw = ImageDraw.Draw(image)
    draw.text((margin_x + offset_x, margin_y + offset_y), text, fill=text_color, font=font)

    rotation = rng.uniform(-2.5, 2.5)
    image = image.rotate(
        rotation,
        resample=Image.Resampling.BICUBIC,
        expand=True,
        fillcolor=background,
    )

    if rng.random() < 0.55:
        image = image.filter(ImageFilter.GaussianBlur(radius=rng.uniform(0.0, 0.45)))

    if rng.random() < 0.55:
        contrast = ImageEnhance.Contrast(image)
        image = contrast.enhance(rng.uniform(0.92, 1.1))

    if rng.random() < 0.35:
        brightness = ImageEnhance.Brightness(image)
        image = brightness.enhance(rng.uniform(0.96, 1.05))

    image_array = np.asarray(image, dtype=np.float32)
    if rng.random() < 0.45:
        noise = np.random.default_rng(rng.randint(0, 1_000_000)).normal(
            loc=0.0, scale=rng.uniform(0.5, 3.0), size=image_array.shape
        )
        image_array = np.clip(image_array + noise, 0, 255)

    image = Image.fromarray(image_array.astype(np.uint8), mode="L")
    text_bbox = ImageOps.invert(image).getbbox()
    if text_bbox:
        crop_margin = 4
        left, top, right, bottom = text_bbox
        image = image.crop(
            (
                max(0, left - crop_margin),
                max(0, top - crop_margin),
                min(image.width, right + crop_margin),
                min(image.height, bottom + crop_margin),
            )
        )

    return image


def create_rng(seed):
    return random.Random(seed)
