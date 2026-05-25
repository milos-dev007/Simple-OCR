import argparse
import json
import shutil
from pathlib import Path

from ocr.config import DEFAULT_RANDOM_SEED, DEFAULT_TRAIN_COUNT, DEFAULT_VAL_COUNT, GENERATED_DIR, ROOT_DIR
from ocr.text_generator import (
    GENERATION_PROFILES,
    create_rng,
    load_font_paths,
    load_word_list,
    render_text_image,
    sample_label,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Generate a synthetic OCR dataset.")
    parser.add_argument("--train-count", type=int, default=DEFAULT_TRAIN_COUNT)
    parser.add_argument("--val-count", type=int, default=DEFAULT_VAL_COUNT)
    parser.add_argument("--output-dir", default=str(GENERATED_DIR))
    parser.add_argument("--seed", type=int, default=DEFAULT_RANDOM_SEED)
    parser.add_argument("--profile", choices=sorted(GENERATION_PROFILES), default="easy")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def write_split(split_name, count, output_dir, words, font_paths, seed, profile):
    rng = create_rng(seed)
    split_dir = output_dir / split_name
    images_dir = split_dir / "images"
    manifest_path = output_dir / f"{split_name}_manifest.jsonl"

    images_dir.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", encoding="utf-8") as manifest_file:
        for index in range(count):
            text = sample_label(rng, words, profile=profile)
            image = render_text_image(text, font_paths, rng, profile=profile)
            image_abs_path = images_dir / f"{split_name}_{index:06d}.png"
            image.save(image_abs_path)
            try:
                image_path_for_manifest = str(image_abs_path.relative_to(ROOT_DIR))
            except ValueError:
                image_path_for_manifest = str(image_abs_path)
            manifest_file.write(
                json.dumps({"image_path": image_path_for_manifest, "text": text}) + "\n"
            )

            if (index + 1) % 1000 == 0 or index + 1 == count:
                print(f"{split_name}: generated {index + 1}/{count}")


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)

    if args.force and output_dir.exists():
        shutil.rmtree(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    words = load_word_list()
    font_paths = load_font_paths()
    write_split("train", args.train_count, output_dir, words, font_paths, args.seed, args.profile)
    write_split("val", args.val_count, output_dir, words, font_paths, args.seed + 1, args.profile)
    print(f"Dataset written to {output_dir}")


if __name__ == "__main__":
    main()
