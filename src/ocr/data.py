import json
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from ocr.charset import DEFAULT_CHARSET
from ocr.config import ROOT_DIR
from ocr.preprocessing import preprocess_pil_image


class OCRDataset(Dataset):
    def __init__(self, manifest_path, charset=DEFAULT_CHARSET):
        self.manifest_path = Path(manifest_path)
        self.charset = charset
        with self.manifest_path.open("r", encoding="utf-8") as file:
            self.records = [json.loads(line) for line in file if line.strip()]

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        record = self.records[index]
        image_path = Path(record["image_path"])
        if not image_path.is_absolute():
            image_path = ROOT_DIR / image_path

        image = Image.open(image_path)
        tensor, content_width = preprocess_pil_image(image, return_width=True)
        encoded = torch.tensor(self.charset.encode(record["text"]), dtype=torch.long)

        return {
            "image": tensor,
            "content_width": content_width,
            "target": encoded,
            "target_length": encoded.numel(),
            "text": record["text"],
            "image_path": str(image_path),
        }


def ctc_collate_fn(batch):
    images = torch.stack([item["image"] for item in batch])
    content_widths = torch.tensor([item["content_width"] for item in batch], dtype=torch.long)
    targets = torch.cat([item["target"] for item in batch])
    target_lengths = torch.tensor([item["target_length"] for item in batch], dtype=torch.long)
    texts = [item["text"] for item in batch]
    image_paths = [item["image_path"] for item in batch]

    return {
        "images": images,
        "content_widths": content_widths,
        "targets": targets,
        "target_lengths": target_lengths,
        "texts": texts,
        "image_paths": image_paths,
    }


def build_dataloader(dataset, batch_size, shuffle, num_workers=0, pin_memory=False):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=ctc_collate_fn,
    )
