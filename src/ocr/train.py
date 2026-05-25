import argparse
import json
import math
import os
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import Subset

from ocr.charset import DEFAULT_CHARSET
from ocr.config import (
    BEST_CHECKPOINT_PATH,
    CHECKPOINT_DIR,
    DEFAULT_RANDOM_SEED,
    LAST_CHECKPOINT_PATH,
    METRICS_DIR,
    METRICS_PATH,
    PREDICTIONS_DIR,
    SAMPLE_PREDICTIONS_PATH,
    TRAIN_MANIFEST_PATH,
    VAL_MANIFEST_PATH,
)
from ocr.data import OCRDataset, build_dataloader
from ocr.decode import greedy_decode
from ocr.metrics import build_sample_predictions, character_error_rate, exact_line_accuracy
from ocr.model import build_model


def parse_args():
    parser = argparse.ArgumentParser(description="Train the OCR model.")
    parser.add_argument("--train-manifest", default=str(TRAIN_MANIFEST_PATH))
    parser.add_argument("--val-manifest", default=str(VAL_MANIFEST_PATH))
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--device", choices=["auto", "cpu", "mps"], default="auto")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=DEFAULT_RANDOM_SEED)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--smoke-samples", type=int, default=32)
    return parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def mps_ctc_supported():
    if not torch.backends.mps.is_available():
        return False

    device = torch.device("mps")
    criterion = nn.CTCLoss(blank=0, zero_infinity=True)

    try:
        logits = torch.randn(4, 1, 3, device=device, requires_grad=True)
        log_probs = logits.log_softmax(2)
        targets = torch.tensor([1, 2], dtype=torch.long, device=device)
        input_lengths = torch.tensor([4], dtype=torch.long, device=device)
        target_lengths = torch.tensor([2], dtype=torch.long, device=device)
        loss = criterion(log_probs, targets, input_lengths, target_lengths)
        loss.backward()
    except Exception:
        return False

    return True


def mps_fallback_enabled():
    return os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK") == "1"


def select_device(requested_device):
    mps_available = torch.backends.mps.is_available()

    if requested_device == "cpu":
        return torch.device("cpu")

    if requested_device == "mps":
        if not mps_available:
            raise SystemExit(
                "--device mps was requested, but MPS is not available in this Python/PyTorch environment."
            )
        if mps_fallback_enabled() or mps_ctc_supported():
            return torch.device("mps")
        raise SystemExit(
            "--device mps was requested, but CTCLoss is not implemented natively on MPS here. "
            "Set PYTORCH_ENABLE_MPS_FALLBACK=1 to allow CPU fallback for unsupported ops, or use --device cpu."
        )

    if mps_available and (mps_fallback_enabled() or mps_ctc_supported()):
        return torch.device("mps")

    return torch.device("cpu")


def resolve_hyperparameters(args):
    if args.smoke:
        epochs = args.epochs if args.epochs is not None else 80
        batch_size = args.batch_size if args.batch_size is not None else 4
        learning_rate = args.learning_rate if args.learning_rate is not None else 3e-3
    else:
        epochs = args.epochs if args.epochs is not None else 20
        batch_size = args.batch_size if args.batch_size is not None else 64
        learning_rate = args.learning_rate if args.learning_rate is not None else 1e-3
    return epochs, batch_size, learning_rate


def evaluate(model, data_loader, criterion, device):
    model.eval()
    total_loss = 0.0
    batch_count = 0
    predictions = []
    targets = []
    image_paths = []

    with torch.no_grad():
        for batch in data_loader:
            images = batch["images"].to(device)
            content_widths = batch["content_widths"].to(device)
            target_tensor = batch["targets"].to(device)
            target_lengths = batch["target_lengths"].to(device)

            log_probs, input_lengths = model(images, content_widths=content_widths)
            loss = criterion(log_probs, target_tensor, input_lengths, target_lengths)

            total_loss += loss.item()
            batch_count += 1
            predictions.extend(greedy_decode(log_probs.detach().cpu(), charset=DEFAULT_CHARSET))
            targets.extend(batch["texts"])
            image_paths.extend(batch["image_paths"])

    average_loss = total_loss / max(1, batch_count)
    cer = character_error_rate(predictions, targets)
    exact_accuracy = exact_line_accuracy(predictions, targets)
    sample_predictions = build_sample_predictions(image_paths, predictions, targets)

    return {
        "loss": average_loss,
        "cer": cer,
        "exact_line_accuracy": exact_accuracy,
        "sample_predictions": sample_predictions,
    }


def save_checkpoint(path, model, optimizer, epoch, metrics):
    payload = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "num_classes": DEFAULT_CHARSET.size,
        "rnn_hidden_size": model.rnn_hidden_size,
        "charset": DEFAULT_CHARSET.to_metadata(),
        "metrics": metrics,
        "preprocessing": {
            "image_height": 32,
            "image_width": 256,
            "normalize_mean": 0.5,
            "normalize_std": 0.5,
        },
    }
    torch.save(payload, path)


def build_datasets(train_manifest_path, val_manifest_path, smoke_samples):
    train_dataset = OCRDataset(train_manifest_path)
    val_dataset = OCRDataset(val_manifest_path)

    if smoke_samples is None:
        return train_dataset, val_dataset

    smoke_count = min(smoke_samples, len(train_dataset))

    def easy_rank(index):
        text = train_dataset.records[index]["text"]
        has_space = " " in text
        has_digit = any(character.isdigit() for character in text)
        is_lowercase_word = text.islower()
        return (
            has_space,
            has_digit,
            not is_lowercase_word,
            len(text),
            text,
        )

    ranked_indices = sorted(range(len(train_dataset.records)), key=easy_rank)
    base_count = min(1, smoke_count)
    base_indices = ranked_indices[:base_count]
    repeats = math.ceil(smoke_count / len(base_indices))
    smoke_indices = (base_indices * repeats)[:smoke_count]
    train_subset = Subset(train_dataset, smoke_indices)
    return train_subset, train_subset


def main():
    args = parse_args()
    train_manifest_path = Path(args.train_manifest)
    val_manifest_path = Path(args.val_manifest)

    if not train_manifest_path.exists() or not val_manifest_path.exists():
        raise SystemExit(
            "Training manifests were not found. Run `python -m ocr.generate_data` before training."
        )

    set_seed(args.seed)
    device = select_device(args.device)
    epochs, batch_size, learning_rate = resolve_hyperparameters(args)

    smoke_samples = args.smoke_samples if args.smoke else None
    train_dataset, val_dataset = build_datasets(
        train_manifest_path=train_manifest_path,
        val_manifest_path=val_manifest_path,
        smoke_samples=smoke_samples,
    )

    train_loader = build_dataloader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )
    val_loader = build_dataloader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    model = build_model(num_classes=DEFAULT_CHARSET.size).to(device)
    optimizer = Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CTCLoss(blank=DEFAULT_CHARSET.blank_index, zero_infinity=True)

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)

    history = []
    best_cer = float("inf")

    print(f"Requested device: {args.device}")
    print(f"Using device: {device}")
    if device.type == "mps" and mps_fallback_enabled():
        print("MPS fallback enabled: unsupported ops such as CTCLoss will run on CPU.")
    print(f"Train samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    print(f"Epochs: {epochs}")
    print(f"Batch size: {batch_size}")
    print(f"Learning rate: {learning_rate}")
    if args.smoke:
        print(
            "Smoke mode enabled: validation reuses a repeated easy training subset to force overfitting."
        )

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        batch_count = 0

        for batch in train_loader:
            images = batch["images"].to(device)
            content_widths = batch["content_widths"].to(device)
            targets = batch["targets"].to(device)
            target_lengths = batch["target_lengths"].to(device)

            optimizer.zero_grad()
            log_probs, input_lengths = model(images, content_widths=content_widths)
            loss = criterion(log_probs, targets, input_lengths, target_lengths)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            batch_count += 1

        train_loss = running_loss / max(1, batch_count)
        validation_metrics = evaluate(model, val_loader, criterion, device)
        epoch_metrics = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": validation_metrics["loss"],
            "val_cer": validation_metrics["cer"],
            "val_exact_line_accuracy": validation_metrics["exact_line_accuracy"],
        }
        history.append(epoch_metrics)

        save_checkpoint(LAST_CHECKPOINT_PATH, model, optimizer, epoch, epoch_metrics)
        if validation_metrics["cer"] <= best_cer:
            best_cer = validation_metrics["cer"]
            save_checkpoint(BEST_CHECKPOINT_PATH, model, optimizer, epoch, epoch_metrics)

        with METRICS_PATH.open("w", encoding="utf-8") as file:
            json.dump(history, file, indent=2)

        with SAMPLE_PREDICTIONS_PATH.open("w", encoding="utf-8") as file:
            json.dump(validation_metrics["sample_predictions"], file, indent=2)

        print(
            f"Epoch {epoch:02d} | "
            f"train_loss={train_loss:.4f} | "
            f"val_loss={validation_metrics['loss']:.4f} | "
            f"val_cer={validation_metrics['cer']:.4f} | "
            f"val_exact={validation_metrics['exact_line_accuracy']:.4f}"
        )

    print(f"Best checkpoint: {BEST_CHECKPOINT_PATH}")
    print(f"Latest checkpoint: {LAST_CHECKPOINT_PATH}")
    print(f"Metrics: {METRICS_PATH}")
    print(f"Sample predictions: {SAMPLE_PREDICTIONS_PATH}")


if __name__ == "__main__":
    main()
