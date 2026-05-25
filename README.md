# OCR From Scratch

This project trains a small OCR model from scratch in Python.

V1 is intentionally narrow:

- input: a cropped grayscale image containing one printed English word or short line
- output: the recognized text string
- model: a CRNN with CTC loss
- data: synthetic images generated locally
- interface: command line only

## Project layout

```text
Nest.js/
  assets/
    fonts/
    words/
  data/
    generated/
  artifacts/
    checkpoints/
    metrics/
    predictions/
  src/
    ocr/
  tests/
  README.md
  pyproject.toml
  requirements.txt
```

## Python version

Use Python `3.12`.

The package is installed in editable mode through `requirements.txt`, so `python -m ocr.<module>` works after one install.

## Setup

```bash
python3.12 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

## Generate synthetic data

Standard synthetic dataset:

```bash
.venv/bin/python -m ocr.generate_data --train-count 20000 --val-count 2000
```

Easier curriculum dataset for first successful training runs:

```bash
.venv/bin/python -m ocr.generate_data --train-count 20000 --val-count 2000 --profile easy --force
```

This creates:

- `data/generated/train/images/*.png`
- `data/generated/val/images/*.png`
- `data/generated/train_manifest.jsonl`
- `data/generated/val_manifest.jsonl`

Each manifest line contains:

```json
{"image_path": "data/generated/train/images/train_000000.png", "text": "Hello 42"}
```

Profiles:

- `standard`: mixed case, optional digits, multi-word labels, heavier augmentation
- `easy`: single lowercase words, no digits, much lighter augmentation

## Train the OCR model

Full training:

```bash
.venv/bin/python -m ocr.train --epochs 20 --batch-size 64 --device cpu
```

Recommended first real training run:

```bash
.venv/bin/python -m ocr.generate_data --train-count 20000 --val-count 2000 --profile easy --force
.venv/bin/python -m ocr.train --epochs 20 --batch-size 64 --num-workers 4 --device cpu
```

Windows or Linux with an NVIDIA GPU:

```bash
.venv\Scripts\python -m ocr.train --epochs 20 --batch-size 64 --device cuda
```

Apple Silicon GPU training:

```bash
PYTORCH_ENABLE_MPS_FALLBACK=1 .venv/bin/python -m ocr.train --epochs 20 --batch-size 64 --device mps
```

This project uses `CTCLoss`, and in the current local PyTorch setup that operation is not implemented natively on MPS. Setting `PYTORCH_ENABLE_MPS_FALLBACK=1` lets the model run on the Apple GPU while unsupported ops fall back to CPU.

For Windows/NVIDIA, install a CUDA-enabled PyTorch build first. The official PyTorch selector provides the current Windows install command for CUDA.

Smoke overfit run:

```bash
.venv/bin/python -m ocr.train --smoke --device cpu
```

`--smoke` is a pipeline check, not a quality benchmark. It repeats one easy training image to `32` samples so the project can prove that generation, preprocessing, CTC training, checkpoint saving, and prediction all work end to end.

Saved artifacts:

- `artifacts/checkpoints/best.pt`
- `artifacts/checkpoints/last.pt`
- `artifacts/metrics/train_metrics.json`
- `artifacts/predictions/sample_predictions.json`

## Predict from an image

```bash
.venv/bin/python -m ocr.predict --image path/to/image.png
```

## What the model stores

The trained checkpoint mainly stores:

- weight tensors for the CNN
- weight tensors for the bidirectional LSTMs
- the linear classifier weights
- metadata for charset, image size, and model hyperparameters

The checkpoint is not only raw numbers. It also needs:

- the model architecture definition
- preprocessing rules
- CTC decoding rules
- the charset mapping

## Architecture

The recognizer is a CRNN:

1. a CNN extracts visual features from the image
2. the feature map height is collapsed to `1`
3. the feature map width becomes a sequence of time steps
4. a two-layer bidirectional LSTM models character order
5. a linear layer predicts class logits at each time step

The class space is:

- blank token for CTC
- space
- digits `0-9`
- uppercase `A-Z`
- lowercase `a-z`

That is `64` classes total.

## Preprocessing

Training and inference use the same preprocessing:

1. convert the image to grayscale
2. crop the dark foreground before resizing
3. resize while preserving aspect ratio
4. paste onto a white `32x256` canvas
5. align content to the left and center it vertically
6. scale pixels to `0..1`
7. normalize with mean `0.5` and std `0.5`

Synthetic augmentation happens during dataset generation, not prediction:

- font size jitter
- small rotation
- blur
- noise
- contrast change
- x/y offset

The `standard` profile intentionally makes the task harder. If the model collapses to the same short prediction for many images, start with `--profile easy` and only move back to `standard` after the model learns the simpler distribution.

## Why CTC

The model predicts a class distribution for every time step instead of predicting a fixed number of characters.

CTC makes training possible without manually marking where each character begins and ends inside the image. During decoding:

1. choose the highest-probability class per time step
2. collapse repeated tokens
3. remove blank tokens

The remaining token sequence becomes the final text.

## Tests

```bash
.venv/bin/python -m pytest
```

The test suite covers:

- charset encode/decode roundtrip
- synthetic label validity
- preprocessing output shape
- CTC greedy decoding
- CRNN forward output shapes
