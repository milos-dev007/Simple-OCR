import torch

from ocr.charset import DEFAULT_CHARSET
from ocr.model import build_model


def test_model_forward_shapes():
    model = build_model(num_classes=DEFAULT_CHARSET.size)
    images = torch.randn(2, 1, 32, 256)

    log_probs, output_lengths = model(images)

    assert log_probs.shape[1] == 2
    assert log_probs.shape[2] == DEFAULT_CHARSET.size
    assert output_lengths.shape == (2,)
    assert output_lengths[0].item() == log_probs.shape[0]
