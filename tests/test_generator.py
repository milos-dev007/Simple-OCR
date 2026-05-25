from ocr.charset import DEFAULT_CHARSET
from ocr.text_generator import create_rng, sample_label


def test_generated_labels_use_supported_characters():
    words = ["hello", "world", "sample", "text", "model"]
    rng = create_rng(123)

    for _ in range(100):
        label = sample_label(rng, words)
        assert 1 <= len(label) <= 24
        assert DEFAULT_CHARSET.contains(label)
