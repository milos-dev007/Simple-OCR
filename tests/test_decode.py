import torch

from ocr.charset import DEFAULT_CHARSET
from ocr.decode import greedy_decode


def test_greedy_decode_collapses_repeats_and_blanks():
    class_count = DEFAULT_CHARSET.size
    blank = DEFAULT_CHARSET.blank_index
    h = DEFAULT_CHARSET.char_to_index["H"]
    i = DEFAULT_CHARSET.char_to_index["i"]

    logits = torch.full((6, 1, class_count), fill_value=-10.0)
    logits[0, 0, h] = 5.0
    logits[1, 0, h] = 5.0
    logits[2, 0, blank] = 5.0
    logits[3, 0, i] = 5.0
    logits[4, 0, i] = 5.0
    logits[5, 0, blank] = 5.0

    predictions = greedy_decode(logits)

    assert predictions == ["Hi"]


def test_greedy_decode_respects_input_lengths():
    class_count = DEFAULT_CHARSET.size
    blank = DEFAULT_CHARSET.blank_index
    h = DEFAULT_CHARSET.char_to_index["H"]
    i = DEFAULT_CHARSET.char_to_index["i"]
    x = DEFAULT_CHARSET.char_to_index["x"]

    logits = torch.full((6, 1, class_count), fill_value=-10.0)
    logits[0, 0, h] = 5.0
    logits[1, 0, blank] = 5.0
    logits[2, 0, i] = 5.0
    logits[3, 0, blank] = 5.0
    logits[4, 0, x] = 5.0
    logits[5, 0, x] = 5.0

    predictions = greedy_decode(logits, input_lengths=[4])

    assert predictions == ["Hi"]
