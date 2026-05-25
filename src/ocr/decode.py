import torch

from ocr.charset import DEFAULT_CHARSET


def collapse_repeats_and_remove_blanks(indices, blank_index):
    collapsed = []
    previous = None

    for index in indices:
        if index == blank_index:
            previous = index
            continue
        if index != previous:
            collapsed.append(index)
        previous = index

    return collapsed


def greedy_decode(log_probs, charset=DEFAULT_CHARSET):
    if log_probs.ndim != 3:
        raise ValueError("Expected log_probs with shape [time, batch, classes].")

    best_indices = torch.argmax(log_probs, dim=2)
    predictions = []

    for batch_index in range(best_indices.shape[1]):
        sequence = best_indices[:, batch_index].tolist()
        collapsed = collapse_repeats_and_remove_blanks(sequence, charset.blank_index)
        predictions.append(charset.decode(collapsed))

    return predictions
