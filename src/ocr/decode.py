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


def greedy_decode(log_probs, input_lengths=None, charset=DEFAULT_CHARSET):
    if log_probs.ndim != 3:
        raise ValueError("Expected log_probs with shape [time, batch, classes].")

    best_indices = torch.argmax(log_probs, dim=2)
    predictions = []
    max_steps, batch_size, _ = log_probs.shape

    if input_lengths is None:
        input_lengths = [max_steps] * batch_size
    else:
        if isinstance(input_lengths, torch.Tensor):
            input_lengths = input_lengths.tolist()
        if len(input_lengths) != batch_size:
            raise ValueError("input_lengths must contain one length per batch item.")

    for batch_index in range(batch_size):
        valid_steps = max(0, min(max_steps, int(input_lengths[batch_index])))
        sequence = best_indices[:valid_steps, batch_index].tolist()
        collapsed = collapse_repeats_and_remove_blanks(sequence, charset.blank_index)
        predictions.append(charset.decode(collapsed))

    return predictions
