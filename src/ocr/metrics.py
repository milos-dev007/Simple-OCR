def edit_distance(source, target):
    if source == target:
        return 0
    if not source:
        return len(target)
    if not target:
        return len(source)

    previous_row = list(range(len(target) + 1))
    for source_index, source_char in enumerate(source, start=1):
        current_row = [source_index]
        for target_index, target_char in enumerate(target, start=1):
            insert_cost = current_row[target_index - 1] + 1
            delete_cost = previous_row[target_index] + 1
            replace_cost = previous_row[target_index - 1] + (source_char != target_char)
            current_row.append(min(insert_cost, delete_cost, replace_cost))
        previous_row = current_row
    return previous_row[-1]


def character_error_rate(predictions, targets):
    total_distance = 0
    total_length = 0
    for prediction, target in zip(predictions, targets, strict=True):
        total_distance += edit_distance(prediction, target)
        total_length += max(1, len(target))
    return total_distance / max(1, total_length)


def exact_line_accuracy(predictions, targets):
    if not targets:
        return 0.0
    correct = sum(prediction == target for prediction, target in zip(predictions, targets, strict=True))
    return correct / len(targets)


def build_sample_predictions(image_paths, predictions, targets, limit=20):
    samples = []
    for image_path, prediction, target in zip(image_paths, predictions, targets, strict=True):
        if len(samples) >= limit:
            break
        samples.append(
            {
                "image_path": image_path,
                "target": target,
                "prediction": prediction,
                "exact_match": prediction == target,
            }
        )
    return samples
