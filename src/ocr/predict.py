import argparse

import torch

from ocr.charset import Charset
from ocr.decode import greedy_decode
from ocr.model import CRNN
from ocr.preprocessing import preprocess_image_file


def parse_args():
    parser = argparse.ArgumentParser(description="Predict text from an image.")
    parser.add_argument("--image", required=True)
    parser.add_argument("--checkpoint", default="artifacts/checkpoints/best.pt")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    return parser.parse_args()


def select_device(requested_device):
    if requested_device == "cpu":
        return torch.device("cpu")
    if requested_device == "cuda":
        if not torch.cuda.is_available():
            raise SystemExit(
                "--device cuda was requested, but CUDA is not available in this Python/PyTorch environment."
            )
        return torch.device("cuda")
    if requested_device == "mps":
        if not torch.backends.mps.is_available():
            raise SystemExit(
                "--device mps was requested, but MPS is not available in this Python/PyTorch environment."
            )
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def main():
    args = parse_args()
    device = select_device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location=device)

    charset_metadata = checkpoint["charset"]
    charset = Charset(
        characters=charset_metadata["characters"],
        blank_token=charset_metadata["blank_token"],
    )
    model = CRNN(
        num_classes=checkpoint["num_classes"],
        rnn_hidden_size=checkpoint["rnn_hidden_size"],
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    image_tensor, content_width = preprocess_image_file(args.image, return_width=True)
    image_tensor = image_tensor.unsqueeze(0).to(device)
    content_widths = torch.tensor([content_width], dtype=torch.long, device=device)

    with torch.no_grad():
        log_probs, input_lengths = model(image_tensor, content_widths=content_widths)
        prediction = greedy_decode(log_probs.cpu(), input_lengths=input_lengths.cpu(), charset=charset)[0]

    print(f"Predicted text: {prediction}")


if __name__ == "__main__":
    main()
