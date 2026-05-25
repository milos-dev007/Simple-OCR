import torch
from torch import nn
from torch.nn import functional as F


class CRNN(nn.Module):
    def __init__(self, num_classes, rnn_hidden_size=128):
        super().__init__()
        self.num_classes = num_classes
        self.rnn_hidden_size = rnn_hidden_size

        self.feature_extractor = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(4, 1), stride=(4, 1)),
        )

        self.sequence_model = nn.LSTM(
            input_size=256,
            hidden_size=rnn_hidden_size,
            num_layers=2,
            bidirectional=True,
        )
        self.classifier = nn.Linear(rnn_hidden_size * 2, num_classes)
        self._init_parameters()

    def _init_parameters(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

        for name, parameter in self.sequence_model.named_parameters():
            if "weight_ih" in name:
                nn.init.xavier_uniform_(parameter)
            elif "weight_hh" in name:
                nn.init.orthogonal_(parameter)
            elif "bias" in name:
                nn.init.zeros_(parameter)
                hidden_size = parameter.shape[0] // 4
                parameter.data[hidden_size : hidden_size * 2] = 1.0

        if self.classifier.bias is not None:
            self.classifier.bias.data[0] = -2.0

    def forward(self, images, content_widths=None):
        features = self.feature_extractor(images)
        if features.shape[2] != 1:
            raise RuntimeError(f"Expected feature height of 1, got {features.shape[2]}")

        features = features.squeeze(2).permute(2, 0, 1)
        sequence_output, _ = self.sequence_model(features)
        logits = self.classifier(sequence_output)
        log_probs = F.log_softmax(logits, dim=2)
        if content_widths is None:
            output_lengths = torch.full(
                size=(images.shape[0],),
                fill_value=log_probs.shape[0],
                dtype=torch.long,
                device=images.device,
            )
        else:
            output_lengths = torch.clamp(content_widths // 8, min=1).to(images.device)
        return log_probs, output_lengths


def build_model(num_classes):
    return CRNN(num_classes=num_classes)
