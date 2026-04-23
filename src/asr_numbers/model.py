from __future__ import annotations

import torch
from torch import nn
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence

from .features import LogMelSpectrogram, apply_spec_augment, masked_mean_and_std


class ConvGRUCTCModel(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        sample_rate: int = 16000,
        n_fft: int = 400,
        win_length: int = 400,
        hop_length: int = 160,
        n_mels: int = 80,
        conv_channels: int = 192,
        encoder_dim: int = 256,
        hidden_size: int = 256,
        num_gru_layers: int = 3,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.feature_extractor = LogMelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            win_length=win_length,
            hop_length=hop_length,
            n_mels=n_mels,
        )
        self.conv = nn.Sequential(
            nn.Conv1d(n_mels, conv_channels, kernel_size=5, stride=2, padding=2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(conv_channels, encoder_dim, kernel_size=5, stride=2, padding=2),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.gru = nn.GRU(
            input_size=encoder_dim,
            hidden_size=hidden_size,
            num_layers=num_gru_layers,
            dropout=dropout if num_gru_layers > 1 else 0.0,
            batch_first=True,
            bidirectional=True,
        )
        self.classifier = nn.Linear(hidden_size * 2, vocab_size)

    @staticmethod
    def _conv_out_length(lengths: torch.Tensor) -> torch.Tensor:
        lengths = torch.div(lengths + 1, 2, rounding_mode="floor")
        lengths = torch.div(lengths + 1, 2, rounding_mode="floor")
        return lengths.clamp_min(1)

    def forward(
        self,
        waveforms: torch.Tensor,
        lengths: torch.Tensor,
        spec_augment_config: dict[str, int | float] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        features, feature_lengths = self.feature_extractor(waveforms, lengths)
        mean, std = masked_mean_and_std(features, feature_lengths)
        features = (features - mean) / std
        if spec_augment_config:
            features = apply_spec_augment(features, feature_lengths, spec_augment_config)

        x = features.transpose(1, 2)
        x = self.conv(x)
        x = x.transpose(1, 2)

        output_lengths = self._conv_out_length(feature_lengths)
        packed = pack_padded_sequence(x, output_lengths.cpu(), batch_first=True, enforce_sorted=False)
        packed_output, _ = self.gru(packed)
        encoded, _ = pad_packed_sequence(packed_output, batch_first=True)
        logits = self.classifier(encoded)
        return logits, output_lengths


def count_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
