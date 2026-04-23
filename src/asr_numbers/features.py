from __future__ import annotations

import math

import numpy as np
import torch
from torch import nn


def _hz_to_mel(freq: np.ndarray) -> np.ndarray:
    return 2595.0 * np.log10(1.0 + freq / 700.0)


def _mel_to_hz(mel: np.ndarray) -> np.ndarray:
    return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)


def create_mel_filterbank(
    sample_rate: int,
    n_fft: int,
    n_mels: int,
    f_min: float = 0.0,
    f_max: float | None = None,
) -> torch.Tensor:
    f_max = float(f_max or sample_rate / 2)
    mel_min = _hz_to_mel(np.array([f_min], dtype=np.float64))[0]
    mel_max = _hz_to_mel(np.array([f_max], dtype=np.float64))[0]
    mel_points = np.linspace(mel_min, mel_max, n_mels + 2)
    hz_points = _mel_to_hz(mel_points)
    bins = np.floor((n_fft + 1) * hz_points / sample_rate).astype(np.int64)
    n_freqs = n_fft // 2 + 1

    fbanks = np.zeros((n_freqs, n_mels), dtype=np.float32)
    for mel_index in range(n_mels):
        left = bins[mel_index]
        center = bins[mel_index + 1]
        right = bins[mel_index + 2]
        center = max(center, left + 1)
        right = max(right, center + 1)
        right = min(right, n_freqs)

        for freq_bin in range(left, min(center, n_freqs)):
            fbanks[freq_bin, mel_index] = (freq_bin - left) / max(1, center - left)
        for freq_bin in range(center, right):
            fbanks[freq_bin, mel_index] = (right - freq_bin) / max(1, right - center)
    return torch.from_numpy(fbanks)


class LogMelSpectrogram(nn.Module):
    def __init__(
        self,
        sample_rate: int = 16000,
        n_fft: int = 400,
        win_length: int = 400,
        hop_length: int = 160,
        n_mels: int = 80,
    ) -> None:
        super().__init__()
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.win_length = win_length
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.register_buffer("window", torch.hann_window(win_length), persistent=False)
        self.register_buffer(
            "mel_filterbank",
            create_mel_filterbank(sample_rate=sample_rate, n_fft=n_fft, n_mels=n_mels),
            persistent=False,
        )

    def forward(self, waveforms: torch.Tensor, lengths: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        spec = torch.stft(
            waveforms,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window,
            center=True,
            return_complex=True,
        )
        power = spec.abs().pow(2.0).transpose(1, 2)
        mel = power @ self.mel_filterbank
        log_mel = torch.log(mel.clamp_min(1e-5))
        frame_lengths = torch.div(lengths, self.hop_length, rounding_mode="floor") + 1
        frame_lengths = torch.clamp(frame_lengths, max=log_mel.size(1))
        return log_mel, frame_lengths


def masked_mean_and_std(features: torch.Tensor, lengths: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    max_frames = features.size(1)
    mask = torch.arange(max_frames, device=features.device).unsqueeze(0) < lengths.unsqueeze(1)
    mask = mask.unsqueeze(-1)
    denom = mask.sum(dim=1, keepdim=True).clamp_min(1)
    mean = (features * mask).sum(dim=1, keepdim=True) / denom
    var = ((features - mean).pow(2) * mask).sum(dim=1, keepdim=True) / denom
    std = torch.sqrt(var + 1e-5)
    return mean, std


def apply_spec_augment(features: torch.Tensor, lengths: torch.Tensor, config: dict[str, int | float]) -> torch.Tensor:
    augmented = features.clone()
    batch_size, _, n_mels = augmented.shape
    time_mask_count = int(config.get("time_mask_count", 0))
    time_mask_max_frames = int(config.get("time_mask_max_frames", 0))
    time_mask_ratio = float(config.get("time_mask_ratio", 0.0))
    freq_mask_count = int(config.get("freq_mask_count", 0))
    freq_mask_max_bins = int(config.get("freq_mask_max_bins", 0))

    if time_mask_count <= 0 and freq_mask_count <= 0:
        return augmented

    for batch_index in range(batch_size):
        valid_frames = int(lengths[batch_index].item())
        if valid_frames <= 0:
            continue

        if time_mask_ratio > 0.0:
            ratio_cap = max(1, int(valid_frames * time_mask_ratio))
            frame_cap = min(valid_frames, max(time_mask_max_frames, ratio_cap))
        else:
            frame_cap = min(valid_frames, time_mask_max_frames)

        for _ in range(time_mask_count):
            if frame_cap <= 0:
                continue
            width = int(torch.randint(1, frame_cap + 1, (1,), device=augmented.device).item())
            if valid_frames - width <= 0:
                start = 0
            else:
                start = int(torch.randint(0, valid_frames - width + 1, (1,), device=augmented.device).item())
            augmented[batch_index, start : start + width, :] = 0.0

        for _ in range(freq_mask_count):
            max_width = min(n_mels, freq_mask_max_bins)
            if max_width <= 0:
                continue
            width = int(torch.randint(1, max_width + 1, (1,), device=augmented.device).item())
            if n_mels - width <= 0:
                start = 0
            else:
                start = int(torch.randint(0, n_mels - width + 1, (1,), device=augmented.device).item())
            augmented[batch_index, :, start : start + width] = 0.0

    return augmented
