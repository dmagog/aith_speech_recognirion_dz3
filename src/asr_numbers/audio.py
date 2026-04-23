from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from scipy.signal import resample_poly


def _resample_waveform(waveform: np.ndarray, src_sr: int, target_sr: int) -> np.ndarray:
    if src_sr == target_sr:
        return waveform
    divisor = math.gcd(src_sr, target_sr)
    up = target_sr // divisor
    down = src_sr // divisor
    return resample_poly(waveform, up, down).astype(np.float32)


def load_audio(path: str | Path, target_sr: int = 16000) -> tuple[torch.Tensor, int]:
    path = Path(path)
    try:
        waveform, sample_rate = sf.read(path, always_2d=False)
    except RuntimeError as exc:
        raise RuntimeError(
            f"Failed to decode audio file {path}. "
            "If this is an mp3 file, ensure the local libsndfile build supports mp3."
        ) from exc

    if waveform.ndim == 2:
        waveform = waveform.mean(axis=1)
    waveform = waveform.astype(np.float32, copy=False)
    waveform = _resample_waveform(waveform, sample_rate, target_sr)
    waveform = np.clip(waveform, -1.0, 1.0)
    return torch.from_numpy(waveform), target_sr
