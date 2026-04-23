from __future__ import annotations

import io
import math
import random

import numpy as np
import soundfile as sf
from scipy.signal import butter, resample_poly, sosfiltfilt

try:
    import lameenc  # type: ignore

    _HAS_LAMEENC = True
except ImportError:
    _HAS_LAMEENC = False


def _ensure_numpy(waveform) -> np.ndarray:
    if hasattr(waveform, "detach"):
        waveform = waveform.detach().cpu().numpy()
    return np.ascontiguousarray(waveform, dtype=np.float32)


def mp3_roundtrip_numpy(waveform: np.ndarray, sample_rate: int, bitrate_kbps: int) -> np.ndarray:
    if not _HAS_LAMEENC:
        return waveform
    encoder = lameenc.Encoder()
    encoder.set_bit_rate(bitrate_kbps)
    encoder.set_in_sample_rate(sample_rate)
    encoder.set_channels(1)
    encoder.set_quality(5)
    pcm = np.clip(waveform, -1.0, 1.0)
    pcm_i16 = (pcm * 32767.0).astype("<i2").tobytes()
    mp3_bytes = encoder.encode(pcm_i16) + encoder.flush()
    decoded, decoded_sr = sf.read(io.BytesIO(mp3_bytes), always_2d=False)
    if decoded.ndim == 2:
        decoded = decoded.mean(axis=1)
    decoded = decoded.astype(np.float32, copy=False)
    if decoded_sr != sample_rate:
        divisor = math.gcd(decoded_sr, sample_rate)
        decoded = resample_poly(decoded, sample_rate // divisor, decoded_sr // divisor).astype(np.float32)
    if len(decoded) >= len(waveform):
        return decoded[: len(waveform)]
    padded = np.zeros_like(waveform)
    padded[: len(decoded)] = decoded
    return padded


def _lowpass_sos(cutoff_hz: float, sample_rate: int, order: int = 6):
    nyquist = 0.5 * sample_rate
    normalized = max(0.05, min(0.95, cutoff_hz / nyquist))
    return butter(order, normalized, btype="low", output="sos")


def bandwidth_limit_numpy(
    waveform: np.ndarray,
    sample_rate: int,
    cutoff_hz: float,
) -> np.ndarray:
    if cutoff_hz <= 0 or cutoff_hz >= sample_rate / 2:
        return waveform
    sos = _lowpass_sos(cutoff_hz, sample_rate)
    return sosfiltfilt(sos, waveform).astype(np.float32, copy=False)


def reverb_numpy(
    waveform: np.ndarray,
    sample_rate: int,
    rt60_sec: float,
    wet_mix: float,
    rng: random.Random | None = None,
) -> np.ndarray:
    """Simulate simple reverberation by convolving with an exponentially
    decaying noise impulse response. Preserves input length.

    rt60_sec: approximate reverberation time in seconds (20-300 ms works).
    wet_mix:  0..1 fraction of wet signal mixed with the dry input.
    """
    if rt60_sec <= 0.001 or wet_mix <= 0.0:
        return waveform.astype(np.float32, copy=False)
    length = int(max(4, round(rt60_sec * sample_rate)))
    tau = max(1.0, rt60_sec * sample_rate / 6.0)
    t = np.arange(length, dtype=np.float32)
    rng_state = rng if rng is not None else random.Random()
    seed = rng_state.randint(0, 2**31 - 1)
    noise = np.random.default_rng(seed).standard_normal(length).astype(np.float32)
    ir = np.exp(-t / tau).astype(np.float32) * noise
    ir[0] = 1.0  # ensure direct path
    ir /= max(1e-6, float(np.sqrt(np.sum(ir * ir))))
    wet = np.convolve(waveform, ir, mode="full")[: len(waveform)]
    # Match RMS of dry to keep gain stable.
    dry_rms = float(np.sqrt(np.mean(waveform * waveform) + 1e-8))
    wet_rms = float(np.sqrt(np.mean(wet * wet) + 1e-8))
    if wet_rms > 1e-6:
        wet = wet * (dry_rms / wet_rms)
    mixed = (1.0 - wet_mix) * waveform + wet_mix * wet
    return np.clip(mixed.astype(np.float32), -1.0, 1.0)


def pitch_shift_numpy(
    waveform: np.ndarray,
    sample_rate: int,
    semitones: float,
) -> np.ndarray:
    if abs(semitones) < 1e-3:
        return waveform
    rate = float(2.0 ** (semitones / 12.0))
    stretch_up = max(1, int(round(sample_rate * rate / 10.0)))
    stretch_down = max(1, int(round(sample_rate / 10.0)))
    stretched = resample_poly(waveform, stretch_up, stretch_down).astype(np.float32, copy=False)
    if len(stretched) == len(waveform):
        return stretched
    original_length = len(waveform)
    if len(stretched) > original_length:
        start = (len(stretched) - original_length) // 2
        return stretched[start : start + original_length]
    padded = np.zeros(original_length, dtype=np.float32)
    start = (original_length - len(stretched)) // 2
    padded[start : start + len(stretched)] = stretched
    return padded


def apply_waveform_augmentations(
    waveform: np.ndarray,
    sample_rate: int,
    cfg: dict,
    rng: random.Random,
) -> np.ndarray:
    if waveform.size == 0:
        return waveform
    wav = waveform

    mp3_prob = float(cfg.get("mp3_prob", 0.0))
    if mp3_prob > 0.0 and rng.random() < mp3_prob and _HAS_LAMEENC:
        bitrates = cfg.get("mp3_bitrates", [64, 96, 128])
        if bitrates:
            bitrate = int(rng.choice(list(bitrates)))
            wav = mp3_roundtrip_numpy(wav, sample_rate, bitrate)

    bandwidth_prob = float(cfg.get("bandwidth_prob", 0.0))
    if bandwidth_prob > 0.0 and rng.random() < bandwidth_prob:
        low = float(cfg.get("bandwidth_cutoff_min_hz", 3000.0))
        high = float(cfg.get("bandwidth_cutoff_max_hz", 7000.0))
        cutoff = rng.uniform(low, high)
        wav = bandwidth_limit_numpy(wav, sample_rate, cutoff)

    pitch_prob = float(cfg.get("pitch_prob", 0.0))
    if pitch_prob > 0.0 and rng.random() < pitch_prob:
        low = float(cfg.get("pitch_semitones_min", -2.0))
        high = float(cfg.get("pitch_semitones_max", 2.0))
        semitones = rng.uniform(low, high)
        wav = pitch_shift_numpy(wav, sample_rate, semitones)

    reverb_prob = float(cfg.get("reverb_prob", 0.0))
    if reverb_prob > 0.0 and rng.random() < reverb_prob:
        rt60_min = float(cfg.get("reverb_rt60_min", 0.05))
        rt60_max = float(cfg.get("reverb_rt60_max", 0.3))
        wet_min = float(cfg.get("reverb_wet_min", 0.15))
        wet_max = float(cfg.get("reverb_wet_max", 0.55))
        rt60 = rng.uniform(rt60_min, rt60_max)
        wet_mix = rng.uniform(wet_min, wet_max)
        wav = reverb_numpy(wav, sample_rate, rt60, wet_mix, rng=rng)

    return np.clip(wav.astype(np.float32, copy=False), -1.0, 1.0)
