"""Test-time augmentation utilities.

Applies length-preserving audio transforms to each waveform, forwards each
variant through the model, and returns averaged log-softmax outputs.
"""

from __future__ import annotations

import numpy as np
import torch

from .augment import bandwidth_limit_numpy, pitch_shift_numpy


def _to_tensor(waveform_np: np.ndarray, device: torch.device) -> torch.Tensor:
    return torch.from_numpy(np.ascontiguousarray(waveform_np, dtype=np.float32)).to(device)


def build_tta_variants(
    waveforms: torch.Tensor,
    lengths: torch.Tensor,
    sample_rate: int,
    variants: list[dict],
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """Return list of (padded_waveforms, lengths) for each TTA variant.

    Each transform preserves the per-sample length, so lengths are unchanged
    across variants. The returned tensors are cloned from the input.
    """
    results: list[tuple[torch.Tensor, torch.Tensor]] = []
    device = waveforms.device
    np_waveforms = [waveforms[i, : int(lengths[i].item())].cpu().numpy() for i in range(waveforms.size(0))]

    for variant in variants:
        transformed: list[np.ndarray] = []
        for waveform in np_waveforms:
            wav = waveform
            semitones = float(variant.get("pitch_semitones", 0.0))
            if abs(semitones) > 1e-3:
                wav = pitch_shift_numpy(wav, sample_rate, semitones)
            cutoff = float(variant.get("bandwidth_cutoff_hz", 0.0))
            if cutoff > 0:
                wav = bandwidth_limit_numpy(wav, sample_rate, cutoff)
            transformed.append(np.clip(wav, -1.0, 1.0).astype(np.float32, copy=False))

        max_len = max(len(w) for w in transformed)
        padded = np.zeros((len(transformed), max_len), dtype=np.float32)
        for i, w in enumerate(transformed):
            padded[i, : len(w)] = w
        tensor = torch.from_numpy(padded).to(device)
        results.append((tensor, lengths.clone()))

    return results


@torch.no_grad()
def tta_forward(
    model: torch.nn.Module,
    waveforms: torch.Tensor,
    lengths: torch.Tensor,
    sample_rate: int,
    variants: list[dict],
    fusion: str = "mean",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run TTA forward, return (log_probs_sum, output_lengths).

    All variants produce identical output lengths because input lengths are
    preserved by the pitch / bandpass transforms.
    """
    if not variants:
        logits, output_lengths = model(waveforms, lengths)
        return logits.log_softmax(dim=-1), output_lengths

    tta_inputs = build_tta_variants(waveforms, lengths, sample_rate, variants)
    tta_inputs.insert(0, (waveforms, lengths))

    log_probs_list: list[torch.Tensor] = []
    output_lengths_ref: torch.Tensor | None = None
    for wf, ln in tta_inputs:
        logits, out_len = model(wf, ln)
        log_probs = logits.log_softmax(dim=-1)
        if output_lengths_ref is None:
            output_lengths_ref = out_len
        else:
            clamp = torch.minimum(output_lengths_ref, out_len)
            output_lengths_ref = clamp
        log_probs_list.append(log_probs)

    assert output_lengths_ref is not None
    T = min(lp.size(1) for lp in log_probs_list)
    stacked = torch.stack([lp[:, :T, :] for lp in log_probs_list], dim=0)
    if fusion == "mean":
        fused = stacked.mean(dim=0)
    elif fusion == "logsumexp":
        fused = torch.logsumexp(stacked, dim=0) - float(np.log(len(log_probs_list)))
    else:
        raise ValueError(f"unknown fusion: {fusion}")
    return fused, output_lengths_ref.clamp_max(T)
