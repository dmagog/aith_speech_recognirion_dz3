from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, WeightedRandomSampler

from .audio import load_audio
from .augment import apply_waveform_augmentations
from .text import normalize_transcription
from .vocab import WordVocabulary


class NumbersDataset(Dataset):
    def __init__(
        self,
        csv_path: str | Path,
        audio_root: str | Path | None = None,
        sample_rate: int = 16000,
        with_labels: bool = True,
        waveform_augment_cfg: dict | None = None,
    ) -> None:
        self.csv_path = Path(csv_path)
        self.audio_root = Path(audio_root) if audio_root is not None else self.csv_path.parent
        self.sample_rate = sample_rate
        self.with_labels = with_labels
        self.waveform_augment_cfg = waveform_augment_cfg or {}
        self.frame = pd.read_csv(self.csv_path)
        if "filename" not in self.frame.columns:
            raise ValueError(f"{self.csv_path} must contain a 'filename' column")
        if self.with_labels and "transcription" not in self.frame.columns:
            raise ValueError(f"{self.csv_path} must contain a 'transcription' column")

    def _resolve_audio_path(self, filename: str) -> Path:
        relative = Path(filename)
        candidates = [
            self.audio_root / relative,
            self.csv_path.parent / relative,
            self.csv_path.parent / relative.name,
        ]
        for candidate in candidates:
            if candidate.exists():
                return candidate
        return candidates[0]

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int) -> dict[str, object]:
        row = self.frame.iloc[index]
        filename = row["filename"]
        path = self._resolve_audio_path(filename)
        waveform, _ = load_audio(path, target_sr=self.sample_rate)
        if self.waveform_augment_cfg:
            rng = random.Random()
            augmented = apply_waveform_augmentations(
                waveform.numpy(), self.sample_rate, self.waveform_augment_cfg, rng
            )
            waveform = torch.from_numpy(np.ascontiguousarray(augmented))
        sample: dict[str, object] = {
            "filename": filename,
            "waveform": waveform,
            "waveform_length": len(waveform),
            "spk_id": row.get("spk_id", ""),
        }
        if self.with_labels:
            transcription = str(int(row["transcription"]))
            sample["reference_digits"] = transcription
            sample["normalized_text"] = normalize_transcription(transcription)
        return sample


def build_speaker_balanced_sampler(frame: pd.DataFrame, alpha: float = 0.5) -> WeightedRandomSampler:
    if "spk_id" not in frame.columns:
        raise ValueError("speaker-balanced sampler requires 'spk_id' column")
    counts = frame["spk_id"].value_counts().to_dict()
    if not counts:
        raise ValueError("cannot build sampler from empty dataset")
    weights_per_speaker = {spk: 1.0 / max(1.0, float(count) ** alpha) for spk, count in counts.items()}
    weights = frame["spk_id"].map(weights_per_speaker).astype(float).tolist()
    return WeightedRandomSampler(weights=weights, num_samples=len(frame), replacement=True)


def collate_batch(batch: list[dict[str, object]], vocab: WordVocabulary) -> dict[str, object]:
    waveforms = [item["waveform"] for item in batch]
    lengths = torch.tensor([int(item["waveform_length"]) for item in batch], dtype=torch.long)
    padded_waveforms = pad_sequence(waveforms, batch_first=True)

    result: dict[str, object] = {
        "waveforms": padded_waveforms,
        "waveform_lengths": lengths,
        "filenames": [str(item["filename"]) for item in batch],
        "spk_ids": [str(item["spk_id"]) for item in batch],
    }

    if "normalized_text" in batch[0]:
        target_sequences = [torch.tensor(vocab.encode(str(item["normalized_text"])), dtype=torch.long) for item in batch]
        target_lengths = torch.tensor([len(sequence) for sequence in target_sequences], dtype=torch.long)
        targets = torch.cat(target_sequences, dim=0)
        result["targets"] = targets
        result["target_lengths"] = target_lengths
        result["normalized_texts"] = [str(item["normalized_text"]) for item in batch]
        result["reference_digits"] = [str(item["reference_digits"]) for item in batch]
    return result
