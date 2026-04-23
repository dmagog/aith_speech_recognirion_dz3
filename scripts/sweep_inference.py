"""Sweep over beam sizes and TTA variants for the current best checkpoint."""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from collections import Counter
from functools import partial
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from asr_numbers.dataset import NumbersDataset, collate_batch
from asr_numbers.decoder import grammar_beam_decode, greedy_decode_words
from asr_numbers.metrics import dataset_cer, domain_cer, speaker_cer
from asr_numbers.model import ConvGRUCTCModel
from asr_numbers.text import best_effort_number_from_text
from asr_numbers.tta import tta_forward
from asr_numbers.vocab import WordVocabulary


TTA_SETS: dict[str, list[dict]] = {
    "none": [],
    "basic": [
        {"pitch_semitones": 0.5},
        {"pitch_semitones": -0.5},
        {"bandwidth_cutoff_hz": 5500.0},
    ],
    "wide": [
        {"pitch_semitones": 0.5},
        {"pitch_semitones": -0.5},
        {"pitch_semitones": 1.0},
        {"pitch_semitones": -1.0},
        {"bandwidth_cutoff_hz": 5500.0},
        {"bandwidth_cutoff_hz": 4500.0},
        {"bandwidth_cutoff_hz": 6500.0},
    ],
    "small": [
        {"pitch_semitones": 0.3},
        {"pitch_semitones": -0.3},
        {"bandwidth_cutoff_hz": 5500.0},
    ],
    "bandwidth_only": [
        {"bandwidth_cutoff_hz": 4500.0},
        {"bandwidth_cutoff_hz": 5500.0},
        {"bandwidth_cutoff_hz": 6500.0},
    ],
    "pitch_only": [
        {"pitch_semitones": 0.5},
        {"pitch_semitones": -0.5},
        {"pitch_semitones": 1.0},
        {"pitch_semitones": -1.0},
    ],
}


def run_eval(
    model: ConvGRUCTCModel,
    loader: DataLoader,
    vocab: WordVocabulary,
    sample_rate: int,
    beam_size: int,
    tta_variants: list[dict],
    fusion: str = "mean",
    voting: bool = False,
) -> tuple[list[str], list[str], list[str], float]:
    references: list[str] = []
    predictions: list[str] = []
    speakers: list[str] = []
    t0 = time.time()

    for batch in loader:
        waveforms = batch["waveforms"]
        waveform_lengths = batch["waveform_lengths"]

        if voting and tta_variants:
            all_numbers: list[list[str]] = [[] for _ in range(waveforms.size(0))]
            configs = [[]] + [[v] for v in tta_variants]
            for cfg in configs:
                logp, out_len = tta_forward(model, waveforms, waveform_lengths, sample_rate, cfg, fusion)
                sentences = grammar_beam_decode(logp, out_len, vocab, beam_size=beam_size)
                for idx, s in enumerate(sentences):
                    all_numbers[idx].append(str(best_effort_number_from_text(s)))
            for per_sample in all_numbers:
                counts = Counter(per_sample)
                best = counts.most_common(1)[0][0]
                predictions.append(best)
        else:
            logp, out_len = tta_forward(model, waveforms, waveform_lengths, sample_rate, tta_variants, fusion)
            sentences = grammar_beam_decode(logp, out_len, vocab, beam_size=beam_size)
            predictions.extend(str(best_effort_number_from_text(s)) for s in sentences)

        references.extend(batch["reference_digits"])
        speakers.extend(batch["spk_ids"])
    return references, predictions, speakers, time.time() - t0


def summary(name: str, refs: list[str], preds: list[str], speakers: list[str], seen: set[str], elapsed: float) -> dict:
    overall = dataset_cer(refs, preds)
    dom = domain_cer(refs, preds, speakers, seen_speakers=seen)
    spk = speaker_cer(refs, preds, speakers)
    info = {
        "name": name,
        "elapsed": round(elapsed, 1),
        "dev_cer": overall,
        **{k: v for k, v in dom.items()},
        "spk_H": spk.get("spk_H", 0.0),
        "spk_K": spk.get("spk_K", 0.0),
    }
    print(json.dumps({k: (round(v, 5) if isinstance(v, float) else v) for k, v in info.items()}, ensure_ascii=False))
    return info


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--beam-sizes", nargs="+", type=int, default=[16, 32, 64])
    parser.add_argument("--tta-sets", nargs="+", default=["none", "basic", "wide", "small"])
    parser.add_argument("--include-voting", action="store_true")
    parser.add_argument("--max-batches", type=int, default=None)
    return parser.parse_args()


@torch.no_grad()
def main() -> None:
    args = parse_args()
    with args.config.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    vocab = WordVocabulary()
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    model = ConvGRUCTCModel(vocab_size=vocab.size, **config["model"])
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    dataset = NumbersDataset(
        csv_path=ROOT / config["paths"]["dev_csv"],
        audio_root=ROOT / config["paths"]["audio_root"],
        sample_rate=int(config["model"]["sample_rate"]),
        with_labels=True,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=partial(collate_batch, vocab=vocab),
    )
    import pandas as pd

    train_frame = pd.read_csv(ROOT / config["paths"]["train_csv"])
    seen = {s for s in train_frame.get("spk_id", []).tolist() if s}
    sample_rate = int(config["model"]["sample_rate"])

    results: list[dict] = []
    for set_name in args.tta_sets:
        variants = TTA_SETS[set_name]
        for beam in args.beam_sizes:
            key = f"tta={set_name}|beam={beam}"
            refs, preds, speakers, elapsed = run_eval(
                model, loader, vocab, sample_rate, beam, variants, fusion="mean", voting=False
            )
            results.append(summary(key, refs, preds, speakers, seen, elapsed))
            if args.include_voting and variants:
                key_v = f"tta={set_name}|beam={beam}|voting"
                refs, preds, speakers, elapsed = run_eval(
                    model, loader, vocab, sample_rate, beam, variants, fusion="mean", voting=True
                )
                results.append(summary(key_v, refs, preds, speakers, seen, elapsed))

    print("\n=== summary (sorted by dev_cer) ===")
    for r in sorted(results, key=lambda r: r["dev_cer"]):
        print(
            f"{r['name']:45s}  dev={r['dev_cer']:.4f}  harm={r.get('harmonic_cer', 0):.4f}  "
            f"ood={r.get('ood_cer', 0):.4f}  spk_K={r['spk_K']:.4f}  t={r['elapsed']}s"
        )


if __name__ == "__main__":
    main()
