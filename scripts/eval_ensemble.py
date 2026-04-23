"""Ensemble evaluation: average log_softmax from multiple models (each with TTA),
then grammar beam decode. Used to pick the final submission.
"""
from __future__ import annotations

import argparse
import sys
import time
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from asr_numbers.dataset import NumbersDataset, collate_batch
from asr_numbers.decoder import grammar_beam_decode
from asr_numbers.metrics import dataset_cer, domain_cer, speaker_cer
from asr_numbers.model import ConvGRUCTCModel
from asr_numbers.text import best_effort_number_from_text
from asr_numbers.tta import tta_forward
from asr_numbers.vocab import WordVocabulary

TTA_VARIANTS = [
    {"pitch_semitones": 0.5},
    {"pitch_semitones": -0.5},
    {"bandwidth_cutoff_hz": 5500.0},
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True, help="model config (same arch for all)")
    parser.add_argument("--checkpoints", nargs="+", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--beam-size", type=int, default=16)
    parser.add_argument("--tta", action="store_true")
    parser.add_argument("--dev-csv", type=Path, default=None)
    parser.add_argument("--audio-root", type=Path, default=None)
    parser.add_argument("--train-csv", type=Path, default=None)
    return parser.parse_args()


@torch.no_grad()
def main() -> None:
    args = parse_args()
    with args.config.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    vocab = WordVocabulary()
    models: list[ConvGRUCTCModel] = []
    for path in args.checkpoints:
        model = ConvGRUCTCModel(vocab_size=vocab.size, **config["model"])
        ck = torch.load(path, map_location="cpu")
        model.load_state_dict(ck["model_state"])
        model.eval()
        models.append(model)
        print(f"loaded {path}")

    dataset = NumbersDataset(
        csv_path=args.dev_csv or (ROOT / config["paths"]["dev_csv"]),
        audio_root=args.audio_root or (ROOT / config["paths"]["audio_root"]),
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

    train_frame = pd.read_csv(args.train_csv or (ROOT / config["paths"]["train_csv"]))
    seen = {s for s in train_frame.get("spk_id", []).tolist() if s}
    sample_rate = int(config["model"]["sample_rate"])

    refs: list[str] = []
    preds: list[str] = []
    speakers: list[str] = []
    t0 = time.time()

    for step, batch in enumerate(loader, start=1):
        waveforms = batch["waveforms"]
        wl = batch["waveform_lengths"]
        per_model_logp: list[torch.Tensor] = []
        per_model_len: list[torch.Tensor] = []
        for model in models:
            if args.tta:
                lp, ol = tta_forward(model, waveforms, wl, sample_rate, TTA_VARIANTS)
            else:
                logits, ol = model(waveforms, wl)
                lp = logits.log_softmax(dim=-1)
            per_model_logp.append(lp)
            per_model_len.append(ol)
        min_T = min(lp.size(1) for lp in per_model_logp)
        stacked = torch.stack([lp[:, :min_T, :] for lp in per_model_logp], dim=0)
        fused = stacked.mean(dim=0)
        out_len = torch.stack(per_model_len, dim=0).min(dim=0).values.clamp_max(min_T)
        sentences = grammar_beam_decode(fused, out_len, vocab, beam_size=args.beam_size)
        preds.extend(str(best_effort_number_from_text(s)) for s in sentences)
        refs.extend(batch["reference_digits"])
        speakers.extend(batch["spk_ids"])
        if step % 10 == 0:
            print(f"[progress] batch={step}")

    elapsed = time.time() - t0
    overall = dataset_cer(refs, preds)
    dom = domain_cer(refs, preds, speakers, seen_speakers=seen)
    spk = speaker_cer(refs, preds, speakers)
    print(f"[ensemble] dev_cer={overall:.4f} time={elapsed:.1f}s")
    for k, v in dom.items():
        print(f"  {k}={v:.4f}")
    for s, v in sorted(spk.items()):
        marker = "in" if s in seen else "OOD"
        print(f"  spk={s:8s} [{marker:3s}] cer={v:.4f}")


if __name__ == "__main__":
    main()
