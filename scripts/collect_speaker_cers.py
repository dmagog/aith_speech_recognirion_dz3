"""Collect per-speaker CER for the 4 key milestones and save as JSON."""
from __future__ import annotations

import json
import sys
from functools import partial
from pathlib import Path

import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from asr_numbers.dataset import NumbersDataset, collate_batch
from asr_numbers.decoder import decode_number_predictions, grammar_beam_decode
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


def run_single(checkpoint_path: Path, use_tta: bool, use_beam: bool, config_path: Path):
    with config_path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    vocab = WordVocabulary()
    model = ConvGRUCTCModel(vocab_size=vocab.size, **config["model"])
    ck = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(ck["model_state"])
    model.eval()

    dataset = NumbersDataset(
        csv_path=ROOT / config["paths"]["dev_csv"],
        audio_root=ROOT / config["paths"]["audio_root"],
        sample_rate=int(config["model"]["sample_rate"]),
        with_labels=True,
    )
    loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=0,
                        collate_fn=partial(collate_batch, vocab=vocab))
    sr = int(config["model"]["sample_rate"])

    refs, preds, speakers = [], [], []
    with torch.no_grad():
        for batch in loader:
            if use_tta:
                log_probs, out_len = tta_forward(model, batch["waveforms"], batch["waveform_lengths"], sr, TTA_VARIANTS)
                logits = log_probs
            else:
                logits, out_len = model(batch["waveforms"], batch["waveform_lengths"])
            if use_beam:
                if not use_tta:
                    logits = logits.log_softmax(dim=-1)
                sentences = grammar_beam_decode(logits, out_len, vocab, beam_size=16)
                batch_preds = [str(best_effort_number_from_text(s)) for s in sentences]
            else:
                batch_preds = decode_number_predictions(logits, out_len, vocab)
            preds.extend(batch_preds)
            refs.extend(batch["reference_digits"])
            speakers.extend(batch["spk_ids"])
    return refs, preds, speakers


def run_ensemble(checkpoint_paths: list[Path], use_tta: bool, config_path: Path):
    with config_path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    vocab = WordVocabulary()
    models = []
    for p in checkpoint_paths:
        m = ConvGRUCTCModel(vocab_size=vocab.size, **config["model"])
        ck = torch.load(p, map_location="cpu")
        m.load_state_dict(ck["model_state"])
        m.eval()
        models.append(m)

    dataset = NumbersDataset(
        csv_path=ROOT / config["paths"]["dev_csv"],
        audio_root=ROOT / config["paths"]["audio_root"],
        sample_rate=int(config["model"]["sample_rate"]),
        with_labels=True,
    )
    loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=0,
                        collate_fn=partial(collate_batch, vocab=vocab))
    sr = int(config["model"]["sample_rate"])

    refs, preds, speakers = [], [], []
    with torch.no_grad():
        for batch in loader:
            per_m_lp, per_m_len = [], []
            for mdl in models:
                if use_tta:
                    lp, ol = tta_forward(mdl, batch["waveforms"], batch["waveform_lengths"], sr, TTA_VARIANTS)
                else:
                    logits, ol = mdl(batch["waveforms"], batch["waveform_lengths"])
                    lp = logits.log_softmax(dim=-1)
                per_m_lp.append(lp); per_m_len.append(ol)
            min_T = min(lp.size(1) for lp in per_m_lp)
            fused = torch.stack([lp[:, :min_T, :] for lp in per_m_lp], dim=0).mean(dim=0)
            out_len = torch.stack(per_m_len, dim=0).min(dim=0).values.clamp_max(min_T)
            sentences = grammar_beam_decode(fused, out_len, vocab, beam_size=16)
            batch_preds = [str(best_effort_number_from_text(s)) for s in sentences]
            preds.extend(batch_preds)
            refs.extend(batch["reference_digits"])
            speakers.extend(batch["spk_ids"])
    return refs, preds, speakers


def summarise(refs, preds, speakers, seen):
    return {
        "overall": dataset_cer(refs, preds),
        "domains": domain_cer(refs, preds, speakers, seen_speakers=seen),
        "per_speaker": speaker_cer(refs, preds, speakers),
    }


def main():
    train_frame = pd.read_csv(ROOT / "data" / "train" / "train.csv")
    seen = {s for s in train_frame.get("spk_id", []).tolist() if s}

    milestones = [
        ("baseline", {
            "config": "configs/baseline_quick.yaml",
            "checkpoints": ["outputs/baseline_quick/best.pt"],
            "tta": False, "beam": False,
        }),
        ("scratch_swa_tta", {
            "config": "configs/scratch_full_aug.yaml",
            "checkpoints": ["outputs/scratch_full_aug/swa_16_18_19_20.pt"],
            "tta": True, "beam": True,
        }),
        ("reverb_swa_tta", {
            "config": "configs/reverb_cont.yaml",
            "checkpoints": ["outputs/reverb_cont/swa_23_25_26.pt"],
            "tta": True, "beam": True,
        }),
        ("ensemble_tta", {
            "config": "configs/reverb_cont.yaml",
            "checkpoints": ["outputs/reverb_cont/swa_23_25_26.pt",
                            "outputs/scratch_reverb_seed43/epoch20.pt"],
            "tta": True, "beam": True,
        }),
    ]

    out = {}
    for name, cfg in milestones:
        print(f"[{name}] starting...")
        cks = [ROOT / p for p in cfg["checkpoints"]]
        if len(cks) == 1:
            refs, preds, speakers = run_single(cks[0], cfg["tta"], cfg["beam"], ROOT / cfg["config"])
        else:
            refs, preds, speakers = run_ensemble(cks, cfg["tta"], ROOT / cfg["config"])
        out[name] = summarise(refs, preds, speakers, seen)
        print(f"[{name}] overall={out[name]['overall']:.4f}")

    save_to = ROOT / "outputs" / "report_figures" / "milestones_cer.json"
    save_to.parent.mkdir(parents=True, exist_ok=True)
    with save_to.open("w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"saved -> {save_to}")


if __name__ == "__main__":
    main()
