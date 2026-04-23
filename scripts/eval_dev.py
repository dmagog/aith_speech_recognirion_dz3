from __future__ import annotations

import argparse
import sys
import time
from functools import partial
from pathlib import Path

import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from asr_numbers.dataset import NumbersDataset, collate_batch
from asr_numbers.decoder import (
    decode_number_predictions,
    decode_number_predictions_beam,
    grammar_beam_decode,
    greedy_decode_words,
)
from asr_numbers.metrics import dataset_cer, domain_cer, speaker_cer
from asr_numbers.model import ConvGRUCTCModel
from asr_numbers.tta import tta_forward
from asr_numbers.text import best_effort_number_from_text
from asr_numbers.vocab import WordVocabulary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dev-csv", type=Path, default=None)
    parser.add_argument("--audio-root", type=Path, default=None)
    parser.add_argument("--train-csv-for-domains", type=Path, default=None)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--decoder", choices=["greedy", "beam", "both"], default="both")
    parser.add_argument("--beam-size", type=int, default=16)
    parser.add_argument("--word-insertion-penalty", type=float, default=0.0)
    parser.add_argument("--max-batches", type=int, default=None)
    parser.add_argument(
        "--tta",
        action="store_true",
        help="Average log-softmax from multiple audio augmentations before decoding",
    )
    return parser.parse_args()


TTA_VARIANTS = [
    {"pitch_semitones": 0.5},
    {"pitch_semitones": -0.5},
    {"bandwidth_cutoff_hz": 5500.0},
]


@torch.no_grad()
def main() -> None:
    args = parse_args()
    with args.config.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    vocab = WordVocabulary()
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    model = ConvGRUCTCModel(vocab_size=vocab.size, **config["model"])
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    dev_csv = args.dev_csv or (ROOT / config["paths"]["dev_csv"])
    audio_root = args.audio_root or (ROOT / config["paths"]["audio_root"])
    dataset = NumbersDataset(
        csv_path=dev_csv,
        audio_root=audio_root,
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

    train_csv = args.train_csv_for_domains or (ROOT / config["paths"]["train_csv"])
    train_frame = pd.read_csv(train_csv)
    seen_speakers = {s for s in train_frame.get("spk_id", []).tolist() if s}

    references: list[str] = []
    speakers: list[str] = []
    greedy_predictions: list[str] = []
    beam_predictions: list[str] = []

    greedy_time = 0.0
    beam_time = 0.0

    sample_rate = int(config["model"]["sample_rate"])
    for step, batch in enumerate(loader, start=1):
        if args.max_batches is not None and step > args.max_batches:
            break
        if args.tta:
            log_probs, output_lengths = tta_forward(
                model,
                batch["waveforms"],
                batch["waveform_lengths"],
                sample_rate,
                TTA_VARIANTS,
            )
            logits = log_probs  # already log-softmax; argmax / grammar_beam still work
        else:
            logits, output_lengths = model(batch["waveforms"], batch["waveform_lengths"])
        if args.decoder in ("greedy", "both"):
            t0 = time.time()
            if args.tta:
                words = greedy_decode_words(logits, output_lengths, vocab)
                preds = [str(best_effort_number_from_text(w)) for w in words]
            else:
                preds = decode_number_predictions(logits, output_lengths, vocab)
            greedy_time += time.time() - t0
            greedy_predictions.extend(preds)
        if args.decoder in ("beam", "both"):
            t0 = time.time()
            if args.tta:
                sentences = grammar_beam_decode(
                    logits,
                    output_lengths,
                    vocab,
                    beam_size=args.beam_size,
                    word_insertion_penalty=args.word_insertion_penalty,
                )
                preds = [str(best_effort_number_from_text(s)) for s in sentences]
            else:
                preds = decode_number_predictions_beam(
                    logits,
                    output_lengths,
                    vocab,
                    beam_size=args.beam_size,
                    word_insertion_penalty=args.word_insertion_penalty,
                )
            beam_time += time.time() - t0
            beam_predictions.extend(preds)
        references.extend(batch["reference_digits"])
        speakers.extend(batch["spk_ids"])
        if step % 10 == 0:
            print(f"[progress] batch={step} total_samples={len(references)}", flush=True)

    def report(name: str, preds: list[str], elapsed: float) -> None:
        if not preds:
            return
        total_cer = dataset_cer(references, preds)
        spk = speaker_cer(references, preds, speakers)
        dom = domain_cer(references, preds, speakers, seen_speakers=seen_speakers)
        print(f"[{name}] dev_cer={total_cer:.4f} time={elapsed:.1f}s")
        for key, value in dom.items():
            print(f"  {key}={value:.4f}")
        for s, value in sorted(spk.items()):
            marker = "in" if s in seen_speakers else "OOD"
            print(f"  spk={s:8s} [{marker:3s}] cer={value:.4f}")

    if args.decoder in ("greedy", "both"):
        report("greedy", greedy_predictions, greedy_time)
    if args.decoder in ("beam", "both"):
        report(f"beam{args.beam_size}", beam_predictions, beam_time)


if __name__ == "__main__":
    main()
