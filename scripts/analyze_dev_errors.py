from __future__ import annotations

import argparse
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
from asr_numbers.decoder import decode_number_predictions
from asr_numbers.metrics import char_error_rate
from asr_numbers.model import ConvGRUCTCModel
from asr_numbers.vocab import WordVocabulary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--input-csv", type=Path, required=True)
    parser.add_argument("--audio-root", type=Path, default=None)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--focus-speakers", nargs="*", default=["spk_K", "spk_I"])
    return parser.parse_args()


def error_bucket(reference: str, prediction: str) -> str:
    if len(reference) != len(prediction):
        return "length_mismatch"
    if len(reference) > 3 and reference[:-3] != prediction[:-3]:
        return "thousands_or_higher"
    if reference[-3:] != prediction[-3:]:
        return "last_3_digits"
    return "other_same_length"


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

    dataset = NumbersDataset(
        csv_path=args.input_csv,
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

    rows: list[dict[str, object]] = []
    for batch in loader:
        logits, output_lengths = model(batch["waveforms"], batch["waveform_lengths"])
        predictions = decode_number_predictions(logits, output_lengths, vocab)
        for filename, speaker, reference, prediction in zip(
            batch["filenames"],
            batch["spk_ids"],
            batch["reference_digits"],
            predictions,
            strict=True,
        ):
            rows.append(
                {
                    "filename": filename,
                    "spk_id": speaker,
                    "reference": reference,
                    "prediction": prediction,
                    "cer": char_error_rate(reference, prediction),
                    "bucket": error_bucket(reference, prediction),
                }
            )

    frame = pd.DataFrame(rows)
    print("[speaker_cer]")
    print(frame.groupby("spk_id")["cer"].mean().sort_values(ascending=False).to_string())

    print("\n[buckets]")
    print(frame.groupby(["spk_id", "bucket"]).size().to_string())

    for speaker in args.focus_speakers:
        part = frame[frame["spk_id"] == speaker].sort_values(["cer", "filename"], ascending=[False, True]).head(args.top_k)
        if part.empty:
            continue
        print(f"\n[top_errors:{speaker}]")
        print(part[["filename", "reference", "prediction", "cer", "bucket"]].to_string(index=False))


if __name__ == "__main__":
    main()
