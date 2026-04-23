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
from asr_numbers.model import ConvGRUCTCModel
from asr_numbers.vocab import WordVocabulary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--input-csv", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--audio-root", type=Path, default=None)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", type=str, default="auto")
    return parser.parse_args()


def resolve_device(device_name: str) -> torch.device:
    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_name)


@torch.no_grad()
def main() -> None:
    args = parse_args()
    with args.config.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    device = resolve_device(args.device)
    vocab = WordVocabulary()
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model = ConvGRUCTCModel(vocab_size=vocab.size, **config["model"]).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    dataset = NumbersDataset(
        csv_path=args.input_csv,
        audio_root=args.audio_root or (ROOT / config["paths"]["audio_root"]),
        sample_rate=int(config["model"]["sample_rate"]),
        with_labels=False,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=partial(collate_batch, vocab=vocab),
    )

    rows: list[dict[str, str]] = []
    for batch in loader:
        waveforms = batch["waveforms"].to(device)
        waveform_lengths = batch["waveform_lengths"].to(device)
        logits, output_lengths = model(waveforms, waveform_lengths)
        predictions = decode_number_predictions(logits.cpu(), output_lengths.cpu(), vocab)
        for filename, transcription in zip(batch["filenames"], predictions, strict=True):
            rows.append({"filename": filename, "transcription": transcription})

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.output_csv, index=False)
    print(f"[done] wrote {len(rows)} predictions to {args.output_csv}")


if __name__ == "__main__":
    main()
