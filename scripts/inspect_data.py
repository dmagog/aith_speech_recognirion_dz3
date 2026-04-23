from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-csv", type=Path, required=True)
    parser.add_argument("--dev-csv", type=Path, required=True)
    parser.add_argument("--test-csv", type=Path, default=None)
    return parser.parse_args()


def summarize_split(name: str, frame: pd.DataFrame) -> dict[str, object]:
    summary: dict[str, object] = {
        "rows": len(frame),
        "speakers": sorted(frame["spk_id"].dropna().astype(str).unique().tolist()) if "spk_id" in frame.columns else [],
        "genders": frame["gender"].value_counts(dropna=False).to_dict() if "gender" in frame.columns else {},
        "exts": frame["ext"].value_counts(dropna=False).to_dict() if "ext" in frame.columns else {},
        "sample_rates": frame["samplerate"].value_counts(dropna=False).sort_index().to_dict() if "samplerate" in frame.columns else {},
    }
    if "transcription" in frame.columns:
        numbers = frame["transcription"].astype(int)
        summary["min_number"] = int(numbers.min())
        summary["max_number"] = int(numbers.max())
        summary["num_unique_transcriptions"] = int(numbers.nunique())
    print(f"[{name}] rows={summary['rows']}")
    print(f"[{name}] speakers={summary['speakers']}")
    print(f"[{name}] genders={summary['genders']}")
    print(f"[{name}] exts={summary['exts']}")
    print(f"[{name}] sample_rates={summary['sample_rates']}")
    if "min_number" in summary:
        print(
            f"[{name}] number_range={summary['min_number']}..{summary['max_number']} "
            f"unique={summary['num_unique_transcriptions']}"
        )
    return summary


def main() -> None:
    args = parse_args()
    train = pd.read_csv(args.train_csv)
    dev = pd.read_csv(args.dev_csv)
    test = pd.read_csv(args.test_csv) if args.test_csv is not None and args.test_csv.exists() else None

    train_summary = summarize_split("train", train)
    dev_summary = summarize_split("dev", dev)
    if test is not None:
        summarize_split("test", test)

    train_speakers = set(train_summary["speakers"])
    dev_speakers = set(dev_summary["speakers"])
    seen_in_dev = sorted(train_speakers & dev_speakers)
    unseen_in_dev = sorted(dev_speakers - train_speakers)
    print(f"[domain] dev_seen_speakers={seen_in_dev}")
    print(f"[domain] dev_unseen_speakers={unseen_in_dev}")


if __name__ == "__main__":
    main()
