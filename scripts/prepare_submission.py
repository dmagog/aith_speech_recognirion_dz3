from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions-csv", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    frame = pd.read_csv(args.predictions_csv)
    required_columns = {"filename", "transcription"}
    missing = required_columns - set(frame.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    frame = frame.loc[:, ["filename", "transcription"]].copy()
    frame["transcription"] = frame["transcription"].astype(str).str.replace(r"\.0$", "", regex=True)

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output_csv, index=False)
    print(f"[done] wrote submission to {args.output_csv}")


if __name__ == "__main__":
    main()
