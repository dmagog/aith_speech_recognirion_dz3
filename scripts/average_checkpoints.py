from __future__ import annotations

import argparse
from pathlib import Path

import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Average model weights from multiple checkpoints.")
    parser.add_argument("--checkpoints", nargs="+", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--template", type=Path, default=None, help="Checkpoint to use as metadata template (default: first)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.checkpoints:
        raise SystemExit("need at least one checkpoint")

    template_path = args.template or args.checkpoints[0]
    template = torch.load(template_path, map_location="cpu")

    accumulator: dict[str, torch.Tensor] | None = None
    count = 0
    for path in args.checkpoints:
        state = torch.load(path, map_location="cpu")
        model_state = state["model_state"]
        if accumulator is None:
            accumulator = {k: v.detach().clone().float() for k, v in model_state.items()}
        else:
            for k, v in model_state.items():
                accumulator[k] += v.detach().float()
        count += 1
        print(f"merged {path} (count={count})")

    assert accumulator is not None
    averaged = {k: (v / count).to(dtype=torch.float32) for k, v in accumulator.items()}
    template["model_state"] = averaged
    template["history"] = template.get("history", []) + [{"averaged_from": [str(p) for p in args.checkpoints]}]
    template["best_cer"] = float(template.get("best_cer", 0.0))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(template, args.output)
    print(f"wrote averaged checkpoint -> {args.output}")


if __name__ == "__main__":
    main()
