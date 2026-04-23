from __future__ import annotations

import argparse
import math
import random
import sys
from functools import partial
from pathlib import Path

import numpy as np
import torch
import yaml
from torch import nn
from torch.nn.utils import clip_grad_norm_
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from asr_numbers.dataset import NumbersDataset, collate_batch, build_speaker_balanced_sampler
from asr_numbers.decoder import decode_number_predictions
from asr_numbers.metrics import dataset_cer, domain_cer, speaker_cer
from asr_numbers.model import ConvGRUCTCModel, count_parameters
from asr_numbers.vocab import WordVocabulary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "baseline_ctc.yaml")
    parser.add_argument("--max-train-batches", type=int, default=None)
    parser.add_argument("--max-dev-batches", type=int, default=None)
    parser.add_argument("--resume-checkpoint", type=Path, default=None)
    parser.add_argument("--fresh-optimizer", action="store_true", help="Do not load optimizer state from checkpoint")
    parser.add_argument("--reset-best-cer", action="store_true", help="Treat resumed checkpoint's best_cer as unset")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(device_name: str) -> torch.device:
    if device_name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        # CTCLoss has no native MPS implementation as of torch 2.x; CPU is safer.
        return torch.device("cpu")
    return torch.device(device_name)


def maybe_augment_waveforms(waveforms: torch.Tensor, lengths: torch.Tensor, augment_cfg: dict[str, float]) -> torch.Tensor:
    del lengths
    augmented = waveforms.clone()
    batch_size = augmented.size(0)

    gain_prob = float(augment_cfg.get("gain_prob", 0.0))
    if gain_prob > 0:
        mask = torch.rand(batch_size, device=augmented.device) < gain_prob
        if mask.any():
            gain_db = torch.empty(mask.sum(), device=augmented.device).uniform_(
                float(augment_cfg.get("gain_db_min", -6.0)),
                float(augment_cfg.get("gain_db_max", 6.0)),
            )
            scales = torch.pow(torch.full_like(gain_db, 10.0), gain_db / 20.0)
            augmented[mask] = augmented[mask] * scales.unsqueeze(1)

    noise_prob = float(augment_cfg.get("noise_prob", 0.0))
    if noise_prob > 0:
        mask = torch.rand(batch_size, device=augmented.device) < noise_prob
        if mask.any():
            noise_std = torch.empty(mask.sum(), device=augmented.device).uniform_(
                float(augment_cfg.get("noise_std_min", 0.001)),
                float(augment_cfg.get("noise_std_max", 0.01)),
            )
            noise = torch.randn_like(augmented[mask]) * noise_std.unsqueeze(1)
            augmented[mask] = augmented[mask] + noise

    return augmented.clamp_(-1.0, 1.0)


def maybe_speed_perturb(
    waveforms: torch.Tensor,
    lengths: torch.Tensor,
    augment_cfg: dict[str, float],
) -> tuple[torch.Tensor, torch.Tensor]:
    speed_prob = float(augment_cfg.get("speed_prob", 0.0))
    if speed_prob <= 0:
        return waveforms, lengths

    min_rate = float(augment_cfg.get("speed_min_rate", 0.95))
    max_rate = float(augment_cfg.get("speed_max_rate", 1.05))
    if min_rate <= 0 or max_rate <= 0:
        raise ValueError("Speed perturbation rates must be positive")

    perturbed_waveforms: list[torch.Tensor] = []
    perturbed_lengths: list[int] = []
    for waveform, length in zip(waveforms, lengths.tolist(), strict=True):
        trimmed = waveform[:length]
        if trimmed.numel() <= 1 or random.random() >= speed_prob:
            perturbed_waveforms.append(trimmed)
            perturbed_lengths.append(int(length))
            continue

        rate = random.uniform(min_rate, max_rate)
        new_length = max(1, int(round(length / rate)))
        if new_length == length:
            perturbed_waveforms.append(trimmed)
            perturbed_lengths.append(int(length))
            continue

        resized = torch.nn.functional.interpolate(
            trimmed.view(1, 1, -1),
            size=new_length,
            mode="linear",
            align_corners=False,
        ).view(-1)
        perturbed_waveforms.append(resized.clamp(-1.0, 1.0))
        perturbed_lengths.append(new_length)

    return (
        pad_sequence(perturbed_waveforms, batch_first=True),
        torch.tensor(perturbed_lengths, dtype=lengths.dtype, device=lengths.device),
    )


def set_lr(optimizer: torch.optim.Optimizer, lr: float) -> None:
    for group in optimizer.param_groups:
        group["lr"] = lr


def compute_warmup_cosine_lr(
    step: int,
    total_steps: int,
    peak_lr: float,
    warmup_steps: int,
    min_lr_ratio: float = 0.05,
) -> float:
    if warmup_steps > 0 and step < warmup_steps:
        return peak_lr * (step + 1) / warmup_steps
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    progress = min(max(progress, 0.0), 1.0)
    min_lr = peak_lr * min_lr_ratio
    return min_lr + 0.5 * (peak_lr - min_lr) * (1.0 + math.cos(math.pi * progress))


def train_one_epoch(
    model: ConvGRUCTCModel,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.CTCLoss,
    device: torch.device,
    scaler: torch.cuda.amp.GradScaler | None,
    print_every: int,
    grad_clip: float,
    augment_cfg: dict[str, float],
    max_batches: int | None = None,
    scheduler_ctx: dict | None = None,
) -> float:
    model.train()
    total_loss = 0.0

    for step, batch in enumerate(loader, start=1):
        if max_batches is not None and step > max_batches:
            break
        if scheduler_ctx is not None:
            global_step = scheduler_ctx["global_step"]
            lr = compute_warmup_cosine_lr(
                global_step,
                scheduler_ctx["total_steps"],
                scheduler_ctx["peak_lr"],
                scheduler_ctx["warmup_steps"],
                scheduler_ctx.get("min_lr_ratio", 0.05),
            )
            set_lr(optimizer, lr)
            scheduler_ctx["global_step"] = global_step + 1
        waveforms = batch["waveforms"].to(device)
        waveform_lengths = batch["waveform_lengths"].to(device)
        targets = batch["targets"].to(device)
        target_lengths = batch["target_lengths"].to(device)

        waveforms, waveform_lengths = maybe_speed_perturb(waveforms, waveform_lengths, augment_cfg)
        waveforms = maybe_augment_waveforms(waveforms, waveform_lengths, augment_cfg)
        optimizer.zero_grad(set_to_none=True)
        spec_augment_cfg = dict(augment_cfg.get("spec_augment", {}))

        if scaler is None:
            logits, output_lengths = model(waveforms, waveform_lengths, spec_augment_config=spec_augment_cfg)
            log_probs = logits.log_softmax(dim=-1).transpose(0, 1)
            loss = criterion(log_probs, targets, output_lengths, target_lengths)
            loss.backward()
            clip_grad_norm_(model.parameters(), max_norm=grad_clip)
            optimizer.step()
        else:
            with torch.cuda.amp.autocast():
                logits, output_lengths = model(waveforms, waveform_lengths, spec_augment_config=spec_augment_cfg)
                log_probs = logits.log_softmax(dim=-1).transpose(0, 1)
                loss = criterion(log_probs, targets, output_lengths, target_lengths)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            clip_grad_norm_(model.parameters(), max_norm=grad_clip)
            scaler.step(optimizer)
            scaler.update()

        total_loss += float(loss.item())
        if step % print_every == 0:
            print(f"[train] step={step} loss={loss.item():.4f}", flush=True)

    effective_steps = min(len(loader), max_batches) if max_batches is not None else len(loader)
    return total_loss / max(1, effective_steps)


@torch.no_grad()
def evaluate(
    model: ConvGRUCTCModel,
    loader: DataLoader,
    criterion: nn.CTCLoss,
    device: torch.device,
    vocab: WordVocabulary,
    seen_speakers: set[str],
    max_batches: int | None = None,
) -> tuple[float, float, dict[str, float], dict[str, float]]:
    model.eval()
    total_loss = 0.0
    all_references: list[str] = []
    all_predictions: list[str] = []
    all_speakers: list[str] = []

    for step, batch in enumerate(loader, start=1):
        if max_batches is not None and step > max_batches:
            break
        waveforms = batch["waveforms"].to(device)
        waveform_lengths = batch["waveform_lengths"].to(device)
        targets = batch["targets"].to(device)
        target_lengths = batch["target_lengths"].to(device)

        logits, output_lengths = model(waveforms, waveform_lengths)
        log_probs = logits.log_softmax(dim=-1).transpose(0, 1)
        loss = criterion(log_probs, targets, output_lengths, target_lengths)
        total_loss += float(loss.item())

        predictions = decode_number_predictions(logits.cpu(), output_lengths.cpu(), vocab)
        all_predictions.extend(predictions)
        all_references.extend(batch["reference_digits"])
        all_speakers.extend(batch["spk_ids"])

    effective_steps = min(len(loader), max_batches) if max_batches is not None else len(loader)
    return (
        total_loss / max(1, effective_steps),
        dataset_cer(all_references, all_predictions),
        speaker_cer(all_references, all_predictions, all_speakers),
        domain_cer(all_references, all_predictions, all_speakers, seen_speakers=seen_speakers),
    )


def main() -> None:
    args = parse_args()
    with args.config.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    set_seed(int(config["seed"]))
    device = resolve_device(str(config["train"]["device"]))
    output_dir = ROOT / config["paths"]["output_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)

    vocab = WordVocabulary()
    model = ConvGRUCTCModel(vocab_size=vocab.size, **config["model"]).to(device)
    parameter_count = count_parameters(model)
    print(f"[setup] device={device} parameters={parameter_count}")
    if parameter_count >= 5_000_000:
        raise RuntimeError(f"Model is too large: {parameter_count}")

    waveform_aug_cfg = dict(config["train"].get("augment", {}).get("waveform", {}))
    train_dataset = NumbersDataset(
        csv_path=ROOT / config["paths"]["train_csv"],
        audio_root=ROOT / config["paths"]["audio_root"],
        sample_rate=int(config["model"]["sample_rate"]),
        with_labels=True,
        waveform_augment_cfg=waveform_aug_cfg,
    )
    dev_dataset = NumbersDataset(
        csv_path=ROOT / config["paths"]["dev_csv"],
        audio_root=ROOT / config["paths"]["audio_root"],
        sample_rate=int(config["model"]["sample_rate"]),
        with_labels=True,
    )

    collate = partial(collate_batch, vocab=vocab)
    sampler_cfg = dict(config["train"].get("sampler", {}))
    use_sampler = bool(sampler_cfg.get("speaker_balanced", False))
    if use_sampler:
        sampler = build_speaker_balanced_sampler(
            train_dataset.frame, alpha=float(sampler_cfg.get("alpha", 0.5))
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=int(config["train"]["batch_size"]),
            sampler=sampler,
            num_workers=int(config["train"]["num_workers"]),
            collate_fn=collate,
        )
    else:
        train_loader = DataLoader(
            train_dataset,
            batch_size=int(config["train"]["batch_size"]),
            shuffle=True,
            num_workers=int(config["train"]["num_workers"]),
            collate_fn=collate,
        )
    dev_loader = DataLoader(
        dev_dataset,
        batch_size=int(config["train"]["batch_size"]),
        shuffle=False,
        num_workers=int(config["train"]["num_workers"]),
        collate_fn=collate,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["train"]["learning_rate"]),
        weight_decay=float(config["train"]["weight_decay"]),
    )
    criterion = nn.CTCLoss(blank=vocab.blank_id, zero_infinity=True)
    use_amp = bool(config["train"].get("use_amp", True)) and device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp) if use_amp else None
    seen_speakers = {speaker for speaker in train_dataset.frame.get("spk_id", []).tolist() if speaker}

    best_cer = float("inf")
    history: list[dict[str, float]] = []
    start_epoch = 1

    if args.resume_checkpoint is not None:
        resume_checkpoint = torch.load(args.resume_checkpoint, map_location=device)
        model.load_state_dict(resume_checkpoint["model_state"])
        optimizer_state = resume_checkpoint.get("optimizer_state")
        if optimizer_state is not None and not args.fresh_optimizer:
            optimizer.load_state_dict(optimizer_state)
            # After loading, the optimizer lr is whatever was saved; re-apply config lr.
            for group in optimizer.param_groups:
                group["lr"] = float(config["train"]["learning_rate"])
        scaler_state = resume_checkpoint.get("scaler_state")
        if scaler is not None and scaler_state is not None and not args.fresh_optimizer:
            scaler.load_state_dict(scaler_state)
        history = list(resume_checkpoint.get("history", []))
        if not args.reset_best_cer:
            best_cer = float(resume_checkpoint.get("best_cer", best_cer))
        last_epoch = int(resume_checkpoint.get("epoch", 0))
        start_epoch = last_epoch + 1
        print(f"[resume] checkpoint={args.resume_checkpoint} start_epoch={start_epoch} best_dev_cer={best_cer:.4f}", flush=True)

    end_epoch = start_epoch + int(config["train"]["epochs"]) - 1
    sched_cfg = dict(config["train"].get("lr_schedule", {}))
    use_scheduler = bool(sched_cfg.get("enabled", False))
    scheduler_ctx: dict | None = None
    if use_scheduler:
        steps_per_epoch = max(1, len(train_loader))
        total_steps = steps_per_epoch * int(config["train"]["epochs"])
        warmup_steps = int(float(sched_cfg.get("warmup_epochs", 1.0)) * steps_per_epoch)
        scheduler_ctx = {
            "global_step": 0,
            "total_steps": total_steps,
            "peak_lr": float(config["train"]["learning_rate"]),
            "warmup_steps": warmup_steps,
            "min_lr_ratio": float(sched_cfg.get("min_lr_ratio", 0.05)),
        }
        print(
            f"[schedule] cosine with warmup: peak_lr={scheduler_ctx['peak_lr']} "
            f"warmup_steps={warmup_steps} total_steps={total_steps}",
            flush=True,
        )

    for epoch in range(start_epoch, end_epoch + 1):
        train_loss = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            scaler=scaler,
            print_every=int(config["train"]["print_every"]),
            grad_clip=float(config["train"]["grad_clip"]),
            augment_cfg=dict(config["train"].get("augment", {})),
            max_batches=args.max_train_batches,
            scheduler_ctx=scheduler_ctx,
        )
        dev_loss, dev_cer, per_speaker, domain_scores = evaluate(
            model=model,
            loader=dev_loader,
            criterion=criterion,
            device=device,
            vocab=vocab,
            seen_speakers=seen_speakers,
            max_batches=args.max_dev_batches,
        )
        speaker_summary = ", ".join(f"{speaker}={value:.4f}" for speaker, value in per_speaker.items())
        domain_summary = ", ".join(f"{key}={value:.4f}" for key, value in domain_scores.items())
        print(
            f"[epoch {epoch}] train_loss={train_loss:.4f} "
            f"dev_loss={dev_loss:.4f} dev_cer={dev_cer:.4f} "
            f"domains=({domain_summary}) speakers=({speaker_summary})",
            flush=True,
        )

        history_row: dict[str, float] = {"epoch": epoch, "train_loss": train_loss, "dev_loss": dev_loss, "dev_cer": dev_cer}
        history_row.update(domain_scores)
        history.append(history_row)

        improved = dev_cer < best_cer
        if improved:
            best_cer = dev_cer

        checkpoint = {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scaler_state": scaler.state_dict() if scaler is not None else None,
            "config": config,
            "vocab_tokens": vocab.tokens,
            "history": history,
            "best_cer": best_cer,
        }
        torch.save(checkpoint, output_dir / "last.pt")
        if improved:
            torch.save(checkpoint, output_dir / "best.pt")

    print(f"[done] best_dev_cer={best_cer:.4f}", flush=True)


if __name__ == "__main__":
    main()
