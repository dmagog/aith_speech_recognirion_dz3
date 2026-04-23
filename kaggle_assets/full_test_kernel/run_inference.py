from __future__ import annotations

import itertools
import math
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
import torch
from scipy.signal import butter, resample_poly, sosfiltfilt
from torch import nn
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence, pad_sequence
from torch.utils.data import DataLoader, Dataset

UNITS_MASC = {
    0: "ноль",
    1: "один",
    2: "два",
    3: "три",
    4: "четыре",
    5: "пять",
    6: "шесть",
    7: "семь",
    8: "восемь",
    9: "девять",
}
UNITS_FEM = {
    0: "ноль",
    1: "одна",
    2: "две",
    3: "три",
    4: "четыре",
    5: "пять",
    6: "шесть",
    7: "семь",
    8: "восемь",
    9: "девять",
}
TEENS = {
    10: "десять",
    11: "одиннадцать",
    12: "двенадцать",
    13: "тринадцать",
    14: "четырнадцать",
    15: "пятнадцать",
    16: "шестнадцать",
    17: "семнадцать",
    18: "восемнадцать",
    19: "девятнадцать",
}
TENS = {
    20: "двадцать",
    30: "тридцать",
    40: "сорок",
    50: "пятьдесят",
    60: "шестьдесят",
    70: "семьдесят",
    80: "восемьдесят",
    90: "девяносто",
}
HUNDREDS = {
    100: "сто",
    200: "двести",
    300: "триста",
    400: "четыреста",
    500: "пятьсот",
    600: "шестьсот",
    700: "семьсот",
    800: "восемьсот",
    900: "девятьсот",
}
THOUSAND_FORMS = ("тысяча", "тысячи", "тысяч")
TOKEN_PATTERN = re.compile(r"[а-яё]+", flags=re.IGNORECASE)
UNITS_REVERSE = {value: key for key, value in itertools.chain(UNITS_MASC.items(), UNITS_FEM.items())}
TEENS_REVERSE = {value: key for key, value in TEENS.items()}
TENS_REVERSE = {value: key for key, value in TENS.items()}
HUNDREDS_REVERSE = {value: key for key, value in HUNDREDS.items()}
THOUSAND_TOKENS = set(THOUSAND_FORMS)


def tokenize_words(text: str) -> list[str]:
    return TOKEN_PATTERN.findall(text.lower())


def _parse_triplet(tokens: list[str]) -> int:
    if not tokens:
        return 0
    if tokens == ["ноль"]:
        return 0
    total = 0
    index = 0
    if index < len(tokens) and tokens[index] in HUNDREDS_REVERSE:
        total += HUNDREDS_REVERSE[tokens[index]]
        index += 1
    if index < len(tokens) and tokens[index] in TEENS_REVERSE:
        total += TEENS_REVERSE[tokens[index]]
        index += 1
        if index != len(tokens):
            raise ValueError(tokens)
        return total
    if index < len(tokens) and tokens[index] in TENS_REVERSE:
        total += TENS_REVERSE[tokens[index]]
        index += 1
    if index < len(tokens) and tokens[index] in UNITS_REVERSE:
        total += UNITS_REVERSE[tokens[index]]
        index += 1
    if index != len(tokens):
        raise ValueError(tokens)
    return total


def denormalize_transcription(text: str) -> int:
    tokens = tokenize_words(text)
    if not tokens:
        raise ValueError("empty")
    thousand_indices = [i for i, token in enumerate(tokens) if token in THOUSAND_TOKENS]
    if len(thousand_indices) > 1:
        raise ValueError(text)
    if not thousand_indices:
        value = _parse_triplet(tokens)
        if not 0 <= value <= 999999:
            raise ValueError(text)
        return value
    split_index = thousand_indices[0]
    thousands = _parse_triplet(tokens[:split_index]) if split_index > 0 else 1
    remainder = _parse_triplet(tokens[split_index + 1 :])
    value = thousands * 1000 + remainder
    if not 0 <= value <= 999999:
        raise ValueError(text)
    return value


def best_effort_number_from_text(text: str, default: int = 0, max_token_drops: int = 2) -> int:
    tokens = tokenize_words(text)
    if not tokens:
        return default
    try:
        return denormalize_transcription(" ".join(tokens))
    except ValueError:
        pass
    max_token_drops = min(max_token_drops, max(0, len(tokens) - 1))
    for drops in range(1, max_token_drops + 1):
        for kept_indices in itertools.combinations(range(len(tokens)), len(tokens) - drops):
            candidate = " ".join(tokens[index] for index in kept_indices)
            try:
                return denormalize_transcription(candidate)
            except ValueError:
                continue
    return default


def default_number_vocabulary() -> list[str]:
    tokens: list[str] = []
    for group in (UNITS_MASC, UNITS_FEM, TEENS, TENS, HUNDREDS):
        for token in group.values():
            if token not in tokens:
                tokens.append(token)
    for token in THOUSAND_FORMS:
        if token not in tokens:
            tokens.append(token)
    return tokens


@dataclass(frozen=True)
class WordVocabulary:
    blank_token: str = "<blank>"

    def __post_init__(self) -> None:
        tokens = [self.blank_token, *default_number_vocabulary()]
        object.__setattr__(self, "tokens", tokens)
        object.__setattr__(self, "stoi", {token: index for index, token in enumerate(tokens)})
        object.__setattr__(self, "itos", {index: token for index, token in enumerate(tokens)})

    @property
    def blank_id(self) -> int:
        return self.stoi[self.blank_token]

    @property
    def size(self) -> int:
        return len(self.tokens)

    def decode_ctc(self, token_ids: list[int]) -> str:
        collapsed: list[str] = []
        previous = None
        for token_id in token_ids:
            if token_id == self.blank_id:
                previous = None
                continue
            if token_id == previous:
                continue
            collapsed.append(self.itos[token_id])
            previous = token_id
        return " ".join(collapsed)


def _resample_waveform(waveform: np.ndarray, src_sr: int, target_sr: int) -> np.ndarray:
    if src_sr == target_sr:
        return waveform
    divisor = math.gcd(src_sr, target_sr)
    up = target_sr // divisor
    down = src_sr // divisor
    return resample_poly(waveform, up, down).astype(np.float32)


def load_audio(path: Path, target_sr: int = 16000) -> tuple[torch.Tensor, int]:
    waveform, sample_rate = sf.read(path, always_2d=False)
    if waveform.ndim == 2:
        waveform = waveform.mean(axis=1)
    waveform = waveform.astype(np.float32, copy=False)
    waveform = _resample_waveform(waveform, sample_rate, target_sr)
    waveform = np.clip(waveform, -1.0, 1.0)
    return torch.from_numpy(waveform), target_sr


def _hz_to_mel(freq: np.ndarray) -> np.ndarray:
    return 2595.0 * np.log10(1.0 + freq / 700.0)


def _mel_to_hz(mel: np.ndarray) -> np.ndarray:
    return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)


def create_mel_filterbank(sample_rate: int, n_fft: int, n_mels: int) -> torch.Tensor:
    mel_min = _hz_to_mel(np.array([0.0], dtype=np.float64))[0]
    mel_max = _hz_to_mel(np.array([sample_rate / 2], dtype=np.float64))[0]
    mel_points = np.linspace(mel_min, mel_max, n_mels + 2)
    hz_points = _mel_to_hz(mel_points)
    bins = np.floor((n_fft + 1) * hz_points / sample_rate).astype(np.int64)
    n_freqs = n_fft // 2 + 1
    fbanks = np.zeros((n_freqs, n_mels), dtype=np.float32)
    for mel_index in range(n_mels):
        left = bins[mel_index]
        center = max(bins[mel_index + 1], left + 1)
        right = min(max(bins[mel_index + 2], center + 1), n_freqs)
        for freq_bin in range(left, min(center, n_freqs)):
            fbanks[freq_bin, mel_index] = (freq_bin - left) / max(1, center - left)
        for freq_bin in range(center, right):
            fbanks[freq_bin, mel_index] = (right - freq_bin) / max(1, right - center)
    return torch.from_numpy(fbanks)


class LogMelSpectrogram(nn.Module):
    def __init__(self, sample_rate: int = 16000, n_fft: int = 400, win_length: int = 400, hop_length: int = 160, n_mels: int = 80) -> None:
        super().__init__()
        self.n_fft = n_fft
        self.win_length = win_length
        self.hop_length = hop_length
        self.register_buffer("window", torch.hann_window(win_length), persistent=False)
        self.register_buffer("mel_filterbank", create_mel_filterbank(sample_rate, n_fft, n_mels), persistent=False)

    def forward(self, waveforms: torch.Tensor, lengths: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        spec = torch.stft(
            waveforms,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window,
            center=True,
            return_complex=True,
        )
        power = spec.abs().pow(2.0).transpose(1, 2)
        mel = power @ self.mel_filterbank
        log_mel = torch.log(mel.clamp_min(1e-5))
        frame_lengths = torch.div(lengths, self.hop_length, rounding_mode="floor") + 1
        frame_lengths = torch.clamp(frame_lengths, max=log_mel.size(1))
        return log_mel, frame_lengths


def masked_mean_and_std(features: torch.Tensor, lengths: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    max_frames = features.size(1)
    mask = torch.arange(max_frames, device=features.device).unsqueeze(0) < lengths.unsqueeze(1)
    mask = mask.unsqueeze(-1)
    denom = mask.sum(dim=1, keepdim=True).clamp_min(1)
    mean = (features * mask).sum(dim=1, keepdim=True) / denom
    var = ((features - mean).pow(2) * mask).sum(dim=1, keepdim=True) / denom
    std = torch.sqrt(var + 1e-5)
    return mean, std


class ConvGRUCTCModel(nn.Module):
    def __init__(self, vocab_size: int, sample_rate: int = 16000, n_fft: int = 400, win_length: int = 400, hop_length: int = 160, n_mels: int = 80, conv_channels: int = 192, encoder_dim: int = 256, hidden_size: int = 256, num_gru_layers: int = 3, dropout: float = 0.2) -> None:
        super().__init__()
        self.feature_extractor = LogMelSpectrogram(sample_rate, n_fft, win_length, hop_length, n_mels)
        self.conv = nn.Sequential(
            nn.Conv1d(n_mels, conv_channels, kernel_size=5, stride=2, padding=2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(conv_channels, encoder_dim, kernel_size=5, stride=2, padding=2),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.gru = nn.GRU(
            input_size=encoder_dim,
            hidden_size=hidden_size,
            num_layers=num_gru_layers,
            dropout=dropout if num_gru_layers > 1 else 0.0,
            batch_first=True,
            bidirectional=True,
        )
        self.classifier = nn.Linear(hidden_size * 2, vocab_size)

    @staticmethod
    def _conv_out_length(lengths: torch.Tensor) -> torch.Tensor:
        lengths = torch.div(lengths + 1, 2, rounding_mode="floor")
        lengths = torch.div(lengths + 1, 2, rounding_mode="floor")
        return lengths.clamp_min(1)

    def forward(self, waveforms: torch.Tensor, lengths: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features, feature_lengths = self.feature_extractor(waveforms, lengths)
        mean, std = masked_mean_and_std(features, feature_lengths)
        features = (features - mean) / std
        x = self.conv(features.transpose(1, 2)).transpose(1, 2)
        output_lengths = self._conv_out_length(feature_lengths)
        packed = pack_padded_sequence(x, output_lengths.cpu(), batch_first=True, enforce_sorted=False)
        packed_output, _ = self.gru(packed)
        encoded, _ = pad_packed_sequence(packed_output, batch_first=True)
        logits = self.classifier(encoded)
        return logits, output_lengths


class NumbersDataset(Dataset):
    def __init__(self, csv_path: Path, audio_root: Path, sample_rate: int = 16000) -> None:
        self.csv_path = csv_path
        self.audio_root = audio_root
        self.sample_rate = sample_rate
        self.frame = pd.read_csv(csv_path)

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int) -> dict[str, object]:
        row = self.frame.iloc[index]
        filename = row["filename"]
        waveform, _ = load_audio(self.audio_root / filename, target_sr=self.sample_rate)
        return {"filename": filename, "waveform": waveform, "waveform_length": len(waveform)}


def collate_batch(batch: list[dict[str, object]]) -> dict[str, object]:
    waveforms = [item["waveform"] for item in batch]
    lengths = torch.tensor([int(item["waveform_length"]) for item in batch], dtype=torch.long)
    return {
        "waveforms": pad_sequence(waveforms, batch_first=True),
        "waveform_lengths": lengths,
        "filenames": [str(item["filename"]) for item in batch],
    }


def greedy_decode_words(logits: torch.Tensor, lengths: torch.Tensor, vocab: WordVocabulary) -> list[str]:
    token_ids = logits.argmax(dim=-1)
    return [vocab.decode_ctc(row[:length].tolist()) for row, length in zip(token_ids, lengths.tolist(), strict=True)]


def decode_number_predictions(logits: torch.Tensor, lengths: torch.Tensor, vocab: WordVocabulary) -> list[str]:
    words = greedy_decode_words(logits, lengths, vocab)
    return [str(best_effort_number_from_text(text)) for text in words]


# ---------------------------------------------------------------------------
# Grammar-constrained CTC prefix beam decoder
# ---------------------------------------------------------------------------


from dataclasses import dataclass as _dataclass
from functools import lru_cache as _lru_cache


@_dataclass(frozen=True)
class FsaState:
    section: str
    triplet: str
    scope: str
    thousands_seen: bool


INITIAL_FSA_STATES: tuple[FsaState, ...] = (
    FsaState("thousands", "S", "fem", False),
    FsaState("remainder", "S", "masc", False),
)


def _transitions_from(state: FsaState) -> dict[str, FsaState]:
    transitions: dict[str, FsaState] = {}
    scope = state.scope

    if state.triplet == "S":
        for hundred_word in HUNDREDS.values():
            transitions[hundred_word] = FsaState(state.section, "H", scope, state.thousands_seen)

    if state.triplet in ("S", "H"):
        for teen_word in TEENS.values():
            transitions[teen_word] = FsaState(state.section, "F", scope, state.thousands_seen)
        for tens_word in TENS.values():
            transitions[tens_word] = FsaState(state.section, "T", scope, state.thousands_seen)
        units_lexicon = UNITS_FEM if scope == "fem" else UNITS_MASC
        for value, unit_word in units_lexicon.items():
            if value == 0:
                continue
            transitions[unit_word] = FsaState(state.section, "F", scope, state.thousands_seen)

    if state.triplet == "T":
        units_lexicon = UNITS_FEM if scope == "fem" else UNITS_MASC
        for value, unit_word in units_lexicon.items():
            if value == 0:
                continue
            transitions[unit_word] = FsaState(state.section, "F", scope, state.thousands_seen)

    if state.section == "thousands" and not state.thousands_seen:
        for thousand_word in THOUSAND_FORMS:
            transitions[thousand_word] = FsaState("remainder", "S", "masc", True)

    return transitions


@_lru_cache(maxsize=None)
def _cached_transitions(state: FsaState) -> dict[str, FsaState]:
    return _transitions_from(state)


def _is_acceptable(state: FsaState) -> bool:
    if state.triplet in ("H", "T", "F"):
        return True
    if state.section == "remainder" and state.thousands_seen and state.triplet == "S":
        return True
    return False


def _logaddexp(a: float, b: float) -> float:
    if a == -1e30:
        return b
    if b == -1e30:
        return a
    if a > b:
        diff = b - a
        return a + math.log1p(math.exp(diff)) if diff > -30 else a
    diff = a - b
    return b + math.log1p(math.exp(diff)) if diff > -30 else b


def grammar_beam_decode(
    log_probs: torch.Tensor,
    lengths: torch.Tensor,
    vocab: WordVocabulary,
    beam_size: int = 16,
    prune_threshold: float = 8.0,
) -> list[str]:
    blank_id = vocab.blank_id
    vocab_tokens = vocab.tokens
    NEG_INF = -1e30

    batch_size = log_probs.size(0)
    log_probs_cpu = log_probs.detach().cpu()
    results: list[str] = []

    for batch_index in range(batch_size):
        valid_length = int(lengths[batch_index].item())
        if valid_length <= 0:
            results.append("")
            continue

        seq_log_probs = log_probs_cpu[batch_index, :valid_length]

        beams: dict[tuple[tuple[int, ...], FsaState], tuple[float, float]] = {}
        for state in INITIAL_FSA_STATES:
            beams[((), state)] = (0.0, NEG_INF)

        for t in range(valid_length):
            frame = seq_log_probs[t]
            frame_list = frame.tolist()
            frame_max = max(frame_list)
            next_beams: dict[tuple[tuple[int, ...], FsaState], tuple[float, float]] = {}

            for (prefix, fsa_state), (lpb, lpnb) in beams.items():
                total_prev = _logaddexp(lpb, lpnb)
                last_token = prefix[-1] if prefix else -1

                blank_score = frame_list[blank_id]
                new_lpb = total_prev + blank_score
                key = (prefix, fsa_state)
                if key in next_beams:
                    ob, onb = next_beams[key]
                    next_beams[key] = (_logaddexp(ob, new_lpb), onb)
                else:
                    next_beams[key] = (new_lpb, NEG_INF)

                if last_token != -1:
                    repeat_score = frame_list[last_token]
                    new_lpnb = lpnb + repeat_score
                    if key in next_beams:
                        ob, onb = next_beams[key]
                        next_beams[key] = (ob, _logaddexp(onb, new_lpnb))
                    else:
                        next_beams[key] = (NEG_INF, new_lpnb)

                transitions = _cached_transitions(fsa_state)
                for word, next_state in transitions.items():
                    token_id = vocab.stoi[word]
                    token_score = frame_list[token_id]
                    if token_score < frame_max - prune_threshold:
                        continue
                    base = lpb if last_token == token_id else total_prev
                    new_lpnb = base + token_score
                    new_prefix = prefix + (token_id,)
                    new_key = (new_prefix, next_state)
                    if new_key in next_beams:
                        ob, onb = next_beams[new_key]
                        next_beams[new_key] = (ob, _logaddexp(onb, new_lpnb))
                    else:
                        next_beams[new_key] = (NEG_INF, new_lpnb)

            if not next_beams:
                beams = {}
                break

            if len(next_beams) > beam_size:
                scored = sorted(
                    next_beams.items(),
                    key=lambda item: _logaddexp(item[1][0], item[1][1]),
                    reverse=True,
                )
                beams = dict(scored[:beam_size])
            else:
                beams = next_beams

        if not beams:
            results.append("")
            continue

        accepted = [
            (prefix, _logaddexp(lpb, lpnb))
            for (prefix, state), (lpb, lpnb) in beams.items()
            if _is_acceptable(state)
        ]
        if accepted:
            accepted.sort(key=lambda item: item[1], reverse=True)
            best_prefix = accepted[0][0]
        else:
            best_key = max(beams.items(), key=lambda item: _logaddexp(item[1][0], item[1][1]))[0]
            best_prefix = best_key[0]

        words = [vocab_tokens[token_id] for token_id in best_prefix]
        results.append(" ".join(words))

    return results


def decode_number_predictions_beam(
    logits: torch.Tensor,
    lengths: torch.Tensor,
    vocab: WordVocabulary,
    beam_size: int = 16,
) -> list[str]:
    log_probs = logits.log_softmax(dim=-1)
    sentences = grammar_beam_decode(log_probs, lengths, vocab, beam_size=beam_size)
    return [str(best_effort_number_from_text(sentence)) for sentence in sentences]


def decode_number_predictions_from_logprobs(
    log_probs: torch.Tensor,
    lengths: torch.Tensor,
    vocab: WordVocabulary,
    beam_size: int = 16,
) -> list[str]:
    sentences = grammar_beam_decode(log_probs, lengths, vocab, beam_size=beam_size)
    return [str(best_effort_number_from_text(sentence)) for sentence in sentences]


# ---------------------------------------------------------------------------
# Test-time augmentation
# ---------------------------------------------------------------------------


def _lowpass_sos(cutoff_hz: float, sample_rate: int, order: int = 6):
    nyquist = 0.5 * sample_rate
    normalized = max(0.05, min(0.95, cutoff_hz / nyquist))
    return butter(order, normalized, btype="low", output="sos")


def bandwidth_limit_numpy(waveform: np.ndarray, sample_rate: int, cutoff_hz: float) -> np.ndarray:
    if cutoff_hz <= 0 or cutoff_hz >= sample_rate / 2:
        return waveform
    sos = _lowpass_sos(cutoff_hz, sample_rate)
    return sosfiltfilt(sos, waveform).astype(np.float32, copy=False)


def pitch_shift_numpy(waveform: np.ndarray, sample_rate: int, semitones: float) -> np.ndarray:
    if abs(semitones) < 1e-3:
        return waveform
    rate = float(2.0 ** (semitones / 12.0))
    stretch_up = max(1, int(round(sample_rate * rate / 10.0)))
    stretch_down = max(1, int(round(sample_rate / 10.0)))
    stretched = resample_poly(waveform, stretch_up, stretch_down).astype(np.float32, copy=False)
    original_length = len(waveform)
    if len(stretched) == original_length:
        return stretched
    if len(stretched) > original_length:
        start = (len(stretched) - original_length) // 2
        return stretched[start : start + original_length]
    padded = np.zeros(original_length, dtype=np.float32)
    start = (original_length - len(stretched)) // 2
    padded[start : start + len(stretched)] = stretched
    return padded


TTA_VARIANTS = [
    {"pitch_semitones": 0.5},
    {"pitch_semitones": -0.5},
    {"bandwidth_cutoff_hz": 5500.0},
]


@torch.no_grad()
def tta_log_probs(
    model: nn.Module,
    waveforms: torch.Tensor,
    lengths: torch.Tensor,
    sample_rate: int,
    variants: list[dict],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    waveforms_np = [waveforms[i, : int(lengths[i].item())].cpu().numpy() for i in range(waveforms.size(0))]
    all_tensors: list[torch.Tensor] = [waveforms]
    for variant in variants:
        transformed = []
        for wav in waveforms_np:
            out = wav
            semitones = float(variant.get("pitch_semitones", 0.0))
            if abs(semitones) > 1e-3:
                out = pitch_shift_numpy(out, sample_rate, semitones)
            cutoff = float(variant.get("bandwidth_cutoff_hz", 0.0))
            if cutoff > 0:
                out = bandwidth_limit_numpy(out, sample_rate, cutoff)
            transformed.append(np.clip(out, -1.0, 1.0).astype(np.float32, copy=False))
        max_len = max(len(w) for w in transformed)
        padded = np.zeros((len(transformed), max_len), dtype=np.float32)
        for i, w in enumerate(transformed):
            padded[i, : len(w)] = w
        all_tensors.append(torch.from_numpy(padded).to(device))

    log_probs_list: list[torch.Tensor] = []
    out_len_ref: torch.Tensor | None = None
    for t in all_tensors:
        logits, ol = model(t, lengths)
        log_probs_list.append(logits.log_softmax(dim=-1))
        out_len_ref = ol if out_len_ref is None else torch.minimum(out_len_ref, ol)
    assert out_len_ref is not None
    min_T = min(lp.size(1) for lp in log_probs_list)
    fused = torch.stack([lp[:, :min_T, :] for lp in log_probs_list], dim=0).mean(dim=0)
    return fused, out_len_ref.clamp_max(min_T)


def locate_test_split() -> tuple[Path, Path]:
    input_root = Path("/kaggle/input")
    candidates = sorted(input_root.rglob("test.csv"))
    for csv_path in candidates:
        root = csv_path.parent
        frame = pd.read_csv(csv_path)
        if "filename" not in frame.columns or len(frame) == 0:
            continue
        probe = root / frame.iloc[0]["filename"]
        if probe.exists():
            return csv_path, root
    raise FileNotFoundError("Could not locate competition test.csv with matching audio files")


def locate_weights() -> list[Path]:
    """Return ordered list of checkpoint paths used for ensemble.

    Picks any files named ``best*.pt`` inside /kaggle/input and sorts them
    so that the primary ``best.pt`` comes first.
    """
    input_root = Path("/kaggle/input")
    candidates = list(input_root.rglob("best*.pt"))
    if not candidates:
        raise FileNotFoundError("Could not locate best*.pt in attached Kaggle datasets")

    def sort_key(p: Path) -> tuple[int, str]:
        return (0 if p.name == "best.pt" else 1, p.name)

    return sorted(candidates, key=sort_key)


def main() -> None:
    test_csv, audio_root = locate_test_split()
    weights_paths = locate_weights()
    print(f"test_csv={test_csv}")
    print(f"audio_root={audio_root}")
    print(f"ensemble_weights={weights_paths}")

    vocab = WordVocabulary()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    models: list[ConvGRUCTCModel] = []
    for path in weights_paths:
        model = ConvGRUCTCModel(vocab_size=vocab.size).to(device)
        checkpoint = torch.load(path, map_location=device)
        model.load_state_dict(checkpoint["model_state"])
        model.eval()
        models.append(model)

    dataset = NumbersDataset(test_csv, audio_root, sample_rate=16000)
    loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=2, collate_fn=collate_batch)

    rows = []
    with torch.no_grad():
        for step, batch in enumerate(loader, start=1):
            waveforms = batch["waveforms"].to(device)
            waveform_lengths = batch["waveform_lengths"].to(device)

            per_model_logp: list[torch.Tensor] = []
            per_model_len: list[torch.Tensor] = []
            for model in models:
                log_probs, out_len = tta_log_probs(
                    model, waveforms, waveform_lengths, 16000, TTA_VARIANTS, device
                )
                per_model_logp.append(log_probs)
                per_model_len.append(out_len)
            min_T = min(lp.size(1) for lp in per_model_logp)
            stacked = torch.stack([lp[:, :min_T, :] for lp in per_model_logp], dim=0)
            fused = stacked.mean(dim=0)
            out_len = torch.stack(per_model_len, dim=0).min(dim=0).values.clamp_max(min_T)

            predictions = decode_number_predictions_from_logprobs(
                fused.cpu(), out_len.cpu(), vocab, beam_size=16
            )
            for filename, transcription in zip(batch["filenames"], predictions, strict=True):
                rows.append({"filename": filename, "transcription": transcription})
            if step % 25 == 0:
                print(f"processed_batches={step} rows={len(rows)}")

    output_path = Path("/kaggle/working/submission.csv")
    submission = pd.DataFrame(rows, columns=["filename", "transcription"])
    submission.to_csv(output_path, index=False)
    print(f"wrote_submission={output_path} rows={len(submission)}")
    print(submission.head().to_string(index=False))


if __name__ == "__main__":
    main()
