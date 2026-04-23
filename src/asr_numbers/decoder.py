from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache

import torch

from .text import (
    HUNDREDS,
    TEENS,
    TENS,
    THOUSAND_FORMS,
    UNITS_FEM,
    UNITS_MASC,
    best_effort_number_from_text,
)
from .vocab import WordVocabulary


def greedy_decode_words(logits: torch.Tensor, lengths: torch.Tensor, vocab: WordVocabulary) -> list[str]:
    token_ids = logits.argmax(dim=-1)
    decoded: list[str] = []
    for row, length in zip(token_ids, lengths.tolist(), strict=True):
        decoded.append(vocab.decode_ctc(row[:length].tolist()))
    return decoded


def decode_number_predictions(logits: torch.Tensor, lengths: torch.Tensor, vocab: WordVocabulary) -> list[str]:
    decoded_words = greedy_decode_words(logits, lengths, vocab)
    return [str(best_effort_number_from_text(words)) for words in decoded_words]


# ---------------------------------------------------------------------------
# Grammar-constrained CTC prefix beam decoder for Russian numerals 0..999999.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FsaState:
    section: str   # "thousands" (before тысяч*) or "remainder" (after тысяч* or no thousands)
    triplet: str   # "S", "H", "T", "F"
    scope: str     # "masc" or "fem" — which unit-word set is valid here
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


@lru_cache(maxsize=None)
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
    word_insertion_penalty: float = 0.0,
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
                    # CTC prefix beam rule:
                    #   different token: base = lpb + lpnb
                    #   same token as last: base = lpb (must cross a blank)
                    base = lpb if last_token == token_id else total_prev
                    new_lpnb = base + token_score - word_insertion_penalty
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
    word_insertion_penalty: float = 0.0,
) -> list[str]:
    log_probs = logits.log_softmax(dim=-1)
    sentences = grammar_beam_decode(
        log_probs,
        lengths,
        vocab,
        beam_size=beam_size,
        word_insertion_penalty=word_insertion_penalty,
    )
    return [str(best_effort_number_from_text(sentence)) for sentence in sentences]
