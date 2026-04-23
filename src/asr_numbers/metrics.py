from __future__ import annotations

from collections import defaultdict
from math import sqrt


def levenshtein_distance(source: str, target: str) -> int:
    if source == target:
        return 0
    if not source:
        return len(target)
    if not target:
        return len(source)

    previous = list(range(len(target) + 1))
    for i, source_char in enumerate(source, start=1):
        current = [i]
        for j, target_char in enumerate(target, start=1):
            insertion = current[j - 1] + 1
            deletion = previous[j] + 1
            substitution = previous[j - 1] + (source_char != target_char)
            current.append(min(insertion, deletion, substitution))
        previous = current
    return previous[-1]


def char_error_rate(reference: str, prediction: str) -> float:
    if not reference:
        return 0.0 if not prediction else 1.0
    return levenshtein_distance(reference, prediction) / len(reference)


def dataset_cer(references: list[str], predictions: list[str]) -> float:
    total_distance = sum(levenshtein_distance(reference, prediction) for reference, prediction in zip(references, predictions, strict=True))
    total_length = sum(len(reference) for reference in references)
    if total_length == 0:
        return 0.0
    return total_distance / total_length


def speaker_cer(references: list[str], predictions: list[str], speakers: list[str]) -> dict[str, float]:
    grouped_refs: dict[str, list[str]] = defaultdict(list)
    grouped_preds: dict[str, list[str]] = defaultdict(list)
    for reference, prediction, speaker in zip(references, predictions, speakers, strict=True):
        grouped_refs[speaker].append(reference)
        grouped_preds[speaker].append(prediction)
    return {speaker: dataset_cer(grouped_refs[speaker], grouped_preds[speaker]) for speaker in sorted(grouped_refs)}


def harmonic_mean(values: list[float]) -> float:
    positive_values = [value for value in values if value > 0]
    if not positive_values:
        return 0.0
    if len(positive_values) != len(values):
        return 0.0
    return len(values) / sum(1.0 / value for value in values)


def domain_cer(
    references: list[str],
    predictions: list[str],
    speakers: list[str],
    seen_speakers: set[str],
) -> dict[str, float]:
    in_domain_refs: list[str] = []
    in_domain_preds: list[str] = []
    out_domain_refs: list[str] = []
    out_domain_preds: list[str] = []

    for reference, prediction, speaker in zip(references, predictions, speakers, strict=True):
        if speaker in seen_speakers:
            in_domain_refs.append(reference)
            in_domain_preds.append(prediction)
        else:
            out_domain_refs.append(reference)
            out_domain_preds.append(prediction)

    result: dict[str, float] = {}
    if in_domain_refs:
        result["indomain_cer"] = dataset_cer(in_domain_refs, in_domain_preds)
    if out_domain_refs:
        result["ood_cer"] = dataset_cer(out_domain_refs, out_domain_preds)
    if "indomain_cer" in result and "ood_cer" in result:
        result["harmonic_cer"] = harmonic_mean([result["indomain_cer"], result["ood_cer"]])
        result["geometric_cer"] = sqrt(result["indomain_cer"] * result["ood_cer"])
    return result
