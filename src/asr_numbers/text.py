from __future__ import annotations

import itertools
import re
from typing import Iterable

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


def _plural_form(value: int, forms: tuple[str, str, str]) -> str:
    value = abs(value) % 100
    if 11 <= value <= 14:
        return forms[2]
    last = value % 10
    if last == 1:
        return forms[0]
    if 2 <= last <= 4:
        return forms[1]
    return forms[2]


def _triplet_to_words(value: int, feminine: bool) -> list[str]:
    if not 0 <= value <= 999:
        raise ValueError(f"Triplet out of range: {value}")

    words: list[str] = []
    hundreds = value // 100 * 100
    if hundreds:
        words.append(HUNDREDS[hundreds])

    remainder = value % 100
    if 10 <= remainder <= 19:
        words.append(TEENS[remainder])
        return words

    tens = remainder // 10 * 10
    units = remainder % 10
    if tens:
        words.append(TENS[tens])
    if units:
        lexicon = UNITS_FEM if feminine else UNITS_MASC
        words.append(lexicon[units])
    return words


def normalize_transcription(value: int | str) -> str:
    number = int(value)
    if not 0 <= number <= 999999:
        raise ValueError(f"Number out of supported range: {number}")

    if number == 0:
        return UNITS_MASC[0]

    if number < 1000:
        return " ".join(_triplet_to_words(number, feminine=False))

    thousands, remainder = divmod(number, 1000)
    words = _triplet_to_words(thousands, feminine=True)
    words.append(_plural_form(thousands, THOUSAND_FORMS))
    words.extend(_triplet_to_words(remainder, feminine=False))
    return " ".join(words)


def _parse_triplet(tokens: Iterable[str]) -> int:
    parts = list(tokens)
    if not parts:
        return 0
    if parts == ["ноль"]:
        return 0

    total = 0
    index = 0

    if index < len(parts) and parts[index] in HUNDREDS_REVERSE:
        total += HUNDREDS_REVERSE[parts[index]]
        index += 1

    if index < len(parts) and parts[index] in TEENS_REVERSE:
        total += TEENS_REVERSE[parts[index]]
        index += 1
        if index != len(parts):
            raise ValueError(f"Unexpected tail after teen token: {parts}")
        return total

    if index < len(parts) and parts[index] in TENS_REVERSE:
        total += TENS_REVERSE[parts[index]]
        index += 1

    if index < len(parts) and parts[index] in UNITS_REVERSE:
        total += UNITS_REVERSE[parts[index]]
        index += 1

    if index != len(parts):
        raise ValueError(f"Invalid triplet tokens: {parts}")
    if total > 999:
        raise ValueError(f"Triplet overflow: {parts}")
    return total


def denormalize_transcription(text: str) -> int:
    tokens = tokenize_words(text)
    if not tokens:
        raise ValueError("Empty transcription")

    thousand_indices = [index for index, token in enumerate(tokens) if token in THOUSAND_TOKENS]
    if len(thousand_indices) > 1:
        raise ValueError(f"Multiple thousand markers: {text}")

    if not thousand_indices:
        value = _parse_triplet(tokens)
        if not 0 <= value <= 999999:
            raise ValueError(f"Parsed value outside supported range: {value}")
        return value

    split_index = thousand_indices[0]
    left = tokens[:split_index]
    right = tokens[split_index + 1 :]

    thousands = _parse_triplet(left) if left else 1
    remainder = _parse_triplet(right)
    value = thousands * 1000 + remainder
    if not 0 <= value <= 999999:
        raise ValueError(f"Parsed value outside supported range: {value}")
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
