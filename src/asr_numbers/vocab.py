from __future__ import annotations

from dataclasses import dataclass

from .text import default_number_vocabulary, tokenize_words


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

    def encode(self, text: str) -> list[int]:
        return [self.stoi[token] for token in tokenize_words(text)]

    def decode(self, token_ids: list[int]) -> str:
        tokens = [self.itos[token_id] for token_id in token_ids if token_id != self.blank_id]
        return " ".join(tokens)

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
