from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from asr_numbers.text import best_effort_number_from_text, denormalize_transcription, normalize_transcription


class TextNormalizationTests(unittest.TestCase):
    def test_normalize_examples(self) -> None:
        self.assertEqual(normalize_transcription(14), "четырнадцать")
        self.assertEqual(normalize_transcription(999), "девятьсот девяносто девять")
        self.assertEqual(normalize_transcription(1000), "одна тысяча")
        self.assertEqual(normalize_transcription(1005), "одна тысяча пять")
        self.assertEqual(normalize_transcription(1011), "одна тысяча одиннадцать")
        self.assertEqual(normalize_transcription(12432), "двенадцать тысяч четыреста тридцать два")

    def test_roundtrip_examples(self) -> None:
        for value in (14, 999, 1000, 1005, 1011, 21543, 139473, 999999):
            text = normalize_transcription(value)
            self.assertEqual(denormalize_transcription(text), value)

    def test_best_effort_recovery(self) -> None:
        self.assertEqual(best_effort_number_from_text("одна тысяча тысяча пять"), 1005)
        self.assertEqual(best_effort_number_from_text("сто тридцать девять тысяч четыреста семьдесят три"), 139473)


if __name__ == "__main__":
    unittest.main()
