from __future__ import annotations

import re
import time
from dataclasses import dataclass


@dataclass(frozen=True)
class WakeDetectionEvent:
    phrase: str
    transcript: str
    confidence: float | None
    detected_at: float
    fuzzy_distance: int | None = None


class WakePhraseDetector:
    def __init__(self, phrases: tuple[str, ...] | list[str], fuzzy_threshold: int = 0) -> None:
        cleaned = tuple(_normalize_space(phrase).lower() for phrase in phrases if phrase.strip())
        if not cleaned:
            raise ValueError("At least one wake phrase is required")
        if fuzzy_threshold < 0:
            raise ValueError("fuzzy_threshold cannot be negative")

        self.phrases = cleaned
        self.fuzzy_threshold = fuzzy_threshold
        self._patterns = {
            phrase: re.compile(_phrase_pattern(phrase), flags=re.IGNORECASE) for phrase in cleaned
        }

    def detect(self, transcript: str, confidence: float | None = None) -> WakeDetectionEvent | None:
        normalized_transcript = _normalize_space(transcript)
        if not normalized_transcript:
            return None

        for phrase, pattern in self._patterns.items():
            if pattern.search(normalized_transcript):
                return WakeDetectionEvent(
                    phrase=phrase,
                    transcript=transcript,
                    confidence=confidence,
                    detected_at=time.time(),
                )

        if self.fuzzy_threshold == 0:
            return None

        normalized_lower = normalized_transcript.lower()
        for phrase in self.phrases:
            distance = _phrase_edit_distance(phrase, normalized_lower)
            if distance is not None and distance <= self.fuzzy_threshold:
                return WakeDetectionEvent(
                    phrase=phrase,
                    transcript=transcript,
                    confidence=confidence,
                    detected_at=time.time(),
                    fuzzy_distance=distance,
                )

        return None


def _phrase_pattern(phrase: str) -> str:
    tokens = [re.escape(token) for token in phrase.split()]
    return r"(?<!\w)" + r"\s+".join(tokens) + r"(?!\w)"


def _normalize_space(value: str) -> str:
    return " ".join(value.strip().split())


def _phrase_edit_distance(phrase: str, transcript: str) -> int | None:
    phrase_tokens = phrase.split()
    transcript_tokens = re.findall(r"\w+", transcript.lower())
    if len(transcript_tokens) < len(phrase_tokens):
        return _levenshtein(" ".join(phrase_tokens), " ".join(transcript_tokens))

    best: int | None = None
    width = len(phrase_tokens)
    for start in range(0, len(transcript_tokens) - width + 1):
        candidate = " ".join(transcript_tokens[start : start + width])
        distance = _levenshtein(phrase, candidate)
        if best is None or distance < best:
            best = distance
    return best


def _levenshtein(left: str, right: str) -> int:
    if left == right:
        return 0
    if not left:
        return len(right)
    if not right:
        return len(left)

    previous = list(range(len(right) + 1))
    for left_index, left_char in enumerate(left, start=1):
        current = [left_index]
        for right_index, right_char in enumerate(right, start=1):
            insert_cost = current[right_index - 1] + 1
            delete_cost = previous[right_index] + 1
            replace_cost = previous[right_index - 1] + (left_char != right_char)
            current.append(min(insert_cost, delete_cost, replace_cost))
        previous = current
    return previous[-1]

