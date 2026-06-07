from __future__ import annotations

import logging
import time
from typing import Protocol

from wake_word_detector.actions import WakeHandler
from wake_word_detector.audio_capture import AudioChunk
from wake_word_detector.transcriber import TranscriptionResult
from wake_word_detector.wake_detector import WakePhraseDetector

LOGGER = logging.getLogger(__name__)


class AudioSource(Protocol):
    def chunks(self):
        """Yield AudioChunk instances."""


class VoiceActivityDetector(Protocol):
    def is_speech(self, samples) -> bool:
        """Return true when audio contains speech-like energy."""


class Transcriber(Protocol):
    def transcribe(self, chunk: AudioChunk) -> TranscriptionResult:
        """Transcribe an audio chunk."""


class WakeWordRuntime:
    def __init__(
        self,
        audio_source: AudioSource,
        vad: VoiceActivityDetector,
        transcriber: Transcriber,
        detector: WakePhraseDetector,
        wake_handler: WakeHandler,
        cooldown_seconds: float,
    ) -> None:
        self.audio_source = audio_source
        self.vad = vad
        self.transcriber = transcriber
        self.detector = detector
        self.wake_handler = wake_handler
        self.cooldown_seconds = cooldown_seconds
        self._last_detection_time = 0.0

    def run_forever(self) -> None:
        LOGGER.info("Listening for wake phrase(s): %s", ", ".join(self.detector.phrases))
        for chunk in self.audio_source.chunks():
            self.process_chunk(chunk)

    def process_chunk(self, chunk: AudioChunk) -> None:
        if not self.vad.is_speech(chunk.samples):
            LOGGER.debug("Skipping silent chunk")
            return

        started = time.perf_counter()
        result = self.transcriber.transcribe(chunk)
        elapsed = time.perf_counter() - started

        if not result.text:
            LOGGER.debug("No transcript returned in %.2fs", elapsed)
            return

        LOGGER.info('Transcript: "%s" (confidence=%s, %.2fs)', result.text, _format_confidence(result.confidence), elapsed)
        event = self.detector.detect(result.text, confidence=result.confidence)
        if event is None:
            return

        now = time.monotonic()
        if now - self._last_detection_time < self.cooldown_seconds:
            LOGGER.debug("Wake phrase ignored during cooldown")
            return

        self._last_detection_time = now
        self.wake_handler(event)


def _format_confidence(confidence: float | None) -> str:
    if confidence is None:
        return "unknown"
    return f"{confidence:.3f}"

