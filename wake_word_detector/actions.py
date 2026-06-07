from __future__ import annotations

import logging
import platform
from typing import Protocol

from wake_word_detector.wake_detector import WakeDetectionEvent

LOGGER = logging.getLogger(__name__)


class WakeHandler(Protocol):
    def __call__(self, event: WakeDetectionEvent) -> None:
        """Handle a detected wake phrase."""


class ConsoleChimeWakeHandler:
    def __init__(self, chime: bool = True) -> None:
        self.chime = chime

    def __call__(self, event: WakeDetectionEvent) -> None:
        LOGGER.info(
            'Wake phrase detected: "%s" in transcript "%s"',
            event.phrase,
            event.transcript,
        )
        if self.chime:
            play_chime()


def play_chime() -> None:
    if platform.system() == "Windows":
        try:
            import winsound

            winsound.MessageBeep(winsound.MB_ICONASTERISK)
            return
        except RuntimeError:
            pass

    print("\a", end="", flush=True)

