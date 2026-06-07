from __future__ import annotations

import logging
import queue
import threading
from dataclasses import dataclass
from typing import Iterator

import numpy as np

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class AudioChunk:
    samples: np.ndarray
    sample_rate: int


class MicrophoneAudioSource:
    """Continuously captures microphone audio and yields overlapping windows."""

    def __init__(
        self,
        sample_rate: int,
        chunk_seconds: float,
        overlap_seconds: float,
        device: int | str | None = None,
        block_seconds: float = 0.1,
    ) -> None:
        if chunk_seconds <= 0:
            raise ValueError("chunk_seconds must be greater than 0")
        if overlap_seconds < 0:
            raise ValueError("overlap_seconds cannot be negative")
        if overlap_seconds >= chunk_seconds:
            raise ValueError("overlap_seconds must be less than chunk_seconds")

        self.sample_rate = sample_rate
        self.chunk_samples = int(sample_rate * chunk_seconds)
        self.hop_samples = int(sample_rate * (chunk_seconds - overlap_seconds))
        self.block_samples = max(1, int(sample_rate * block_seconds))
        self.device = device
        self._audio_queue: queue.Queue[np.ndarray] = queue.Queue(maxsize=100)
        self._stop_event = threading.Event()

    def stop(self) -> None:
        self._stop_event.set()

    def chunks(self) -> Iterator[AudioChunk]:
        try:
            import sounddevice as sd
        except ImportError as exc:
            raise RuntimeError(
                "sounddevice is required for microphone capture. Install requirements.txt first."
            ) from exc

        def callback(indata, frames, time_info, status) -> None:
            if status:
                LOGGER.warning("Audio capture status: %s", status)
            samples = indata[:, 0].copy()
            try:
                self._audio_queue.put_nowait(samples)
            except queue.Full:
                LOGGER.warning("Audio queue is full; dropping microphone block")

        buffer = np.empty(0, dtype=np.float32)
        with sd.InputStream(
            channels=1,
            samplerate=self.sample_rate,
            dtype="float32",
            blocksize=self.block_samples,
            device=self.device,
            callback=callback,
        ):
            LOGGER.info("Microphone stream started at %s Hz", self.sample_rate)
            while not self._stop_event.is_set():
                try:
                    block = self._audio_queue.get(timeout=0.5)
                except queue.Empty:
                    continue

                buffer = np.concatenate((buffer, block.astype(np.float32, copy=False)))
                while len(buffer) >= self.chunk_samples:
                    yield AudioChunk(
                        samples=buffer[: self.chunk_samples].copy(),
                        sample_rate=self.sample_rate,
                    )
                    buffer = buffer[self.hop_samples :]

