from __future__ import annotations

from dataclasses import dataclass

from wake_word_detector.audio_capture import AudioChunk


@dataclass(frozen=True)
class TranscriptionResult:
    text: str
    confidence: float | None = None


class FasterWhisperTranscriber:
    def __init__(
        self,
        model_name: str,
        device: str = "auto",
        compute_type: str = "int8",
        language: str = "en",
        beam_size: int = 1,
    ) -> None:
        try:
            from faster_whisper import WhisperModel
        except ImportError as exc:
            raise RuntimeError(
                "faster-whisper is required for transcription. Install requirements.txt first."
            ) from exc

        self.language = language
        self.beam_size = beam_size
        self.model = WhisperModel(model_name, device=device, compute_type=compute_type)

    def transcribe(self, chunk: AudioChunk) -> TranscriptionResult:
        segments, _info = self.model.transcribe(
            chunk.samples,
            language=self.language,
            beam_size=self.beam_size,
            vad_filter=False,
            condition_on_previous_text=False,
        )
        texts: list[str] = []
        log_probs: list[float] = []

        for segment in segments:
            text = segment.text.strip()
            if text:
                texts.append(text)
            avg_logprob = getattr(segment, "avg_logprob", None)
            if avg_logprob is not None:
                log_probs.append(float(avg_logprob))

        confidence = None
        if log_probs:
            confidence = sum(log_probs) / len(log_probs)

        return TranscriptionResult(text=" ".join(texts).strip(), confidence=confidence)

