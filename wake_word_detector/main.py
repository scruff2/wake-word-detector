from __future__ import annotations

import argparse
import logging
from pathlib import Path

from wake_word_detector.actions import ConsoleChimeWakeHandler
from wake_word_detector.audio_capture import MicrophoneAudioSource
from wake_word_detector.config import load_config
from wake_word_detector.runtime import WakeWordRuntime
from wake_word_detector.transcriber import FasterWhisperTranscriber
from wake_word_detector.vad import EnergyVAD
from wake_word_detector.wake_detector import WakePhraseDetector


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    overrides = {
        "model": args.model,
        "compute_type": args.compute_type,
        "device": args.whisper_device,
        "wake_phrases": tuple(args.wake_phrase) if args.wake_phrase else None,
        "chunk_seconds": args.chunk_seconds,
        "overlap_seconds": args.overlap_seconds,
        "vad_energy_threshold": args.vad_energy_threshold,
        "debug": args.debug,
        "chime": args.chime,
    }
    config = load_config(args.config, overrides=overrides)
    configure_logging(config.debug, config.log_file)

    audio_source = MicrophoneAudioSource(
        sample_rate=config.sample_rate,
        chunk_seconds=config.chunk_seconds,
        overlap_seconds=config.overlap_seconds,
        device=args.audio_device,
    )
    vad = EnergyVAD(config.vad_energy_threshold)
    transcriber = FasterWhisperTranscriber(
        model_name=config.model,
        device=config.device,
        compute_type=config.compute_type,
        language=config.language,
    )
    detector = WakePhraseDetector(config.wake_phrases, config.fuzzy_threshold)
    handler = ConsoleChimeWakeHandler(chime=config.chime)

    runtime = WakeWordRuntime(
        audio_source=audio_source,
        vad=vad,
        transcriber=transcriber,
        detector=detector,
        wake_handler=handler,
        cooldown_seconds=config.cooldown_seconds,
    )

    try:
        runtime.run_forever()
    except KeyboardInterrupt:
        logging.getLogger(__name__).info("Stopping wake-word detector")
    finally:
        audio_source.stop()
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Offline faster-whisper wake-word detector")
    parser.add_argument(
        "--config",
        default="config.yaml",
        type=Path,
        help="Path to YAML or JSON config file",
    )
    parser.add_argument("--wake-phrase", action="append", help="Wake phrase override; repeat for alternatives")
    parser.add_argument("--model", help="faster-whisper model name, e.g. tiny.en or base.en")
    parser.add_argument("--compute-type", help="faster-whisper compute type, e.g. int8 or float16")
    parser.add_argument("--whisper-device", choices=["auto", "cpu", "cuda"], help="Whisper device")
    parser.add_argument("--audio-device", help="sounddevice input device name or index")
    parser.add_argument("--chunk-seconds", type=float, help="Audio window duration in seconds")
    parser.add_argument("--overlap-seconds", type=float, help="Audio overlap between windows in seconds")
    parser.add_argument("--vad-energy-threshold", type=float, help="RMS energy threshold for speech detection")
    parser.add_argument("--debug", action="store_true", help="Enable verbose debug logging")
    parser.add_argument(
        "--no-chime",
        action="store_false",
        dest="chime",
        default=None,
        help="Disable audible chime on detection",
    )
    return parser


def configure_logging(debug: bool, log_file: str | None) -> None:
    handlers: list[logging.Handler] = [logging.StreamHandler()]
    if log_file:
        handlers.append(logging.FileHandler(log_file, encoding="utf-8"))

    logging.basicConfig(
        level=logging.DEBUG if debug else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=handlers,
    )


if __name__ == "__main__":
    raise SystemExit(main())
