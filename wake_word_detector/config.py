from __future__ import annotations

import json
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class AppConfig:
    model: str = "tiny.en"
    compute_type: str = "int8"
    device: str = "cpu"
    language: str = "en"
    wake_phrases: tuple[str, ...] = ("computer",)
    chunk_seconds: float = 2.0
    overlap_seconds: float = 0.8
    sample_rate: int = 16000
    vad_energy_threshold: float = 0.01
    fuzzy_threshold: int = 0
    cooldown_seconds: float = 2.0
    debug: bool = False
    log_file: str | None = None
    chime: bool = True

    def validate(self) -> None:
        if not self.wake_phrases:
            raise ValueError("At least one wake phrase is required")
        if any(not phrase.strip() for phrase in self.wake_phrases):
            raise ValueError("Wake phrases cannot be blank")
        if self.sample_rate != 16000:
            raise ValueError("sample_rate must be 16000 for faster-whisper microphone input")
        if self.chunk_seconds <= 0:
            raise ValueError("chunk_seconds must be greater than 0")
        if self.overlap_seconds < 0:
            raise ValueError("overlap_seconds cannot be negative")
        if self.overlap_seconds >= self.chunk_seconds:
            raise ValueError("overlap_seconds must be less than chunk_seconds")
        if self.vad_energy_threshold < 0:
            raise ValueError("vad_energy_threshold cannot be negative")
        if self.fuzzy_threshold < 0:
            raise ValueError("fuzzy_threshold cannot be negative")
        if self.cooldown_seconds < 0:
            raise ValueError("cooldown_seconds cannot be negative")


def load_config(path: str | Path = "config.yaml", overrides: dict[str, Any] | None = None) -> AppConfig:
    config_path = Path(path)
    data: dict[str, Any] = {}
    if config_path.exists():
        data = _read_config_file(config_path)
    data.update({key: value for key, value in (overrides or {}).items() if value is not None})
    config = _config_from_mapping(data)
    config.validate()
    return config


def _read_config_file(path: Path) -> dict[str, Any]:
    if path.suffix.lower() == ".json":
        return json.loads(path.read_text(encoding="utf-8"))

    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("PyYAML is required to read YAML config files") from exc

    loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    if loaded is None:
        return {}
    if not isinstance(loaded, dict):
        raise ValueError(f"Config file must contain a mapping: {path}")
    return loaded


def _config_from_mapping(data: dict[str, Any]) -> AppConfig:
    allowed = {field.name for field in fields(AppConfig)}
    unknown = sorted(set(data) - allowed)
    if unknown:
        raise ValueError(f"Unknown config option(s): {', '.join(unknown)}")

    if "wake_phrases" in data and isinstance(data["wake_phrases"], list):
        data = dict(data)
        data["wake_phrases"] = tuple(str(phrase) for phrase in data["wake_phrases"])
    return AppConfig(**data)
