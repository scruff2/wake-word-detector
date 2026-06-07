from pathlib import Path

import pytest

from wake_word_detector.config import AppConfig, load_config


def test_default_config_uses_computer() -> None:
    config = AppConfig()

    assert config.wake_phrases == ("computer",)


def test_load_config_rejects_unknown_option(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text("unknown: true\n", encoding="utf-8")

    with pytest.raises(ValueError, match="Unknown config"):
        load_config(config_path)


def test_load_config_accepts_wake_phrase_list(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text('wake_phrases:\n  - "computer"\n  - "ok assistant"\n', encoding="utf-8")

    config = load_config(config_path)

    assert config.wake_phrases == ("computer", "ok assistant")

