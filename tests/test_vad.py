import numpy as np

from wake_word_detector.vad import EnergyVAD


def test_energy_vad_rejects_silence() -> None:
    vad = EnergyVAD(threshold=0.01)

    assert not vad.is_speech(np.zeros(16000, dtype=np.float32))


def test_energy_vad_accepts_loud_samples() -> None:
    vad = EnergyVAD(threshold=0.01)

    assert vad.is_speech(np.ones(16000, dtype=np.float32) * 0.02)

