from __future__ import annotations

import numpy as np


class EnergyVAD:
    """Simple speech gate based on root-mean-square audio energy."""

    def __init__(self, threshold: float = 0.01) -> None:
        if threshold < 0:
            raise ValueError("threshold cannot be negative")
        self.threshold = threshold

    def is_speech(self, samples: np.ndarray) -> bool:
        if samples.size == 0:
            return False
        rms = float(np.sqrt(np.mean(np.square(samples.astype(np.float32, copy=False)))))
        return rms >= self.threshold

