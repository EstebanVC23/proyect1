"""Modelos de datos del analizador de audio."""
from dataclasses import dataclass, field
from typing import Optional
import numpy as np


@dataclass
class AudioTrack:
    """Audio asignado a una de las cinco categorías del proyecto."""
    slot: int
    genre: str
    title: str = ""
    artist: str = ""
    year: str = ""
    source: str = ""
    file_path: Optional[str] = None
    sample_rate: Optional[int] = None
    channels: Optional[int] = None
    bit_depth: Optional[int] = None
    duration: Optional[float] = None
    samples: Optional[np.ndarray] = field(default=None, repr=False)
    mono: Optional[np.ndarray] = field(default=None, repr=False)

    @property
    def loaded(self) -> bool:
        return self.mono is not None and self.sample_rate is not None
