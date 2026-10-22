"""Carga y normalización de archivos de audio WAV y MP3."""
from __future__ import annotations

import wave
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import soundfile as sf


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _infer_bit_depth_wav(path: str) -> Optional[int]:
    """Lee la profundidad de bits de un WAV real usando el módulo estándar wave."""
    try:
        with wave.open(str(path), "rb") as wf:
            width = wf.getsampwidth()
            if width in (1, 2, 3, 4):
                return width * 8
    except (wave.Error, EOFError, OSError):
        pass
    return None


def to_float_audio(data: np.ndarray) -> np.ndarray:
    """Convierte muestras PCM o flotantes a float64 normalizado en [-1, 1]."""
    arr = np.asarray(data)
    if np.issubdtype(arr.dtype, np.unsignedinteger):
        info = np.iinfo(arr.dtype)
        midpoint = (info.max + 1) / 2.0
        result = (arr.astype(np.float64) - midpoint) / midpoint
    elif np.issubdtype(arr.dtype, np.signedinteger):
        info = np.iinfo(arr.dtype)
        result = arr.astype(np.float64) / float(max(abs(info.min), info.max + 1))
    elif np.issubdtype(arr.dtype, np.floating):
        result = arr.astype(np.float64)
        result = np.nan_to_num(result, nan=0.0, posinf=1.0, neginf=-1.0)
    else:
        raise ValueError(f"Tipo de muestra no compatible: {arr.dtype}")
    return result


# ---------------------------------------------------------------------------
# Cargador principal
# ---------------------------------------------------------------------------

def load_audio(path: str) -> Tuple[int, np.ndarray, np.ndarray, int, Optional[int]]:
    """Carga un archivo WAV o MP3 y devuelve (fs, datos_raw, mono_float64, canales, bits).

    - fs              : frecuencia de muestreo en Hz
    - datos_raw       : array con los datos en su dtype original (float32 para MP3)
    - mono_float64    : señal mono normalizada en [-1, 1] como float64
    - canales         : número de canales originales
    - bits            : profundidad en bits (None si no aplica)
    """
    ext = Path(path).suffix.lower()

    if ext in (".mp3",):
        return _load_mp3(path)
    else:
        return _load_soundfile(path)


def _load_soundfile(path: str) -> Tuple[int, np.ndarray, np.ndarray, int, Optional[int]]:
    """Carga WAV u otros formatos compatibles con soundfile."""
    data, fs = sf.read(path, always_2d=False, dtype="float64")
    if fs <= 0:
        raise ValueError("Frecuencia de muestreo inválida en el archivo.")
    if data.size == 0:
        raise ValueError("El archivo no contiene muestras de audio.")

    ext = Path(path).suffix.lower()
    depth = _infer_bit_depth_wav(path) if ext == ".wav" else None

    # Datos originales como float32 para guardar memoria
    raw = data.astype(np.float32)

    if data.ndim == 1:
        channels = 1
        mono = data.astype(np.float64)
    else:
        channels = int(data.shape[1])
        mono = np.mean(data, axis=1).astype(np.float64)

    mono = np.nan_to_num(mono, nan=0.0, posinf=1.0, neginf=-1.0)
    if not np.all(np.isfinite(mono)):
        raise ValueError("El archivo contiene valores de audio no finitos.")

    # Normalizar si hay clipping (valores fuera de [-1,1])
    peak = np.max(np.abs(mono))
    if peak > 1.0:
        mono = mono / peak

    return int(fs), raw, mono, channels, depth


def _load_mp3(path: str) -> Tuple[int, np.ndarray, np.ndarray, int, Optional[int]]:
    """Carga MP3 usando soundfile (libsndfile con MPEG support)."""
    try:
        return _load_soundfile(path)
    except Exception as exc:
        raise ValueError(
            f"No se pudo leer el MP3 '{Path(path).name}'.\n"
            "Asegúrate de que libsndfile tenga soporte MPEG (soundfile >= 0.12 + libsndfile >= 1.1).\n"
            f"Detalle: {exc}"
        ) from exc


def load_wav(path: str) -> Tuple[int, np.ndarray, np.ndarray, int, Optional[int]]:
    """Alias de load_audio para compatibilidad con código existente."""
    return load_audio(path)


def filename_title(path: str) -> str:
    """Extrae un título legible a partir del nombre del archivo."""
    return Path(path).stem.replace("_", " ").replace("-", " ").strip()
