"""Rutinas DSP, muestreo, cuantización ADC, reconstrucción DAC y FFT."""
from __future__ import annotations

from fractions import Fraction
from typing import Dict, Any, Tuple

import numpy as np
from scipy.signal import resample_poly

MAX_ANALYSIS_SECONDS = 10.0


def resample_signal(signal: np.ndarray, fs_in: int, fs_out: int) -> np.ndarray:
    """Remuestrea con filtro antialiasing polifásico de SciPy."""
    x = np.asarray(signal, dtype=np.float64).reshape(-1)
    if fs_in <= 0 or fs_out <= 0:
        raise ValueError("Las frecuencias de muestreo deben ser mayores que cero.")
    if x.size == 0:
        return x.copy()
    if fs_in == fs_out:
        return x.copy()
    ratio = Fraction(int(fs_out), int(fs_in)).limit_denominator(10000)
    return np.asarray(resample_poly(x, ratio.numerator, ratio.denominator), dtype=np.float64)


def magnitude_spectrum(signal: np.ndarray, fs: int) -> Tuple[np.ndarray, np.ndarray]:
    """FFT de magnitud con ventana Hann; la FFT se usa para analizar, no para muestrear."""
    if fs <= 0:
        raise ValueError("La frecuencia de muestreo debe ser mayor que cero.")
    x = np.asarray(signal, dtype=np.float64).reshape(-1)
    if x.size < 2:
        return np.array([0.0]), np.array([0.0])
    centered = x - np.mean(x)
    window = np.hanning(centered.size)
    window_sum = np.sum(window)
    if window_sum <= 0:
        window_sum = float(centered.size)
    spec = np.abs(np.fft.rfft(centered * window)) / (window_sum / 2.0)
    if spec.size:
        spec[0] *= 0.5
        if centered.size % 2 == 0 and spec.size > 1:
            spec[-1] *= 0.5
    freqs = np.fft.rfftfreq(centered.size, d=1.0 / float(fs))
    return freqs, spec


def dominant_frequency(signal: np.ndarray, fs: int) -> float:
    """Frecuencia del mayor pico distinto de DC (estimación espectral)."""
    freqs, magnitudes = magnitude_spectrum(signal, fs)
    if magnitudes.size <= 1 or np.max(magnitudes[1:]) <= np.finfo(float).eps:
        return 0.0
    idx = int(np.argmax(magnitudes[1:]) + 1)
    return float(freqs[idx])


def quantize_adc(signal: np.ndarray, bits: int, vmin: float, vmax: float) -> Dict[str, Any]:
    """Cuantización uniforme mid-rise. La señal de audio normalizada [-1,1] se asigna al rango V.
    
    Nota de diseño / comparación:
    - Esta función implementa el cuantizador uniforme mid-rise especificado en la práctica de AudioLab DSP,
      donde los niveles reconstruidos se sitúan en los centros de los intervalos ((codes + 0.5) * delta_v).
    - Para la formulación alternativa directa basada en redondeo a extremos (Fragmento 6), véase
      `dsp.calculations.quantize_adc_dac`.
    """
    x = np.asarray(signal, dtype=np.float64).reshape(-1)
    if not 1 <= int(bits) <= 24:
        raise ValueError("La resolución debe estar entre 1 y 24 bits.")
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
        raise ValueError("El rango ADC debe cumplir Vmax > Vmin y usar valores finitos.")

    levels = 2 ** int(bits)
    full_range = float(vmax - vmin)
    delta_v = full_range / levels
    clipped = np.clip(x, -1.0, 1.0)
    volts = float(vmin) + ((clipped + 1.0) / 2.0) * full_range
    codes = np.floor((volts - float(vmin)) / delta_v).astype(np.int64)
    codes = np.clip(codes, 0, levels - 1)
    quantized_volts = float(vmin) + (codes + 0.5) * delta_v
    quantized = 2.0 * ((quantized_volts - float(vmin)) / full_range) - 1.0
    error_volts = quantized_volts - volts
    error_signal = quantized - clipped
    rmse_norm = float(np.sqrt(np.mean(np.square(error_signal)))) if x.size else 0.0
    rmse_volts = float(np.sqrt(np.mean(np.square(error_volts)))) if x.size else 0.0
    max_error_volts = float(np.max(np.abs(error_volts))) if x.size else 0.0
    signal_rms = float(np.sqrt(np.mean(np.square(clipped)))) if x.size else 0.0
    error_rms = float(np.sqrt(np.mean(np.square(error_signal)))) if x.size else 0.0
    if error_rms <= 0:
        snr_db = float("inf")
    elif signal_rms <= 0:
        snr_db = float("-inf")
    else:
        snr_db = float(20.0 * np.log10(signal_rms / error_rms))

    return {
        "bits": int(bits),
        "levels": int(levels),
        "vmin": float(vmin),
        "vmax": float(vmax),
        "delta_v": float(delta_v),
        "scale_v_per_unit": float(full_range / 2.0),
        "input_volts": volts,
        "codes": codes,
        "quantized_volts": quantized_volts,
        "quantized": quantized,
        "error_volts": error_volts,
        "error_signal": error_signal,
        "rmse_normalized": rmse_norm,
        "rmse_volts": rmse_volts,
        "max_error_volts": max_error_volts,
        "snr_quantization_db": snr_db,
    }


def zero_order_hold(samples: np.ndarray, fs: int, times: np.ndarray) -> np.ndarray:
    """Reconstrucción DAC básica por retención de orden cero (ZOH)."""
    x = np.asarray(samples, dtype=np.float64).reshape(-1)
    t = np.asarray(times, dtype=np.float64)
    if x.size == 0:
        return np.zeros_like(t, dtype=np.float64)
    idx = np.floor(np.maximum(t, 0.0) * int(fs)).astype(np.int64)
    idx = np.clip(idx, 0, x.size - 1)
    return x[idx]


def analyze_audio(signal: np.ndarray, fs_original: int, fs_processing: int,
                  bits: int, vmin: float, vmax: float,
                  max_seconds: float = MAX_ANALYSIS_SECONDS) -> Dict[str, Any]:
    """Procesa un fragmento coherente del audio y devuelve datos/medidas para la GUI."""
    original_all = np.asarray(signal, dtype=np.float64).reshape(-1)
    if original_all.size == 0:
        raise ValueError("No hay muestras para analizar.")
    if fs_original <= 0 or fs_processing <= 0:
        raise ValueError("La frecuencia de muestreo debe ser positiva.")
    if fs_processing > 192000:
        raise ValueError("Para esta aplicación, la frecuencia de procesamiento máxima es 192000 Hz.")
    if max_seconds <= 0:
        raise ValueError("La duración máxima de análisis debe ser positiva.")

    n_limit = max(1, int(round(float(max_seconds) * fs_original)))
    original = original_all[:n_limit]
    duration = len(original) / float(fs_original)
    t_original = np.arange(original.size, dtype=np.float64) / float(fs_original)
    sampled = resample_signal(original, fs_original, fs_processing)
    t_sampled = np.arange(sampled.size, dtype=np.float64) / float(fs_processing)

    adc = quantize_adc(sampled, bits, vmin, vmax)
    reconstructed = zero_order_hold(adc["quantized"], fs_processing, t_original)
    compare_len = min(original.size, reconstructed.size)
    err = original[:compare_len] - reconstructed[:compare_len]
    reconstruction_rmse = float(np.sqrt(np.mean(np.square(err)))) if compare_len else 0.0
    reconstruction_mae = float(np.mean(np.abs(err))) if compare_len else 0.0

    f, mag = magnitude_spectrum(sampled, fs_processing)
    fq, magq = magnitude_spectrum(adc["quantized"], fs_processing)
    dominant = dominant_frequency(sampled, fs_processing)

    high_frequency_pct = 0.0
    if fs_processing < fs_original and original.size > 1:
        fo, mo = magnitude_spectrum(original, fs_original)
        energy = np.square(mo)
        total = float(np.sum(energy))
        if total > 0:
            high_frequency_pct = 100.0 * float(np.sum(energy[fo > fs_processing / 2.0])) / total

    return {
        "fs_original": int(fs_original),
        "fs_processing": int(fs_processing),
        "nyquist": float(fs_processing / 2.0),
        "analysis_duration": float(duration),
        "original": original,
        "t_original": t_original,
        "sampled": sampled,
        "t_sampled": t_sampled,
        "adc": adc,
        "reconstructed": reconstructed,
        "reconstruction_rmse": reconstruction_rmse,
        "reconstruction_mae": reconstruction_mae,
        "frequencies": f,
        "magnitudes": mag,
        "quantized_frequencies": fq,
        "quantized_magnitudes": magq,
        "dominant_frequency": dominant,
        "energy_above_nyquist_pct": high_frequency_pct,
    }
