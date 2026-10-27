"""Paquete de análisis de audio DSP/ADC/DAC para Mira / Señales DSP."""

from .models import AudioTrack
from .audio_io import load_audio, load_wav, filename_title, to_float_audio
from .calculations import (
    generate_multitone_signal,
    generate_twotone_signal,
    generate_rectangular_pulse,
    calculate_nyquist_rate,
    calculate_sampling_grid,
    calculate_apparent_frequency,
    transformada_fourier,
    theoretical_rectangular_pulse_fourier,
    fourier_transform_error,
    compute_fft_spectrum,
    quantize_adc_dac,
    interpolate_signal,
    compute_minmax_downsample,
    compute_relative_db_spectrum,
)
from .processing import (
    MAX_ANALYSIS_SECONDS,
    resample_signal,
    magnitude_spectrum,
    dominant_frequency,
    quantize_adc,
    zero_order_hold,
    analyze_audio,
)

__all__ = [
    "AudioTrack",
    "load_audio",
    "load_wav",
    "filename_title",
    "to_float_audio",
    "generate_multitone_signal",
    "generate_twotone_signal",
    "generate_rectangular_pulse",
    "calculate_nyquist_rate",
    "calculate_sampling_grid",
    "calculate_apparent_frequency",
    "transformada_fourier",
    "theoretical_rectangular_pulse_fourier",
    "fourier_transform_error",
    "compute_fft_spectrum",
    "quantize_adc_dac",
    "interpolate_signal",
    "compute_minmax_downsample",
    "compute_relative_db_spectrum",
    "MAX_ANALYSIS_SECONDS",
    "resample_signal",
    "magnitude_spectrum",
    "dominant_frequency",
    "quantize_adc",
    "zero_order_hold",
    "analyze_audio",
]
