"""Pruebas unitarias para el módulo de cálculos dsp.calculations."""
from __future__ import annotations

import unittest
import numpy as np

from dsp.calculations import (
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


def test_generate_multitone_signal():
    freqs = [5.0, 15.0, 30.0]
    amps = [1.0, 0.6, 0.4]
    duration = 1.0
    fs = 20000
    t, x = generate_multitone_signal(freqs, amps, duration=duration, sample_rate=fs)
    assert len(t) == 20000
    assert len(x) == 20000
    assert np.isclose(t[0], 0.0)
    assert t[-1] < 1.0  # endpoint=False
    assert np.max(np.abs(x)) <= sum(amps) + 1e-6


def test_generate_twotone_and_rectangular_pulse():
    t, x = generate_twotone_signal(5.0, 15.0, duration=1.0, n_points=4000)
    assert len(t) == 4000
    assert len(x) == 4000

    t_pulse = np.linspace(-2, 2, 4001)
    pulse = generate_rectangular_pulse(t_pulse, pulse_width=1.0)
    # En t=0 debe ser 1.0
    zero_idx = np.argmin(np.abs(t_pulse))
    assert pulse[zero_idx] == 1.0
    # En |t| > 0.5 debe ser 0.0
    assert pulse[0] == 0.0
    assert pulse[-1] == 0.0


def test_calculate_nyquist_and_sampling_grid():
    f_max = 30.0
    fs_nyquist = calculate_nyquist_rate(f_max)
    assert fs_nyquist == 60.0

    t_muestras, ts = calculate_sampling_grid(duration=1.0, fs=100.0)
    assert np.isclose(ts, 0.01)
    assert len(t_muestras) == 100
    assert np.isclose(t_muestras[0], 0.0)
    assert np.isclose(t_muestras[-1], 0.99)


def test_calculate_apparent_frequency_aliasing():
    # 15 Hz muestreado a 20 Hz produce un alias a 5 Hz
    f_signal = 15.0
    fs_alias = 20.0
    f_alias_gen = calculate_apparent_frequency(f_signal, fs_alias, use_general_formula=True)
    f_alias_basic = calculate_apparent_frequency(f_signal, fs_alias, use_general_formula=False)
    assert np.isclose(f_alias_gen, 5.0)
    assert np.isclose(f_alias_basic, 5.0)

    # Armónico de segundo orden: 35 Hz con fs=20 Hz -> 5 Hz
    assert np.isclose(calculate_apparent_frequency(35.0, 20.0, use_general_formula=True), 5.0)


def test_transformada_fourier_direct_vs_theoretical():
    # Pulso rectangular de ancho T=1.0 s
    T = 1.0
    t = np.linspace(-5.0, 5.0, 4000)
    x = np.where(np.abs(t) <= T / 2.0, 1.0, 0.0)

    # Evaluación en frecuencias entre -10 y 10 Hz
    f = np.linspace(-10.0, 10.0, 401)
    X_num = transformada_fourier(x, t, f)
    X_teo = theoretical_rectangular_pulse_fourier(f, pulse_width=T)

    error = fourier_transform_error(X_num, X_teo)
    # La integración trapezoidal de un pulso rectangular continuo converge muy bien
    assert error < 0.05


def test_compute_fft_spectrum():
    fs = 1000.0
    duration = 1.0
    t = np.arange(int(fs * duration)) / fs
    # Tono de 50 Hz de amplitud 1.0
    f_tone = 50.0
    x = np.sin(2.0 * np.pi * f_tone * t)

    freqs, mag = compute_fft_spectrum(x, fs)
    peak_idx = int(np.argmax(mag))
    detected_freq = freqs[peak_idx]

    assert abs(detected_freq - f_tone) < 1.0
    assert np.isclose(mag[peak_idx], 1.0, atol=0.05)


def test_quantize_adc_dac():
    bits = 8
    vmin = -5.0
    vmax = 5.0
    v_sampled = np.array([-5.0, -2.5, 0.0, 2.5, 5.0])

    codes, v_recon, err, q_step = quantize_adc_dac(v_sampled, bits=bits, v_min=vmin, v_max=vmax)
    assert q_step == 10.0 / 256
    assert np.all(codes >= 0)
    assert np.all(codes < 256)
    assert np.allclose(err, v_sampled - v_recon)
    assert np.all(np.abs(err) <= q_step)


def test_interpolate_signal():
    t_m = np.array([0.0, 1.0, 2.0])
    muestras = np.array([0.0, 10.0, 20.0])
    t_rec = np.array([0.0, 0.5, 1.0, 1.5, 2.0])

    senal_rec = interpolate_signal(muestras, t_m, t_rec)
    assert np.allclose(senal_rec, [0.0, 5.0, 10.0, 15.0, 20.0])


def test_compute_minmax_downsample():
    # Señal larga con un pico distintivo
    x = np.zeros(100000, dtype=np.float64)
    x[54321] = 99.0
    x[12345] = -88.0

    indices = compute_minmax_downsample(x, 0, len(x), max_points=1000)
    assert len(indices) <= 2000
    assert 54321 in indices
    assert 12345 in indices


def test_compute_relative_db_spectrum():
    f = np.array([10.0, 20.0, 30.0, 40.0])
    mag = np.array([0.1, 1.0, 0.01, 0.001])

    f_out, db = compute_relative_db_spectrum(f, mag, max_frequency=30.0)
    assert len(f_out) == 3
    assert np.isclose(db[1], 0.0)  # Pico relativo en 20 Hz
    assert db[0] < 0.0
    assert db[2] < db[0]


class TestCalculations(unittest.TestCase):
    def test_multitone(self):
        test_generate_multitone_signal()

    def test_twotone_and_pulse(self):
        test_generate_twotone_and_rectangular_pulse()

    def test_nyquist_grid(self):
        test_calculate_nyquist_and_sampling_grid()

    def test_aliasing(self):
        test_calculate_apparent_frequency_aliasing()

    def test_fourier_direct(self):
        test_transformada_fourier_direct_vs_theoretical()

    def test_fft(self):
        test_compute_fft_spectrum()

    def test_quantize(self):
        test_quantize_adc_dac()

    def test_interpolation(self):
        test_interpolate_signal()

    def test_downsample(self):
        test_compute_minmax_downsample()

    def test_db_spectrum(self):
        test_compute_relative_db_spectrum()


if __name__ == "__main__":
    unittest.main()

