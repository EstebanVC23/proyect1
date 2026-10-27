"""Módulo de cálculos matemáticos para señales, muestreo, espectros y gráficas.

Incorpora fragmentos de cálculo numérico reutilizables para:
1. Generación de señales continuas simuladas (sinusoides compuestas y pulsos).
2. Cálculo de muestreo, tasa mínima de Nyquist-Shannon y cuadrículas temporales.
3. Estimación de frecuencia aparente y plegamiento por submuestreo (aliasing).
4. Transformada de Fourier continua mediante integración directa numérica y comparación teórica.
5. Transformada rápida de Fourier (FFT), frecuencias positivas y magnitud unilateral normalizada.
6. Cuantización ADC uniforme, códigos digitales, reconstrucción DAC y error.
7. Reconstrucción aproximada mediante interpolación lineal.
8. Reducción y preparación de datos para alimentación de gráficas (decimación min-max y espectro en dB).
"""
from __future__ import annotations

from typing import Optional, Sequence, Tuple

import numpy as np


# ============================================================================
# 1. Generación de señales continuas simuladas
# ============================================================================

def generate_multitone_signal(
    frequencies: Sequence[float],
    amplitudes: Sequence[float],
    duration: float = 1.0,
    sample_rate: int = 20000,
    endpoint: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """Genera una señal compuesta por suma de sinusoides en alta resolución temporal.

    Fragmento 1:
        t_c = np.linspace(0, T, 20000, endpoint=False)
        x_c = sum(A_i * np.sin(2 * np.pi * f_i * t_c))

    Args:
        frequencies: Lista de frecuencias en Hz (ej. [5.0, 15.0, 30.0]).
        amplitudes: Lista de amplitudes asociadas (ej. [1.0, 0.6, 0.4]).
        duration: Duración temporal T en segundos.
        sample_rate: Tasa de muestreo de simulación continua (por defecto 20 000 Hz).
        endpoint: Si se incluye o no el punto final del intervalo temporal.

    Returns:
        (t_c, x_c): Vector temporal continuo y señal compuesta resultante.
    """
    if len(frequencies) != len(amplitudes):
        raise ValueError("Las listas de frecuencias y amplitudes deben tener la misma longitud.")
    if duration <= 0:
        raise ValueError("La duración debe ser positiva.")
    if sample_rate <= 0:
        raise ValueError("La tasa de muestreo debe ser mayor que cero.")

    n_points = max(1, int(round(duration * sample_rate)))
    t_c = np.linspace(0.0, float(duration), n_points, endpoint=endpoint, dtype=np.float64)
    x_c = np.zeros_like(t_c)
    for f, a in zip(frequencies, amplitudes):
        x_c += float(a) * np.sin(2.0 * np.pi * float(f) * t_c)
    return t_c, x_c


def generate_twotone_signal(
    f1: float = 5.0,
    f2: float = 15.0,
    duration: float = 1.0,
    n_points: int = 4000,
) -> Tuple[np.ndarray, np.ndarray]:
    """Genera una señal suma de dos componentes sinusoidales elementales.

    Fragmento 1 (Ejemplo 2):
        t = np.linspace(0, 1, 4000)
        x = np.sin(2 * np.pi * f1 * t) + np.sin(2 * np.pi * f2 * t)
    """
    if duration <= 0:
        raise ValueError("La duración debe ser positiva.")
    if n_points <= 0:
        raise ValueError("El número de puntos debe ser mayor que cero.")
    t = np.linspace(0.0, float(duration), int(n_points), dtype=np.float64)
    x = np.sin(2.0 * np.pi * float(f1) * t) + np.sin(2.0 * np.pi * float(f2) * t)
    return t, x


def generate_rectangular_pulse(
    t: np.ndarray,
    pulse_width: float = 1.0,
    amplitude: float = 1.0,
) -> np.ndarray:
    """Genera un pulso rectangular centrado en t = 0.

    Fragmento 1 (Ejemplo 3):
        x = np.where(np.abs(t) <= T / 2, 1.0, 0.0)

    Args:
        t: Vector temporal donde evaluar la señal.
        pulse_width: Ancho total T del pulso rectangular.
        amplitude: Amplitud del pulso.
    """
    t_arr = np.asarray(t, dtype=np.float64)
    half_width = float(pulse_width) / 2.0
    return np.where(np.abs(t_arr) <= half_width, float(amplitude), 0.0)


# ============================================================================
# 2. Muestreo según Nyquist-Shannon
# ============================================================================

def calculate_nyquist_rate(f_max: float) -> float:
    """Calcula la frecuencia mínima de muestreo según el criterio de Nyquist-Shannon.

    Fragmento 2:
        fs_nyquist = 2 * f_max
    """
    if f_max < 0:
        raise ValueError("La frecuencia máxima no puede ser negativa.")
    return 2.0 * float(f_max)


def calculate_sampling_grid(duration: float, fs: float) -> Tuple[np.ndarray, float]:
    """Calcula el vector de instantes de muestreo y el período de muestreo.

    Fragmento 2:
        Ts = 1 / fs
        N = int(fs * T)
        n = np.arange(N)
        t_muestras = n / fs

    Returns:
        (t_muestras, Ts): Vector de tiempos discretos y período de muestreo.
    """
    fs_val = float(fs)
    if fs_val <= 0:
        raise ValueError("La frecuencia de muestreo debe ser positiva.")
    if duration < 0:
        raise ValueError("La duración no puede ser negativa.")
    ts = 1.0 / fs_val
    n_samples = max(0, int(round(duration * fs_val)))
    t_muestras = np.arange(n_samples, dtype=np.float64) / fs_val
    return t_muestras, ts


# ============================================================================
# 3. Frecuencia aparente por aliasing
# ============================================================================

def calculate_apparent_frequency(
    f_signal: float,
    fs_alias: float,
    use_general_formula: bool = True,
) -> float:
    """Calcula la frecuencia aparente (alias) observada tras submuestreo.

    Fragmento 3:
    - Formulación general plegada al intervalo de Nyquist [0, fs/2]:
        f_alias = abs((f_signal + fs_alias / 2) % fs_alias - fs_alias / 2)
    - Formulación básica por orden armónico más cercano:
        k = round(f_signal / fs_alias)
        f_alias = abs(f_signal - k * fs_alias)

    La formulación general previene interpretaciones erróneas y maneja
    automáticamente armónicos superiores plegados dentro de la banda base.
    """
    fs = float(fs_alias)
    if fs <= 0:
        raise ValueError("La frecuencia de muestreo debe ser positiva.")
    f_sig = float(f_signal)

    if use_general_formula:
        # Plegamiento modular en el intervalo de Nyquist [-fs/2, fs/2] y toma de valor absoluto
        return float(abs((f_sig + fs / 2.0) % fs - fs / 2.0))

    k = round(f_sig / fs)
    return float(abs(f_sig - k * fs))


# ============================================================================
# 4. Transformada de Fourier mediante integración directa
# ============================================================================

def transformada_fourier(x: np.ndarray, t: np.ndarray, f: np.ndarray) -> np.ndarray:
    """Calcula la Transformada de Fourier clásica continua mediante integración directa.

    Fragmento 4:
        X(f) = ∫ x(t) * exp(-j * 2 * pi * f * t) dt

    Adaptación respecto al código fuente original:
        NumPy 2.0+ reemplazó `np.trapz` por `np.trapezoid`. Para máxima compatibilidad
        con cualquier entorno de Python/NumPy, se utiliza `np.trapezoid` si está disponible
        y se recurre a `np.trapz` como respaldo si se ejecuta en NumPy < 2.0.
    """
    x_arr = np.asarray(x, dtype=np.float64)
    t_arr = np.asarray(t, dtype=np.float64)
    f_arr = np.asarray(f, dtype=np.float64)

    # Selección retrocompatible del integrador trapezoidal
    trapz_func = getattr(np, "trapezoid", getattr(np, "trapz", None))
    if trapz_func is None:
        raise RuntimeError("No se encontró función de integración numérica en el entorno de NumPy.")

    X = np.zeros_like(f_arr, dtype=np.complex128)
    for i, fi in enumerate(f_arr):
        integrando = x_arr * np.exp(-1j * 2.0 * np.pi * fi * t_arr)
        X[i] = trapz_func(integrando, t_arr)
    return X


def theoretical_rectangular_pulse_fourier(f: np.ndarray, pulse_width: float = 1.0) -> np.ndarray:
    """Transformada de Fourier analítica de un pulso rectangular: X_teorica = T * sinc(f * T).

    Fragmento 4:
        np.sinc en NumPy calcula sin(pi * x) / (pi * x), coincidiendo con la definición
        de la transformada de un pulso de ancho T.
    """
    T = float(pulse_width)
    f_arr = np.asarray(f, dtype=np.float64)
    return T * np.sinc(f_arr * T)


def fourier_transform_error(X_numerical: np.ndarray, X_theoretical: np.ndarray) -> float:
    """Diferencia máxima absoluta entre el espectro numérico y el teórico."""
    return float(np.max(np.abs(X_numerical - X_theoretical)))


# ============================================================================
# 5. Cálculo de FFT y eje de frecuencias
# ============================================================================

def compute_fft_spectrum(x_muestras: np.ndarray, fs: float) -> Tuple[np.ndarray, np.ndarray]:
    """Cálculo del espectro de amplitud unilateral normalizado mediante FFT.

    Fragmento 5:
        N = len(x_muestras)
        fft_vals = np.fft.fft(x_muestras)
        fft_freqs = np.fft.fftfreq(N, d=1 / fs)
        pos_mask = fft_freqs >= 0
        freqs_pos = fft_freqs[pos_mask]
        fft_magnitude = (2.0 / N) * np.abs(fft_vals[pos_mask])

    Adaptación rigurosa:
        Para un espectro unilateral de amplitud exacto, la componente DC (f = 0)
        y la de Nyquist (f = fs/2 en caso de N par) se escalan por 1/N en lugar de 2/N.
    """
    x = np.asarray(x_muestras, dtype=np.float64).reshape(-1)
    N = len(x)
    if N == 0:
        return np.empty(0, dtype=np.float64), np.empty(0, dtype=np.float64)
    if fs <= 0:
        raise ValueError("La frecuencia de muestreo debe ser mayor que cero.")

    fft_vals = np.fft.fft(x)
    fft_freqs = np.fft.fftfreq(N, d=1.0 / float(fs))

    pos_mask = fft_freqs >= 0
    freqs_pos = fft_freqs[pos_mask]
    fft_magnitude = (2.0 / float(N)) * np.abs(fft_vals[pos_mask])

    # Corrección rigurosa para DC y Nyquist
    if fft_magnitude.size > 0:
        fft_magnitude[0] *= 0.5
        if N % 2 == 0 and fft_magnitude.size > 1:
            fft_magnitude[-1] *= 0.5

    return freqs_pos, fft_magnitude


# ============================================================================
# 6. Cuantización ADC y reconstrucción DAC
# ============================================================================

def quantize_adc_dac(
    v_sampled: np.ndarray,
    bits: int,
    v_min: float = -5.0,
    v_max: float = 5.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Cuantización directa ADC a código digital y reconstrucción DAC a voltaje.

    Fragmento 6:
        n_niveles = 2**bits
        q_step = (v_max - v_min) / n_niveles
        digital_code = np.round((v_sampled - v_min) / q_step)
        digital_code = np.clip(digital_code, 0, n_niveles - 1).astype(int)
        v_reconstructed = digital_code * q_step + v_min
        quant_error = v_sampled - v_reconstructed

    Returns:
        (digital_code, v_reconstructed, quant_error, q_step)
    """
    if bits < 1 or bits > 32:
        raise ValueError("El número de bits debe ser un entero positivo válido.")
    if v_max <= v_min:
        raise ValueError("El rango de voltaje debe cumplir v_max > v_min.")

    v = np.asarray(v_sampled, dtype=np.float64)
    n_niveles = 2 ** int(bits)
    q_step = float(v_max - v_min) / float(n_niveles)

    digital_code = np.round((v - float(v_min)) / q_step)
    digital_code = np.clip(digital_code, 0, n_niveles - 1).astype(np.int64)

    v_reconstructed = digital_code * q_step + float(v_min)
    quant_error = v - v_reconstructed

    return digital_code, v_reconstructed, quant_error, q_step


# ============================================================================
# 7. Reconstrucción por interpolación
# ============================================================================

def interpolate_signal(
    muestras: np.ndarray,
    t_muestreo: np.ndarray,
    t_reconstruido: np.ndarray,
) -> np.ndarray:
    """Aproxima la señal continua mediante interpolación lineal entre muestras discretas.

    Fragmento 7:
        senal_reconstruida = np.interp(t_reconstruido, t_muestreo, muestras)
    """
    return np.interp(
        np.asarray(t_reconstruido, dtype=np.float64),
        np.asarray(t_muestreo, dtype=np.float64),
        np.asarray(muestras, dtype=np.float64),
    )


# ============================================================================
# 8. Cálculos que alimentan las gráficas
# ============================================================================

def compute_minmax_downsample(
    data: np.ndarray,
    start: int,
    end: int,
    max_points: int = 50000,
) -> np.ndarray:
    """Calcula los índices óptimos para trazar formas de onda extensas conservando picos.

    Extraído de la interfaz gráfica a este módulo matemático para desacoplar el cómputo
    del renderizado visual. Al hacer zoom o pan, permite reducir millones de muestras a un
    número visualmente nítido sin perder transitorios ni picos de envolvente.
    """
    n = max(0, int(end) - int(start))
    if n <= 0:
        return np.empty(0, dtype=np.int64)
    if n <= max_points:
        return np.arange(start, end, dtype=np.int64)

    segment = np.asarray(data[start:end])
    bucket_count = max(1, max_points // 2)
    bucket_size = max(1, int(np.ceil(n / bucket_count)))
    full_n = (n // bucket_size) * bucket_size
    indices_parts = []

    if full_n:
        blocks = segment[:full_n].reshape(-1, bucket_size)
        block_starts = start + np.arange(blocks.shape[0], dtype=np.int64) * bucket_size
        indices_parts.append(block_starts + np.argmin(blocks, axis=1))
        indices_parts.append(block_starts + np.argmax(blocks, axis=1))

    if full_n < n:
        tail = segment[full_n:]
        tail_start = start + full_n
        indices_parts.append(np.asarray([
            tail_start + int(np.argmin(tail)),
            tail_start + int(np.argmax(tail)),
        ], dtype=np.int64))

    return np.unique(np.concatenate(indices_parts))


def compute_relative_db_spectrum(
    frequencies: np.ndarray,
    magnitudes: np.ndarray,
    max_frequency: Optional[float] = None,
    floor_db: float = -120.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Calcula la curva espectral normalizada en dB respecto a su pico máximo.

    Utilizado para alimentar la gráfica de comparación espectral entre pistas musicales.
    """
    f = np.asarray(frequencies, dtype=np.float64)
    mag = np.asarray(magnitudes, dtype=np.float64)

    if max_frequency is not None:
        if max_frequency < 0:
            raise ValueError("La frecuencia máxima de corte no puede ser negativa.")
        mask = f <= float(max_frequency)
        f = f[mask]
        mag = mag[mask]

    if f.size == 0 or mag.size == 0:
        return np.empty(0, dtype=np.float64), np.empty(0, dtype=np.float64)

    if mag.size and np.max(mag) > 0:
        peak = float(np.max(mag))
        db = 20.0 * np.log10(np.maximum(mag / peak, 1e-6))
    else:
        db = np.full_like(mag, float(floor_db))

    return f, db

