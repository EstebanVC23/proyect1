import unittest
import numpy as np

from dsp.processing import analyze_audio, dominant_frequency, quantize_adc, resample_signal, zero_order_hold
from dsp.audio_io import to_float_audio


def test_quantizer_levels_and_step():
    x = np.linspace(-1, 1, 1001)
    result = quantize_adc(x, bits=4, vmin=-2, vmax=2)
    assert result["levels"] == 16
    assert np.isclose(result["delta_v"], 0.25)
    assert result["codes"].min() >= 0
    assert result["codes"].max() < 16
    assert np.all(np.isfinite(result["quantized"]))


def test_resample_changes_number_of_samples_approximately():
    fs_in, fs_out, duration = 8000, 4000, 1
    x = np.sin(2 * np.pi * 440 * np.arange(fs_in * duration) / fs_in)
    y = resample_signal(x, fs_in, fs_out)
    assert abs(len(y) - fs_out * duration) <= 1


def test_dominant_frequency_for_tone():
    fs = 8000
    t = np.arange(fs) / fs
    x = np.sin(2 * np.pi * 440 * t)
    assert abs(dominant_frequency(x, fs) - 440) < 2


def test_zero_order_hold():
    samples = np.array([0.0, 0.5, -0.5])
    times = np.array([0.0, 0.2, 1.0, 2.7])
    got = zero_order_hold(samples, fs=1, times=times)
    assert np.allclose(got, [0.0, 0.0, 0.5, -0.5])


def test_analyze_audio_returns_adc_dac_metrics():
    fs = 8000
    t = np.arange(fs // 2) / fs
    x = 0.8 * np.sin(2 * np.pi * 440 * t)
    result = analyze_audio(x, fs, fs, bits=8, vmin=-1, vmax=1)
    assert result["adc"]["levels"] == 256
    assert abs(result["dominant_frequency"] - 440) < 10
    assert result["reconstructed"].shape == x.shape
    assert result["reconstruction_rmse"] >= 0



def test_quantize_adc_silencio_snr():
    """Verifica que el SNR ante una señal en silencio absoluto sea -inf y no +inf."""
    silencio = np.zeros(100)
    res = quantize_adc(silencio, bits=8, vmin=-5.0, vmax=5.0)
    # Ante silencio con error no nulo (debido a offset mid-rise), SNR es -inf
    assert res["snr_quantization_db"] == float("-inf")


def test_quantize_adc_resoluciones_4_8_12_16():
    """Verifica exactitud de niveles y pasos de cuantización para 4, 8, 12 y 16 bits."""
    x = np.sin(2 * np.pi * 5 * np.linspace(0, 1, 500))
    for b in (4, 8, 12, 16):
        res = quantize_adc(x, bits=b, vmin=-5.0, vmax=5.0)
        assert res["levels"] == 2 ** b
        expected_step = 10.0 / (2 ** b)
        assert np.isclose(res["delta_v"], expected_step)
        assert res["max_error_volts"] <= expected_step


def test_magnitude_spectrum_fs_invalida():
    """Verifica que fs <= 0 genere ValueError en magnitude_spectrum."""
    from dsp.processing import magnitude_spectrum
    with unittest.TestCase().assertRaises(ValueError):
        magnitude_spectrum(np.array([1.0, 2.0]), fs=0)


def test_analyze_audio_duracion_invalida():
    """Verifica que max_seconds <= 0 genere ValueError en analyze_audio."""
    x = np.sin(np.linspace(0, 10, 100))
    with unittest.TestCase().assertRaises(ValueError):
        analyze_audio(x, 8000, 8000, bits=8, vmin=-1.0, vmax=1.0, max_seconds=0.0)


def test_to_float_audio_tipos_datos_y_vacio():
    """Verifica la normalización de muestras PCM y manejo de arrays vacíos."""
    # Array vacío
    vacio = to_float_audio(np.array([], dtype=np.int16))
    assert len(vacio) == 0

    # int16
    pcm16 = np.array([-32768, 0, 32767], dtype=np.int16)
    fl16 = to_float_audio(pcm16)
    assert np.all(fl16 >= -1.0) and np.all(fl16 <= 1.0)
    assert np.isclose(fl16[1], 0.0)

    # uint8
    pcm8 = np.array([0, 128, 255], dtype=np.uint8)
    fl8 = to_float_audio(pcm8)
    assert np.all(fl8 >= -1.0) and np.all(fl8 <= 1.0)
    assert np.isclose(fl8[1], 0.0)


class TestDSPProcessing(unittest.TestCase):
    def test_quantizer(self):
        test_quantizer_levels_and_step()

    def test_quantizer_silencio(self):
        test_quantize_adc_silencio_snr()

    def test_quantizer_resoluciones(self):
        test_quantize_adc_resoluciones_4_8_12_16()

    def test_resample(self):
        test_resample_changes_number_of_samples_approximately()

    def test_dominant_freq(self):
        test_dominant_frequency_for_tone()

    def test_magnitude_spectrum_fs(self):
        test_magnitude_spectrum_fs_invalida()

    def test_zoh(self):
        test_zero_order_hold()

    def test_analyze_audio(self):
        test_analyze_audio_returns_adc_dac_metrics()

    def test_analyze_audio_duracion(self):
        test_analyze_audio_duracion_invalida()

    def test_to_float_audio(self):
        test_to_float_audio_tipos_datos_y_vacio()


if __name__ == "__main__":
    unittest.main()
