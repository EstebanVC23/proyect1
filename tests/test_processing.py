import unittest
import numpy as np

from dsp.processing import analyze_audio, dominant_frequency, quantize_adc, resample_signal, zero_order_hold


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


class TestDSPProcessing(unittest.TestCase):
    def test_quantizer(self):
        test_quantizer_levels_and_step()

    def test_resample(self):
        test_resample_changes_number_of_samples_approximately()

    def test_dominant_freq(self):
        test_dominant_frequency_for_tone()

    def test_zoh(self):
        test_zero_order_hold()

    def test_analyze_audio(self):
        test_analyze_audio_returns_adc_dac_metrics()


if __name__ == "__main__":
    unittest.main()
