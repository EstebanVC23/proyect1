"""Gestión y renderizado de gráficas Matplotlib para Mira / Señales DSP.

Desacopla la lógica de visualización y trazado de ui/app.py.
Todos los cálculos que alimentan las figuras se obtienen directamente
desde dsp.calculations (decimación min-max, espectros en dB, etc.).
"""
from __future__ import annotations

import os
import tkinter as tk
from typing import Dict, List, Optional

import matplotlib
from cycler import cycler
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
import numpy as np

from dsp.calculations import compute_minmax_downsample, compute_relative_db_spectrum
from dsp.models import AudioTrack

# ----------------------------------------------------------------------------
# Constantes visuales compartidas
# ----------------------------------------------------------------------------
CARD = "#ffffff"
BORDER = "#dbe3ee"
BORDER_DARK = "#b9c6d8"
MUTED = "#64748b"

C_BLUE = "#2563eb"
C_ORANGE = "#ea7a1a"
C_GREEN = "#16a34a"
C_PURPLE = "#8b5cf6"
C_RED = "#dc2626"
C_TEAL = "#0d9488"
C_PINK = "#db2777"

# Configuración global del motor Matplotlib
matplotlib.rcParams.update({
    "font.family": "DejaVu Sans",
    "axes.titlesize": 10,
    "axes.titleweight": "bold",
    "axes.titlecolor": "#14243a",
    "axes.labelsize": 9,
    "axes.labelcolor": "#475569",
    "axes.edgecolor": BORDER_DARK,
    "axes.facecolor": "#fbfcfe",
    "figure.facecolor": CARD,
    "grid.color": "#d9e1ec",
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "legend.frameon": True,
    "legend.framealpha": 0.92,
    "legend.edgecolor": BORDER,
    "axes.prop_cycle": cycler(color=[C_BLUE, C_ORANGE, C_GREEN, C_PURPLE, C_RED, C_TEAL, C_PINK]),
})


def style_axes(ax) -> None:
    """Aplica un estilo uniforme, limpio y moderno a un eje de Matplotlib."""
    ax.set_facecolor("#fbfcfe")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(BORDER_DARK)
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.grid(True, alpha=0.55, linewidth=0.6)


def pack_plot(canvas: FigureCanvasTkAgg) -> NavigationToolbar2Tk:
    """Empaqueta una figura Matplotlib con su barra de herramientas estándar."""
    parent = canvas.get_tk_widget().master
    toolbar = NavigationToolbar2Tk(canvas, parent, pack_toolbar=False)
    toolbar.update()
    toolbar.pack(side=tk.BOTTOM, fill=tk.X)
    canvas.get_tk_widget().configure(bg=CARD, highlightthickness=0)
    canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
    return toolbar


class PlotManager:
    """Controlador especializado en la creación y actualización de gráficas."""

    def __init__(
        self,
        tab_full_wave: tk.Widget,
        tab_signal: tk.Widget,
        tab_fft: tk.Widget,
        compare_plot_frame: tk.Widget,
    ):
        # 1. Pestaña: Canción completa
        self.fig_full_wave = Figure(figsize=(10.0, 6.0), dpi=90, constrained_layout=True, facecolor=CARD)
        self.ax_full_wave = self.fig_full_wave.add_subplot(111)
        self.canvas_full_wave = FigureCanvasTkAgg(self.fig_full_wave, master=tab_full_wave)
        self.toolbar_full_wave = pack_plot(self.canvas_full_wave)
        self._overview_line = None
        self._overview_xlim_cid = None
        self._current_track_for_wave: Optional[AudioTrack] = None

        # 2. Pestaña: Señal y ADC-DAC
        self.fig_signal = Figure(figsize=(9.5, 6.5), dpi=90, constrained_layout=True, facecolor=CARD)
        self.ax_original = self.fig_signal.add_subplot(221)
        self.ax_sampled = self.fig_signal.add_subplot(222)
        self.ax_quantized = self.fig_signal.add_subplot(223)
        self.ax_reconstructed = self.fig_signal.add_subplot(224)
        self.canvas_signal = FigureCanvasTkAgg(self.fig_signal, master=tab_signal)
        self.toolbar_signal = pack_plot(self.canvas_signal)

        # 3. Pestaña: Espectro FFT
        self.fig_fft = Figure(figsize=(9.5, 6.5), dpi=90, constrained_layout=True, facecolor=CARD)
        self.ax_fft = self.fig_fft.add_subplot(111)
        self.canvas_fft = FigureCanvasTkAgg(self.fig_fft, master=tab_fft)
        self.toolbar_fft = pack_plot(self.canvas_fft)

        # 4. Pestaña: Comparación de géneros
        self.fig_compare = Figure(figsize=(9.5, 4.5), dpi=90, constrained_layout=True, facecolor=CARD)
        self.ax_compare = self.fig_compare.add_subplot(111)
        self.canvas_compare = FigureCanvasTkAgg(self.fig_compare, master=compare_plot_frame)
        self.toolbar_compare = pack_plot(self.canvas_compare)

    # ------------------------------------------------------------------------
    # Estado vacío (Empty State)
    # ------------------------------------------------------------------------
    def draw_empty_state(self, message: str = "Carga un archivo WAV") -> None:
        """Limpia y coloca leyendas informativas cuando no hay audio disponible."""
        for ax in (self.ax_original, self.ax_sampled, self.ax_quantized, self.ax_reconstructed):
            ax.clear()
            ax.text(0.5, 0.5, message, ha="center", va="center", transform=ax.transAxes, color="#68788f", wrap=True, fontsize=10)
            ax.set_axis_off()
        self.canvas_signal.draw_idle()

        self.ax_fft.clear()
        self.ax_fft.text(0.5, 0.5, "El espectro aparecerá aquí", ha="center", va="center", transform=self.ax_fft.transAxes, color="#68788f", fontsize=11)
        self.ax_fft.set_axis_off()
        self.canvas_fft.draw_idle()

        self.ax_compare.clear()
        self.ax_compare.text(0.5, 0.5, "Carga uno o más WAV para comparar", ha="center", va="center", transform=self.ax_compare.transAxes, color="#68788f", fontsize=11)
        self.ax_compare.set_axis_off()
        self.canvas_compare.draw_idle()

    # ------------------------------------------------------------------------
    # Trazado de la canción completa
    # ------------------------------------------------------------------------
    def draw_full_song(self, track: AudioTrack, start_seconds: float, end_seconds: float) -> None:
        """Dibuja la forma de onda íntegra y sombrea el intervalo de análisis seleccionado."""
        ax = self.ax_full_wave
        if self._overview_xlim_cid is not None:
            try:
                ax.callbacks.disconnect(self._overview_xlim_cid)
            except (ValueError, KeyError):
                pass
            self._overview_xlim_cid = None

        ax.clear()
        self._overview_line = None
        self._current_track_for_wave = track

        if not track.loaded or track.mono is None or track.sample_rate is None:
            ax.text(0.5, 0.5, "Carga un archivo WAV para ver la canción completa", ha="center", va="center", transform=ax.transAxes, color="#68788f", wrap=True, fontsize=11)
            ax.set_axis_off()
            self.fig_full_wave.suptitle("Forma de onda completa", fontsize=11, color="#203651", fontweight="bold")
            self.canvas_full_wave.draw_idle()
            return

        ax.set_axis_on()
        style_axes(ax)
        fs = int(track.sample_rate)
        n = len(track.mono)
        duration = n / float(fs)

        # Cálculo de reducción mediante el módulo matemático dsp.calculations
        initial_indices = compute_minmax_downsample(track.mono, 0, n)
        self._overview_line, = ax.plot(
            initial_indices / float(fs), track.mono[initial_indices],
            color=C_BLUE, linewidth=0.7, label="Audio original"
        )

        selected_start = max(0.0, min(duration, start_seconds))
        selected_end = max(0.0, min(duration, end_seconds))
        if selected_end > selected_start:
            ax.axvspan(selected_start, selected_end, color="#f59e0b", alpha=0.22, label="Intervalo de análisis")
            ax.axvline(selected_start, color="#d97706", linestyle="--", linewidth=0.9)
            ax.axvline(selected_end, color="#d97706", linestyle="--", linewidth=0.9)

        ax.set_title(f"Canción completa · {track.title or os.path.basename(track.file_path or '')}")
        ax.set_xlabel("Tiempo de la canción (s)")
        ax.set_ylabel("Amplitud normalizada")
        ax.set_xlim(0.0, max(duration, 1.0 / fs))

        ymin = float(np.min(track.mono))
        ymax = float(np.max(track.mono))
        if np.isclose(ymin, ymax):
            padding = max(0.05, abs(ymin) * 0.05)
        else:
            padding = (ymax - ymin) * 0.06
        ax.set_ylim(ymin - padding, ymax + padding)
        ax.legend(fontsize=8, loc="upper right")

        self.fig_full_wave.suptitle(
            "Usa Zoom o Pan en la barra inferior. Inicio restablece la vista completa.",
            fontsize=9, color=MUTED
        )
        self._overview_xlim_cid = ax.callbacks.connect("xlim_changed", self._on_wave_zoom_changed)
        self.canvas_full_wave.draw_idle()

    def _on_wave_zoom_changed(self, _axes=None) -> None:
        """Callback dinámico para recalcular los puntos visibles al hacer zoom/pan en la forma de onda."""
        track = self._current_track_for_wave
        if track is None or not track.loaded or track.mono is None or self._overview_line is None:
            return
        x_min, x_max = self.ax_full_wave.get_xlim()
        if x_max < x_min:
            x_min, x_max = x_max, x_min
        fs = int(track.sample_rate)
        start = max(0, int(np.floor(x_min * fs)))
        end = min(len(track.mono), int(np.ceil(x_max * fs)) + 1)
        if end <= start:
            return

        # Recálculo de envolvente min/max
        indices = compute_minmax_downsample(track.mono, start, end)
        self._overview_line.set_data(indices / float(fs), track.mono[indices])
        self.canvas_full_wave.draw_idle()

    # ------------------------------------------------------------------------
    # Trazado de señales temporales y espectro FFT de la pista activa
    # ------------------------------------------------------------------------
    def draw_current_analysis(
        self,
        track: AudioTrack,
        result: Optional[dict],
        start_seconds: float,
        end_seconds: float,
        fs_processing: int,
        bits: int,
    ) -> None:
        """Actualiza los cuatro cuadrantes temporales y el espectro de la pista seleccionada."""
        self.draw_full_song(track, start_seconds, end_seconds)

        if not track.loaded or result is None:
            msg = "Carga un archivo WAV" if not track.loaded else "El intervalo seleccionado está fuera de la duración de esta pista"
            self.draw_empty_state(msg)
            return

        time_offset = result.get("analysis_start_seconds", 0.0)
        t0 = result["t_original"] + time_offset
        ts = result["t_sampled"] + time_offset

        # 1. Señal original
        ax = self.ax_original
        ax.clear()
        ax.set_axis_on()
        ax.plot(t0, result["original"], linewidth=0.65, color=C_BLUE)
        ax.set_title("Señal original · intervalo completo")
        ax.set_xlabel("Tiempo de la canción (s)")
        ax.set_ylabel("Amplitud normalizada")
        style_axes(ax)

        # 2. Muestreo
        ax = self.ax_sampled
        ax.clear()
        ax.set_axis_on()
        ax.plot(t0, result["original"], linewidth=0.55, alpha=0.4, color=C_BLUE, label="Referencia WAV")
        marker_step = max(1, int(np.ceil(len(ts) / 9000)))
        ax.plot(
            ts, result["sampled"], linewidth=0.65, color=C_ORANGE, alpha=0.9,
            marker=".", markersize=2, markevery=marker_step,
            label="Muestras remuestreadas", zorder=3
        )
        ax.set_title(f"Muestreo · fs={fs_processing} Hz")
        ax.set_xlabel("Tiempo de la canción (s)")
        ax.set_ylabel("Amplitud")
        style_axes(ax)
        ax.legend(fontsize=7, loc="best")

        # 3. Señal cuantizada ADC (en Voltaje real)
        ax = self.ax_quantized
        ax.clear()
        ax.set_axis_on()
        adc = result["adc"]
        quant_v = adc["quantized_volts"]
        levels = adc["levels"]
        delta_v = adc["delta_v"]
        snr_db = adc.get("snr_quantization_db", float("inf"))
        vmin_adc = float(adc["vmin"])
        vmax_adc = float(adc["vmax"])

        ax.step(ts, quant_v, where="post", linewidth=0.85, color=C_GREEN, label=f"ADC ({bits} bits, {levels} niv.)")
        snr_str = f"{snr_db:.1f} dB" if snr_db < 200 else "∞"
        ax.set_title(f"ADC · {bits} bits ({levels} niveles)")
        ax.set_xlabel("Tiempo de la canción (s)")
        ax.set_ylabel("Voltaje cuantizado (V)")
        ax.set_ylim(vmin_adc - 0.4, vmax_adc + 0.4)
        ax.axhline(vmax_adc, color="#94a3b8", linestyle=":", linewidth=0.7, alpha=0.6)
        ax.axhline(vmin_adc, color="#94a3b8", linestyle=":", linewidth=0.7, alpha=0.6)
        ax.axhline(0.0, color="#cbd5e1", linestyle="-", linewidth=0.5, alpha=0.4)
        style_axes(ax)
        ax.legend(fontsize=7, loc="upper right")

        # 4. Reconstrucción DAC
        ax = self.ax_reconstructed
        ax.clear()
        ax.set_axis_on()
        ax.plot(t0, result["original"], linewidth=0.65, alpha=0.65, color=C_BLUE, label="Original")
        ax.step(t0, result["reconstructed"], where="post", linewidth=0.8, color=C_PURPLE, label=f"DAC ({bits} bits)")
        ax.set_title(f"Reconstrucción DAC")
        ax.set_xlabel("Tiempo de la canción (s)")
        ax.set_ylabel("Amplitud normalizada")
        style_axes(ax)
        ax.legend(fontsize=7, loc="best")

        for time_ax in (self.ax_original, self.ax_sampled, self.ax_quantized, self.ax_reconstructed):
            time_ax.set_xlim(time_offset, time_offset + result["analysis_duration"])

        self.fig_signal.suptitle(
            f"{track.genre} — {track.title or os.path.basename(track.file_path or '')} "
            f"[{result.get('analysis_start_seconds', 0.0):.2f}–{result.get('analysis_end_seconds', result['analysis_duration']):.2f} s]",
            fontsize=11, color="#203651", fontweight="bold"
        )
        self.canvas_signal.draw_idle()

        # 5. Espectro FFT
        self.ax_fft.clear()
        self.ax_fft.set_axis_on()
        f = result["frequencies"]
        mag = result["magnitudes"]
        fq = result["quantized_frequencies"]
        magq = result["quantized_magnitudes"]

        # Nyquist de procesamiento: determina la banda analizable con fs_processing
        nyquist_proc = float(result["nyquist"])  # = fs_processing / 2
        # Nyquist del WAV original: cambia entre pistas con distinta fs_original
        nyquist_orig = float(result["fs_original"]) / 2.0
        # Eje X hasta el Nyquist de procesamiento (o 12 kHz si es mayor)
        max_f = min(nyquist_proc, 12000.0)
        mask = f <= max_f
        maskq = fq <= max_f

        self.ax_fft.plot(f[mask], mag[mask], linewidth=0.9, color=C_BLUE, label="Muestras remuestreadas")
        self.ax_fft.plot(fq[maskq], magq[maskq], linewidth=0.8, alpha=0.8, color=C_ORANGE, label="Señal cuantizada")
        # Línea Nyquist del procesamiento (límite que impone el ADC configurado)
        self.ax_fft.axvline(
            nyquist_proc, color=C_RED, linestyle="--", linewidth=1.1,
            label=f"Nyquist proc. {nyquist_proc:.0f} Hz  (fs={fs_processing} Hz)"
        )
        # Línea Nyquist del WAV original: solo si difiere del de procesamiento y es visible
        if abs(nyquist_orig - nyquist_proc) > 10 and nyquist_orig <= max_f * 1.05:
            self.ax_fft.axvline(
                nyquist_orig, color=C_TEAL, linestyle=":", linewidth=1.0,
                label=f"Nyquist WAV {nyquist_orig:.0f} Hz  (fs={result['fs_original']} Hz)"
            )
        self.ax_fft.set_title(f"Espectro FFT · pico ≈ {result['dominant_frequency']:.2f} Hz · Nyquist proc.={nyquist_proc:.0f} Hz")
        self.ax_fft.set_xlabel("Frecuencia (Hz)")
        self.ax_fft.set_ylabel("Magnitud (escala relativa)")
        style_axes(self.ax_fft)
        self.ax_fft.legend(fontsize=8, loc="best")
        self.ax_fft.set_xlim(0, max_f if max_f > 0 else nyquist_proc)
        self.canvas_fft.draw_idle()

    # ------------------------------------------------------------------------
    # Comparación espectral y métricas comparativas
    # ------------------------------------------------------------------------
    def draw_genre_comparison(
        self,
        tracks: List[AudioTrack],
        analysis_results: Dict[int, dict],
        fs_processing: int,
        compare_tree: tk.Widget,
    ) -> None:
        """Actualiza la gráfica comparativa en decibelios y llena la tabla de métricas."""
        self.ax_compare.clear()
        for item in compare_tree.get_children():
            compare_tree.delete(item)

        loaded_tracks = [t for t in tracks if t.loaded and t.slot in analysis_results]
        if not loaded_tracks:
            self.ax_compare.text(0.5, 0.5, "Carga uno o más WAV para comparar", ha="center", va="center", transform=self.ax_compare.transAxes, color="#68788f", fontsize=11)
            self.ax_compare.set_axis_off()
            self.canvas_compare.draw_idle()
            return

        self.ax_compare.set_axis_on()
        max_f = min(fs_processing / 2.0, 12000.0)

        for row_index, track in enumerate(loaded_tracks):
            result = analysis_results[track.slot]
            # Cálculo desacoplado de espectro normalizado en dB desde dsp.calculations
            f_keep, db_keep = compute_relative_db_spectrum(
                result["frequencies"],
                result["magnitudes"],
                max_frequency=max_f,
            )
            self.ax_compare.plot(f_keep, db_keep, linewidth=0.9, label=f"{track.genre} (pista {track.slot})")

            adc = result["adc"]
            compare_tree.insert("", "end", values=(
                track.genre,
                f"{result['dominant_frequency']:.2f}",
                f"{result['nyquist']:.0f}",
                adc["bits"],
                adc["levels"],
                f"{adc['rmse_normalized']:.5f}",
                f"{result['reconstruction_rmse']:.5f}",
            ), tags=("odd" if row_index % 2 else "even",))

        self.ax_compare.set_title("Comparación espectral · cada espectro se normaliza respecto a su pico")
        self.ax_compare.set_xlabel("Frecuencia (Hz)")
        self.ax_compare.set_ylabel("Magnitud relativa (dB respecto al pico)")
        self.ax_compare.set_xlim(0, max_f if max_f > 0 else fs_processing / 2)
        self.ax_compare.set_ylim(-100, 2)
        style_axes(self.ax_compare)
        self.ax_compare.legend(fontsize=8, loc="best")
        self.canvas_compare.draw_idle()

