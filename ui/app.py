"""Interfaz Tkinter para la aplicación Mira / AudioLab DSP.

Controlador principal de la interfaz de usuario.
Refactorizado para mantener ui/app.py liviano y legible:
- Los cálculos matemáticos para señales y gráficas residen en `dsp.calculations`.
- La creación, maquetación y renderizado de gráficos reside en `ui.plots.PlotManager`.
"""
from __future__ import annotations

import os
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

import numpy as np

from dsp.audio_io import load_wav, filename_title, to_float_audio
from dsp.models import AudioTrack
from dsp.processing import analyze_audio, MAX_ANALYSIS_SECONDS
from dsp.reporting import export_csv, export_text_report
from ui.plots import PlotManager


GENRES = ["Clásica / barroca", "Metal", "Salsa", "Electrónica", "Música de cantina", "Reguetón"]
DEFAULT_GENRES = ["Clásica / barroca", "Metal", "Salsa", "Música de cantina", "Reguetón"]
FS_OPTIONS = ["8000", "11025", "16000", "22050", "32000", "44100", "48000"]
BIT_OPTIONS = ["4", "8", "12", "16"]

# ----------------------------------------------------------------------------
# Paleta de colores de la interfaz
# ----------------------------------------------------------------------------
BG = "#eef2f7"            # fondo general
CARD = "#ffffff"          # tarjetas / paneles
NAVY = "#0f1b2d"          # cabecera y barra de estado
NAVY_SOFT = "#1b2b44"
PRIMARY = "#2563eb"       # acento principal
PRIMARY_DARK = "#1d4ed8"
PRIMARY_LIGHT = "#dbeafe"
SUCCESS = "#16a34a"
SUCCESS_DARK = "#15803d"
DANGER = "#dc2626"
DANGER_DARK = "#b91c1c"
TEXT = "#1e293b"
MUTED = "#64748b"
BORDER = "#dbe3ee"
BORDER_DARK = "#b9c6d8"
STRIPE = "#f6f9fd"        # filas alternas de las tablas
INFO_BG = "#f1f5fb"


class AudioDSPApp:
    """Controlador y ventana principal de la aplicación Mira / AudioLab DSP."""

    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Mira — AudioLab DSP | ADC · DAC · FFT")
        self.root.geometry("1450x900")
        self.root.minsize(1120, 720)
        self.root.configure(bg=BG)

        self.fs_processing = 22050
        self.bits = 8
        self.vmin = -1.0
        self.vmax = 1.0
        self.analysis_results: dict[int, dict] = {}
        self.analysis_start_seconds = 0.0
        self.analysis_end_seconds = MAX_ANALYSIS_SECONDS
        self._audio_backend = None
        self.current_slot = 1
        self._changing_form = False

        self.tracks = [
            AudioTrack(slot=i + 1, genre=DEFAULT_GENRES[i], title="", artist="", year="", source="")
            for i in range(5)
        ]

        self._configure_style()
        self._build_header()
        self._build_parameters()
        self._build_main_area()
        self._build_status_bar()
        self._refresh_track_table()
        self.track_tree.selection_set("1")
        self.track_tree.focus("1")
        self._load_metadata_form(self.tracks[0])
        self._draw_empty_state()
        self._draw_full_song(self.tracks[0])
        self._update_conclusions()
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    # ------------------------------------------------------------------
    # Estilo visual de la interfaz
    # ------------------------------------------------------------------
    def _configure_style(self):
        style = ttk.Style()
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass

        base_font = ("Segoe UI", 9)
        bold_font = ("Segoe UI", 9, "bold")

        style.configure(".", font=base_font)
        style.configure("TFrame", background=BG)
        style.configure("Card.TFrame", background=CARD)
        style.configure("Bar.TFrame", background=CARD)

        style.configure("TLabel", background=BG, foreground=TEXT, font=base_font)
        style.configure("Card.TLabel", background=CARD, foreground=TEXT, font=base_font)
        style.configure("CardMuted.TLabel", background=CARD, foreground=MUTED, font=("Segoe UI", 9))
        style.configure("Field.TLabel", background=CARD, foreground="#475569", font=("Segoe UI", 8, "bold"))
        style.configure("Title.TLabel", font=("Segoe UI", 20, "bold"), foreground="#14243a", background=BG)
        style.configure("Sub.TLabel", font=("Segoe UI", 9), foreground=MUTED, background=BG)
        style.configure("CardTitle.TLabel", font=("Segoe UI", 11, "bold"), foreground="#183153", background=CARD)

        # Paneles con título
        style.configure("Card.TLabelframe", background=CARD, bordercolor=BORDER, relief="solid", borderwidth=1)
        style.configure("Card.TLabelframe.Label", background=CARD, foreground=PRIMARY_DARK, font=("Segoe UI", 10, "bold"))

        # Botones
        style.configure("TButton", padding=(10, 6), font=base_font, background="#e6edf7",
                        foreground=TEXT, bordercolor=BORDER_DARK, focusthickness=0, focuscolor="none", relief="flat")
        style.map("TButton",
                  background=[("pressed", "#c9d7ea"), ("active", "#d8e4f4")],
                  bordercolor=[("active", PRIMARY)])

        style.configure("Accent.TButton", font=bold_font, padding=(12, 7), background=PRIMARY,
                        foreground="#ffffff", bordercolor=PRIMARY, focusthickness=0, focuscolor="none")
        style.map("Accent.TButton",
                  background=[("pressed", PRIMARY_DARK), ("active", PRIMARY_DARK)],
                  foreground=[("disabled", "#e2e8f0")])

        style.configure("Play.TButton", font=bold_font, padding=(10, 6), background=SUCCESS,
                        foreground="#ffffff", bordercolor=SUCCESS, focusthickness=0, focuscolor="none")
        style.map("Play.TButton", background=[("pressed", SUCCESS_DARK), ("active", SUCCESS_DARK)])

        style.configure("Stop.TButton", font=bold_font, padding=(10, 6), background=DANGER,
                        foreground="#ffffff", bordercolor=DANGER, focusthickness=0, focuscolor="none")
        style.map("Stop.TButton", background=[("pressed", DANGER_DARK), ("active", DANGER_DARK)])

        # Campos de entrada
        style.configure("TEntry", fieldbackground="#ffffff", bordercolor=BORDER_DARK,
                        lightcolor=BORDER, darkcolor=BORDER, padding=4)
        style.map("TEntry", bordercolor=[("focus", PRIMARY)], lightcolor=[("focus", PRIMARY)])
        style.configure("TCombobox", fieldbackground="#ffffff", background="#e6edf7",
                        bordercolor=BORDER_DARK, arrowcolor=MUTED, padding=4)
        style.map("TCombobox",
                  fieldbackground=[("readonly", "#ffffff")],
                  bordercolor=[("focus", PRIMARY)],
                  arrowcolor=[("active", PRIMARY)])
        self.root.option_add("*TCombobox*Listbox.background", "#ffffff")
        self.root.option_add("*TCombobox*Listbox.foreground", TEXT)
        self.root.option_add("*TCombobox*Listbox.selectBackground", PRIMARY)
        self.root.option_add("*TCombobox*Listbox.selectForeground", "#ffffff")
        self.root.option_add("*TCombobox*Listbox.font", ("Segoe UI", 9))

        # Pestañas
        style.configure("TNotebook", background=BG, borderwidth=0, tabmargins=(0, 4, 0, 0))
        style.configure("TNotebook.Tab", padding=(16, 9), font=bold_font, background="#dde5f0",
                        foreground=MUTED, borderwidth=0)
        style.map("TNotebook.Tab",
                  background=[("selected", CARD), ("active", "#eaf1fb")],
                  foreground=[("selected", PRIMARY_DARK), ("active", TEXT)])

        # Tablas
        style.configure("Treeview", font=base_font, rowheight=29, background="#ffffff",
                        fieldbackground="#ffffff", foreground=TEXT, bordercolor=BORDER, borderwidth=1)
        style.configure("Treeview.Heading", font=bold_font, background="#e4ecf7", foreground="#183153",
                        relief="flat", padding=(6, 6))
        style.map("Treeview.Heading", background=[("active", "#d4e1f3")])
        style.map("Treeview",
                  background=[("selected", PRIMARY_LIGHT)],
                  foreground=[("selected", "#14243a")])

        # Separadores
        style.configure("TSeparator", background=BORDER)
        style.configure("TPanedwindow", background=BG)
        style.configure("Sash", sashthickness=6, background=BG)

    def _stripe_tree(self, tree: ttk.Treeview):
        """Colorea filas alternas en una tabla."""
        tree.tag_configure("odd", background=STRIPE)
        tree.tag_configure("even", background="#ffffff")

    # ------------------------------------------------------------------
    # Construcción de la interfaz
    # ------------------------------------------------------------------
    def _build_header(self):
        header = tk.Frame(self.root, bg=NAVY)
        header.pack(fill="x")
        inner = tk.Frame(header, bg=NAVY)
        inner.pack(fill="x", padx=22, pady=(14, 12))

        badge = tk.Label(inner, text=" ∿ ", bg=PRIMARY, fg="#ffffff", font=("Segoe UI", 16, "bold"))
        badge.pack(side="left", padx=(0, 12))
        tk.Label(inner, text="Mira — AudioLab DSP", bg=NAVY, fg="#ffffff", font=("Segoe UI", 20, "bold")).pack(side="left")
        tk.Label(
            inner, text="Análisis de audio · muestreo · ADC / DAC · FFT",
            bg=NAVY, fg="#9fb3cf", font=("Segoe UI", 10)
        ).pack(side="left", padx=(16, 0), pady=(8, 0))
        tk.Label(
            inner, text="LABORATORIO DE SEÑALES", bg=NAVY_SOFT, fg="#93c5fd",
            font=("Segoe UI", 8, "bold"), padx=12, pady=5
        ).pack(side="right", pady=(4, 0))
        tk.Frame(header, bg=PRIMARY, height=3).pack(fill="x")

    def _build_parameters(self):
        outer = ttk.Frame(self.root, padding=(18, 14, 18, 10))
        outer.pack(fill="x")
        panel = ttk.LabelFrame(outer, text="  Parámetros de procesamiento  ", padding=(14, 10), style="Card.TLabelframe")
        panel.pack(fill="x")

        self.fs_var = tk.StringVar(value=str(self.fs_processing))
        self.bits_var = tk.StringVar(value=str(self.bits))
        self.vmin_var = tk.StringVar(value=str(self.vmin))
        self.vmax_var = tk.StringVar(value=str(self.vmax))

        ttk.Label(panel, text="FRECUENCIA DE PROCESAMIENTO (HZ)", style="Field.TLabel").grid(row=0, column=0, sticky="w", padx=(0, 7))
        self.fs_combo = ttk.Combobox(panel, textvariable=self.fs_var, values=FS_OPTIONS, width=10)
        self.fs_combo.grid(row=1, column=0, sticky="ew", padx=(0, 14), pady=(4, 0))
        ttk.Label(panel, text="RESOLUCIÓN ADC (BITS)", style="Field.TLabel").grid(row=0, column=1, sticky="w", padx=(0, 7))
        ttk.Combobox(panel, textvariable=self.bits_var, values=BIT_OPTIONS, width=7, state="readonly").grid(row=1, column=1, sticky="ew", padx=(0, 14), pady=(4, 0))
        ttk.Label(panel, text="V MÍNIMO (V)", style="Field.TLabel").grid(row=0, column=2, sticky="w", padx=(0, 7))
        ttk.Entry(panel, textvariable=self.vmin_var, width=9).grid(row=1, column=2, sticky="ew", padx=(0, 14), pady=(4, 0))
        ttk.Label(panel, text="V MÁXIMO (V)", style="Field.TLabel").grid(row=0, column=3, sticky="w", padx=(0, 7))
        ttk.Entry(panel, textvariable=self.vmax_var, width=9).grid(row=1, column=3, sticky="ew", padx=(0, 14), pady=(4, 0))
        ttk.Button(panel, text="Aplicar y recalcular", style="Accent.TButton", command=self._apply_parameters).grid(row=1, column=4, padx=(0, 10), pady=(4, 0))
        ttk.Button(panel, text="Exportar CSV", command=self._export_csv).grid(row=1, column=5, padx=(0, 8), pady=(4, 0))
        ttk.Button(panel, text="Guardar informe TXT", command=self._export_txt).grid(row=1, column=6, pady=(4, 0))
        for col in range(7):
            panel.grid_columnconfigure(col, weight=0)
        panel.grid_columnconfigure(0, weight=1)

        # Controles de reproducción y selección del intervalo de la canción a analizar.
        self.analysis_start_var = tk.StringVar(value="0")
        self.analysis_end_var = tk.StringVar(value=f"{MAX_ANALYSIS_SECONDS:g}")

        ttk.Separator(panel).grid(row=2, column=0, columnspan=7, sticky="ew", pady=(12, 0))
        interval_bar = ttk.Frame(panel, style="Card.TFrame")
        interval_bar.grid(row=3, column=0, columnspan=7, sticky="ew", pady=(10, 0))
        ttk.Label(interval_bar, text="Intervalo del audio (s):", style="Card.TLabel", font=("Segoe UI", 9, "bold")).pack(side="left", padx=(0, 10))
        ttk.Label(interval_bar, text="Inicio", style="Card.TLabel").pack(side="left", padx=(0, 4))
        ttk.Entry(interval_bar, textvariable=self.analysis_start_var, width=8).pack(side="left", padx=(0, 10))
        ttk.Label(interval_bar, text="Fin", style="Card.TLabel").pack(side="left", padx=(0, 4))
        ttk.Entry(interval_bar, textvariable=self.analysis_end_var, width=8).pack(side="left", padx=(0, 8))
        ttk.Button(interval_bar, text="Aplicar intervalo", command=self._apply_time_range).pack(side="left", padx=(0, 14))
        ttk.Button(interval_bar, text="▶ Reproducir canción", style="Play.TButton", command=self._play_full_audio).pack(side="left", padx=(0, 8))
        ttk.Button(interval_bar, text="▶ Reproducir intervalo", style="Play.TButton", command=self._play_selected_fragment).pack(side="left", padx=(0, 8))
        ttk.Button(interval_bar, text="■ Detener", style="Stop.TButton", command=self._stop_audio).pack(side="left")
        ttk.Label(interval_bar, text=f"Máximo {MAX_ANALYSIS_SECONDS:g} s por análisis", style="CardMuted.TLabel").pack(side="right", padx=(10, 0))

    def _build_main_area(self):
        main = ttk.Frame(self.root, padding=(18, 0, 18, 10))
        main.pack(fill="both", expand=True)
        paned = ttk.Panedwindow(main, orient="horizontal")
        paned.pack(fill="both", expand=True)

        left = ttk.Frame(paned, padding=14, style="Card.TFrame")
        right = ttk.Frame(paned, padding=(12, 0, 0, 0))
        paned.add(left, weight=0)
        paned.add(right, weight=1)

        ttk.Label(left, text="Biblioteca de audios", style="CardTitle.TLabel").pack(anchor="w", pady=(0, 4))
        ttk.Label(left, text="Carga un WAV por categoría. Se permite reemplazarlo.", wraplength=300, style="CardMuted.TLabel").pack(anchor="w", pady=(0, 10))

        columns = ("slot", "genre", "file", "rate", "duration")
        self.track_tree = ttk.Treeview(left, columns=columns, show="headings", height=7, selectmode="browse")
        self.track_tree.heading("slot", text="#")
        self.track_tree.heading("genre", text="Género")
        self.track_tree.heading("file", text="Archivo WAV")
        self.track_tree.heading("rate", text="Fs")
        self.track_tree.heading("duration", text="Duración")
        self.track_tree.column("slot", width=25, anchor="center", stretch=False)
        self.track_tree.column("genre", width=112, anchor="w")
        self.track_tree.column("file", width=115, anchor="w")
        self.track_tree.column("rate", width=56, anchor="e")
        self.track_tree.column("duration", width=55, anchor="e")
        self._stripe_tree(self.track_tree)
        self.track_tree.pack(fill="x")
        self.track_tree.bind("<<TreeviewSelect>>", self._on_track_select)

        row_buttons = ttk.Frame(left, style="Card.TFrame")
        row_buttons.pack(fill="x", pady=(10, 14))
        ttk.Button(row_buttons, text="Cargar / reemplazar WAV", style="Accent.TButton", command=self._load_wav).pack(fill="x", pady=(0, 6))
        ttk.Button(row_buttons, text="Quitar audio", command=self._remove_audio).pack(fill="x")

        ttk.Separator(left).pack(fill="x", pady=(0, 12))
        ttk.Label(left, text="Ficha musical", style="CardTitle.TLabel").pack(anchor="w", pady=(0, 6))
        form = ttk.Frame(left, style="Card.TFrame")
        form.pack(fill="x")
        self.genre_var = tk.StringVar()
        self.title_var = tk.StringVar()
        self.artist_var = tk.StringVar()
        self.year_var = tk.StringVar()
        self.source_var = tk.StringVar()

        self._form_row(form, 0, "Género / categoría", ttk.Combobox(form, textvariable=self.genre_var, values=GENRES, state="readonly"))
        self._form_row(form, 1, "Título de la obra", ttk.Entry(form, textvariable=self.title_var))
        self._form_row(form, 2, "Compositor / intérprete", ttk.Entry(form, textvariable=self.artist_var))
        self._form_row(form, 3, "Año", ttk.Entry(form, textvariable=self.year_var))
        self._form_row(form, 4, "Fuente / referencia", ttk.Entry(form, textvariable=self.source_var))
        ttk.Button(left, text="Guardar ficha", command=self._save_metadata, style="Accent.TButton").pack(fill="x", pady=(12, 10))

        self.file_info = tk.Text(
            left, height=6, wrap="word", relief="flat", bg=INFO_BG, fg="#40516a",
            font=("Consolas", 9), padx=10, pady=8, highlightthickness=1,
            highlightbackground=BORDER, highlightcolor=BORDER, borderwidth=0
        )
        self.file_info.pack(fill="x", pady=(2, 0))
        self.file_info.insert("1.0", "Los datos técnicos del archivo aparecerán aquí.")
        self.file_info.configure(state="disabled")

        self.notebook = ttk.Notebook(right)
        self.notebook.pack(fill="both", expand=True)
        self.tab_signal = ttk.Frame(self.notebook, padding=8, style="Card.TFrame")
        self.tab_full_wave = ttk.Frame(self.notebook, padding=8, style="Card.TFrame")
        self.tab_fft = ttk.Frame(self.notebook, padding=8, style="Card.TFrame")
        self.tab_compare = ttk.Frame(self.notebook, padding=8, style="Card.TFrame")
        self.tab_conclusion = ttk.Frame(self.notebook, padding=10, style="Card.TFrame")
        self.notebook.add(self.tab_signal, text="Señal y ADC-DAC")
        self.notebook.add(self.tab_full_wave, text="Canción completa")
        self.notebook.add(self.tab_fft, text="Espectro FFT")
        self.notebook.add(self.tab_compare, text="Comparación de géneros")
        self.notebook.add(self.tab_conclusion, text="Conclusiones")

        # Contenedores para las figuras
        compare_paned = ttk.Panedwindow(self.tab_compare, orient="vertical")
        compare_paned.pack(fill="both", expand=True)
        compare_plot_frame = ttk.Frame(compare_paned, style="Card.TFrame")
        compare_table_frame = ttk.Frame(compare_paned, style="Card.TFrame")
        compare_paned.add(compare_plot_frame, weight=3)
        compare_paned.add(compare_table_frame, weight=2)

        compare_table_columns = ("genre", "dominant", "nyquist", "bits", "levels", "qerror", "dacerror")
        self.compare_tree = ttk.Treeview(compare_table_frame, columns=compare_table_columns, show="headings", height=6)
        headings = {"genre": "Género", "dominant": "Frecuencia dominante (Hz)", "nyquist": "Nyquist (Hz)", "bits": "Bits", "levels": "Niveles", "qerror": "RMSE cuantización", "dacerror": "RMSE DAC"}
        widths = {"genre": 130, "dominant": 145, "nyquist": 100, "bits": 50, "levels": 65, "qerror": 120, "dacerror": 100}
        for col in compare_table_columns:
            self.compare_tree.heading(col, text=headings[col])
            self.compare_tree.column(col, width=widths[col], anchor="center")
        self._stripe_tree(self.compare_tree)
        self.compare_tree.pack(fill="both", expand=True, pady=(6, 0))

        # Delegación del motor de gráficas a PlotManager
        self.plot_manager = PlotManager(
            tab_full_wave=self.tab_full_wave,
            tab_signal=self.tab_signal,
            tab_fft=self.tab_fft,
            compare_plot_frame=compare_plot_frame,
        )

        # Referencias de retrocompatibilidad
        self.fig_full_wave = self.plot_manager.fig_full_wave
        self.ax_full_wave = self.plot_manager.ax_full_wave
        self.canvas_full_wave = self.plot_manager.canvas_full_wave
        self.toolbar_full_wave = self.plot_manager.toolbar_full_wave

        self.fig_signal = self.plot_manager.fig_signal
        self.ax_original = self.plot_manager.ax_original
        self.ax_sampled = self.plot_manager.ax_sampled
        self.ax_quantized = self.plot_manager.ax_quantized
        self.ax_reconstructed = self.plot_manager.ax_reconstructed
        self.canvas_signal = self.plot_manager.canvas_signal
        self.toolbar_signal = self.plot_manager.toolbar_signal

        self.fig_fft = self.plot_manager.fig_fft
        self.ax_fft = self.plot_manager.ax_fft
        self.canvas_fft = self.plot_manager.canvas_fft
        self.toolbar_fft = self.plot_manager.toolbar_fft

        self.fig_compare = self.plot_manager.fig_compare
        self.ax_compare = self.plot_manager.ax_compare
        self.canvas_compare = self.plot_manager.canvas_compare
        self.toolbar_compare = self.plot_manager.toolbar_compare

        self.conclusion_text = tk.Text(
            self.tab_conclusion, wrap="word", relief="flat", bg="#ffffff", fg="#29384d",
            padx=18, pady=14, font=("Segoe UI", 10), spacing1=3, spacing3=5, borderwidth=0,
            highlightthickness=0
        )
        self.conclusion_text.pack(fill="both", expand=True)
        self.conclusion_text.tag_configure(
            "heading", font=("Segoe UI", 11, "bold"), foreground=PRIMARY_DARK,
            spacing1=10, spacing3=4, background="#eef4ff"
        )
        self.conclusion_text.configure(state="disabled")

    def _form_row(self, parent, row, label, widget):
        ttk.Label(parent, text=label.upper(), style="Field.TLabel").grid(row=row * 2, column=0, sticky="w", pady=(6, 2))
        widget.grid(row=row * 2 + 1, column=0, sticky="ew", pady=(0, 2))
        parent.grid_columnconfigure(0, weight=1)

    def _build_status_bar(self):
        tk.Frame(self.root, bg=BORDER, height=1).pack(fill="x")
        bottom = tk.Frame(self.root, bg=NAVY)
        bottom.pack(fill="x")
        self.status_var = tk.StringVar(value="Listo. Carga archivos WAV para comenzar.")
        tk.Label(bottom, text="●", bg=NAVY, fg="#22c55e", font=("Segoe UI", 9)).pack(side="left", padx=(18, 6), pady=7)
        tk.Label(bottom, textvariable=self.status_var, bg=NAVY, fg="#dbe6f5", font=("Segoe UI", 9)).pack(side="left", pady=7)
        tk.Label(
            bottom, text=f"Intervalo configurable · máximo {MAX_ANALYSIS_SECONDS:g} s por pista",
            bg=NAVY, fg="#8aa0bf", font=("Segoe UI", 9)
        ).pack(side="right", padx=18, pady=7)

    # ------------------------------------------------------------------
    # Delegación de trazado de gráficas
    # ------------------------------------------------------------------
    def _draw_empty_state(self, message="Carga un archivo WAV"):
        self.plot_manager.draw_empty_state(message)

    def _draw_full_song(self, track: AudioTrack):
        self.plot_manager.draw_full_song(track, self.analysis_start_seconds, self.analysis_end_seconds)

    def _draw_current(self):
        track = self._selected_track()
        result = self.analysis_results.get(track.slot)
        self.plot_manager.draw_current_analysis(
            track=track,
            result=result,
            start_seconds=self.analysis_start_seconds,
            end_seconds=self.analysis_end_seconds,
            fs_processing=self.fs_processing,
            bits=self.bits,
        )
        self._update_file_info(track)

    def _draw_comparison(self):
        self.plot_manager.draw_genre_comparison(
            tracks=self.tracks,
            analysis_results=self.analysis_results,
            fs_processing=self.fs_processing,
            compare_tree=self.compare_tree,
        )

    # ------------------------------------------------------------------
    # Lógica de gestión de pistas y audio
    # ------------------------------------------------------------------
    def _selected_track(self) -> AudioTrack:
        return self.tracks[self.current_slot - 1]

    def _refresh_track_table(self):
        selected = str(self.current_slot)
        for item in self.track_tree.get_children():
            self.track_tree.delete(item)
        for index, track in enumerate(self.tracks):
            if track.loaded:
                filename = os.path.basename(track.file_path or "")
                rate = str(track.sample_rate)
                duration = f"{track.duration:.1f} s" if track.duration is not None else "—"
            else:
                filename, rate, duration = "Sin cargar", "—", "—"
            label = track.genre
            self.track_tree.insert(
                "", "end", iid=str(track.slot),
                values=(track.slot, label, filename, rate, duration),
                tags=("odd" if index % 2 else "even",)
            )
        if self.track_tree.exists(selected):
            self.track_tree.selection_set(selected)
            self.track_tree.focus(selected)

    def _on_track_select(self, _event=None):
        selection = self.track_tree.selection()
        if not selection:
            return
        try:
            slot = int(selection[0])
        except ValueError:
            return
        if slot == self.current_slot and not self._changing_form:
            return
        self._stop_audio(update_status=False)
        self.current_slot = slot
        self._load_metadata_form(self._selected_track())
        self._draw_current()
        self._update_file_info(self._selected_track())

    def _load_metadata_form(self, track: AudioTrack):
        self._changing_form = True
        self.genre_var.set(track.genre)
        self.title_var.set(track.title)
        self.artist_var.set(track.artist)
        self.year_var.set(track.year)
        self.source_var.set(track.source)
        self._changing_form = False

    def _save_metadata(self):
        track = self._selected_track()
        genre = self.genre_var.get().strip()
        if genre:
            track.genre = genre
        track.title = self.title_var.get().strip()
        track.artist = self.artist_var.get().strip()
        track.year = self.year_var.get().strip()
        track.source = self.source_var.get().strip()
        self._refresh_track_table()
        self._update_file_info(track)
        self._draw_current()
        self._draw_comparison()
        self._update_conclusions()
        self.status_var.set(f"Ficha musical guardada para la pista {track.slot}.")

    def _read_time_interval(self):
        """Valida y devuelve el intervalo escrito en la interfaz, en segundos."""
        try:
            start = float(self.analysis_start_var.get().strip().replace(",", "."))
            end = float(self.analysis_end_var.get().strip().replace(",", "."))
        except (ValueError, AttributeError):
            messagebox.showwarning("Intervalo no válido", "Escribe el inicio y el fin en segundos, usando números válidos.")
            return None
        if not np.isfinite(start) or not np.isfinite(end) or start < 0 or end <= start:
            messagebox.showwarning("Intervalo no válido", "El intervalo debe cumplir: 0 ≤ inicio < fin.")
            return None
        if end - start > MAX_ANALYSIS_SECONDS + 1e-9:
            messagebox.showwarning(
                "Intervalo demasiado largo",
                f"Para conservar el comportamiento actual, cada análisis puede abarcar como máximo {MAX_ANALYSIS_SECONDS:g} segundos."
            )
            return None
        return start, end

    def _apply_time_range(self):
        """Aplica el intervalo temporal a las pistas cargadas y recalcula sus resultados."""
        interval = self._read_time_interval()
        if interval is None:
            return
        self.analysis_start_seconds, self.analysis_end_seconds = interval
        self._stop_audio(update_status=False)
        try:
            self._calculate_all()
            self._draw_current()
            self._draw_comparison()
            self._update_conclusions()
            self.status_var.set(
                f"Intervalo aplicado: {self.analysis_start_seconds:g}–{self.analysis_end_seconds:g} s. "
                "Los resultados corresponden a ese fragmento."
            )
        except Exception as exc:
            messagebox.showerror("No se pudo analizar el intervalo", str(exc))

    def _get_audio_backend(self):
        """Carga sounddevice solamente cuando se solicita reproducir audio."""
        if self._audio_backend is None:
            try:
                import sounddevice as sd
            except (ImportError, OSError):
                messagebox.showerror(
                    "Falta el componente de audio",
                    "Para reproducir audio instala la dependencia con:\n\npython -m pip install sounddevice"
                )
                return None
            self._audio_backend = sd
        return self._audio_backend

    def _play_full_audio(self):
        """Reproduce la canción completa de la pista seleccionada."""
        track = self._selected_track()
        if not track.loaded or track.samples is None:
            messagebox.showwarning("No hay audio", "Primero carga un archivo WAV en la pista seleccionada.")
            return
        backend = self._get_audio_backend()
        if backend is None:
            return
        try:
            backend.stop()
            audio = np.asarray(to_float_audio(track.samples), dtype=np.float32)
            backend.play(audio, samplerate=int(track.sample_rate))
            self.status_var.set(f"Reproduciendo canción completa: {track.title or os.path.basename(track.file_path or '')}.")
        except Exception as exc:
            messagebox.showerror("No se pudo reproducir el audio", f"Detalle: {exc}")

    def _play_selected_fragment(self):
        """Reproduce el fragmento definido en los controles de inicio y fin."""
        track = self._selected_track()
        if not track.loaded or track.samples is None:
            messagebox.showwarning("No hay audio", "Primero carga un archivo WAV en la pista seleccionada.")
            return
        interval = self._read_time_interval()
        if interval is None:
            return
        start, end = interval
        fs = int(track.sample_rate)
        start_index = int(round(start * fs))
        end_index = min(len(track.samples), int(round(end * fs)))
        if start_index >= len(track.samples) or end_index <= start_index:
            messagebox.showwarning(
                "Intervalo fuera del audio",
                f"La pista dura {track.duration:.3f} s. Ajusta el inicio y el fin a ese rango."
            )
            return
        backend = self._get_audio_backend()
        if backend is None:
            return
        try:
            backend.stop()
            audio = np.asarray(to_float_audio(track.samples[start_index:end_index]), dtype=np.float32)
            backend.play(audio, samplerate=fs)
            self.status_var.set(f"Reproduciendo fragmento: {start_index / fs:.3f}–{end_index / fs:.3f} s.")
        except Exception as exc:
            messagebox.showerror("No se pudo reproducir el fragmento", f"Detalle: {exc}")

    def _stop_audio(self, update_status=True):
        """Detiene la reproducción activa, si la hay."""
        if self._audio_backend is not None:
            try:
                self._audio_backend.stop()
            except Exception:
                pass
        if update_status and hasattr(self, "status_var"):
            self.status_var.set("Reproducción detenida.")

    def _on_close(self):
        self._stop_audio(update_status=False)
        self.root.destroy()

    def _load_wav(self):
        self._stop_audio(update_status=False)
        track = self._selected_track()
        path = filedialog.askopenfilename(
            title=f"Seleccionar WAV para la pista {track.slot}",
            filetypes=[("Archivos de audio WAV", "*.wav *.wave"), ("Todos los archivos", "*.*")],
        )
        if not path:
            return
        try:
            fs, raw, mono, channels, bit_depth = load_wav(path)
        except Exception as exc:
            messagebox.showerror("No se pudo cargar el WAV", f"El archivo no se pudo leer.\n\nDetalle: {exc}")
            return

        track.file_path = path
        track.sample_rate = fs
        track.samples = raw
        track.mono = mono
        track.channels = channels
        track.bit_depth = bit_depth
        track.duration = len(mono) / float(fs)
        track.title = filename_title(path)
        if not track.source:
            track.source = path
        self.analysis_results.pop(track.slot, None)
        self._load_metadata_form(track)
        self._refresh_track_table()
        self._update_file_info(track)
        try:
            self._calculate_all()
            self._draw_current()
            self._draw_comparison()
            self._update_conclusions()
            self.status_var.set(f"Cargado: {os.path.basename(path)} ({fs} Hz, {channels} canal(es)).")
        except Exception as exc:
            messagebox.showerror("Error al procesar el WAV", str(exc))

    def _remove_audio(self):
        self._stop_audio(update_status=False)
        track = self._selected_track()
        track.file_path = None
        track.sample_rate = None
        track.channels = None
        track.bit_depth = None
        track.duration = None
        track.samples = None
        track.mono = None
        track.title = ""
        track.artist = ""
        track.year = ""
        track.source = ""
        self.analysis_results.pop(track.slot, None)
        self._load_metadata_form(track)
        self._refresh_track_table()
        self._update_file_info(track)
        self._draw_current()
        self._draw_comparison()
        self._update_conclusions()
        self.status_var.set(f"Se quitó el audio de la pista {track.slot}.")

    def _apply_parameters(self):
        try:
            fs = int(self.fs_var.get().strip())
            bits = int(self.bits_var.get().strip())
            vmin = float(self.vmin_var.get().strip().replace(",", "."))
            vmax = float(self.vmax_var.get().strip().replace(",", "."))
            if fs < 1000 or fs > 192000:
                raise ValueError("La frecuencia de procesamiento debe estar entre 1000 y 192000 Hz.")
            if bits < 1 or bits > 24:
                raise ValueError("La cantidad de bits debe estar entre 1 y 24.")
            if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
                raise ValueError("El rango debe usar valores finitos y cumplir Vmax > Vmin.")
        except ValueError as exc:
            messagebox.showwarning("Parámetros no válidos", str(exc))
            return
        self.fs_processing, self.bits, self.vmin, self.vmax = fs, bits, vmin, vmax
        try:
            self._calculate_all()
            self._draw_current()
            self._draw_comparison()
            self._update_conclusions()
            self.status_var.set(f"Parámetros aplicados: fs={fs} Hz, {bits} bits, rango [{vmin:g}, {vmax:g}] V.")
        except Exception as exc:
            messagebox.showerror("No se pudo procesar", str(exc))

    def _calculate_all(self):
        new_results = {}
        for track in self.tracks:
            if not track.loaded:
                continue
            fs_original = int(track.sample_rate)
            start_index = int(round(self.analysis_start_seconds * fs_original))
            end_index = min(len(track.mono), int(round(self.analysis_end_seconds * fs_original)))
            if start_index >= len(track.mono) or end_index <= start_index:
                continue
            fragment = track.mono[start_index:end_index]
            result = analyze_audio(
                fragment, fs_original, self.fs_processing, self.bits, self.vmin, self.vmax,
                max_seconds=MAX_ANALYSIS_SECONDS,
            )
            result["analysis_start_seconds"] = start_index / fs_original
            result["analysis_end_seconds"] = end_index / fs_original
            new_results[track.slot] = result
        self.analysis_results = new_results

    def _update_file_info(self, track: AudioTrack):
        self.file_info.configure(state="normal")
        self.file_info.delete("1.0", "end")
        if not track.loaded:
            info = f"Pista {track.slot} · {track.genre}\n\nNo hay un archivo WAV cargado.\nUsa ‘Cargar / reemplazar WAV’ para seleccionar uno."
        else:
            info = (
                f"Pista {track.slot} · {track.genre}\n"
                f"Archivo: {os.path.basename(track.file_path or '')}\n"
                f"Frecuencia: {track.sample_rate} Hz\n"
                f"Canales: {track.channels}\n"
                f"Duración: {track.duration:.3f} s\n"
                f"Profundidad: {track.bit_depth if track.bit_depth else 'No disponible'} bits\n"
                f"Intervalo analizado: {self.analysis_start_seconds:.2f}–{self.analysis_end_seconds:.2f} s\n"
                f"Canales estandarizados a mono para el análisis."
            )
        self.file_info.insert("1.0", info)
        self.file_info.configure(state="disabled")

    def _update_conclusions(self):
        lines = [
            "LECTURA TÉCNICA DE LOS RESULTADOS",
            "La frecuencia dominante corresponde al pico más alto distinto de DC dentro del fragmento procesado. No representa necesariamente una nota musical única: un audio mezcla voz, instrumentos, ritmo, armónicos y ruido.",
            "",
            f"Intervalo seleccionado: {self.analysis_start_seconds:g}–{self.analysis_end_seconds:g} s (hasta {MAX_ANALYSIS_SECONDS:g} s por pista).",
            f"Configuración vigente: fs = {self.fs_processing} Hz; Nyquist = {self.fs_processing / 2:g} Hz; ADC = {self.bits} bits ({2 ** self.bits} niveles); rango = [{self.vmin:g}, {self.vmax:g}] V; ΔV = {(self.vmax-self.vmin)/(2**self.bits):.6g} V.",
            "El ADC mapea la amplitud normalizada al rango indicado y aplica cuantización uniforme mid-rise. El DAC se representa mediante retención de orden cero (ZOH). Ambas son simulaciones numéricas; el audio original no se modifica.",
            "",
            "RESULTADOS POR PISTA",
        ]
        any_loaded = False
        for track in self.tracks:
            result = self.analysis_results.get(track.slot)
            if not track.loaded:
                lines.append(f"• Pista {track.slot} — {track.genre}: falta cargar un WAV.")
                continue
            if result is None:
                lines.append(
                    f"• Pista {track.slot} — {track.genre}: el intervalo seleccionado "
                    f"({self.analysis_start_seconds:g}–{self.analysis_end_seconds:g} s) queda fuera de la duración del audio."
                )
                continue
            any_loaded = True
            lines.append(
                f"• Pista {track.slot} — {track.genre}: {track.title or os.path.basename(track.file_path or '')}; "
                f"pico dominante ≈ {result['dominant_frequency']:.2f} Hz; "
                f"RMSE de cuantización = {result['adc']['rmse_normalized']:.5f}; "
                f"RMSE de reconstrucción = {result['reconstruction_rmse']:.5f}."
            )
            if result["fs_processing"] < result["fs_original"] and result["energy_above_nyquist_pct"] > 5:
                lines.append(f"  Atención: aproximadamente {result['energy_above_nyquist_pct']:.1f}% de la energía del espectro original queda por encima del Nyquist configurado; parte del contenido de alta frecuencia no estará disponible tras el remuestreo.")
        if not any_loaded:
            if any(track.loaded for track in self.tracks):
                lines.append("No hay resultados para el intervalo seleccionado; revisa que el rango exista dentro de la duración de los audios cargados.")
            else:
                lines.append("Carga archivos WAV en el panel izquierdo para generar resultados técnicos.")
        lines += [
            "",
            "MÚSICA, CONCENTRACIÓN Y TDAH: INTERPRETACIÓN PRUDENTE",
            "Esta aplicación no recomienda automáticamente un género ni diagnostica o trata el TDAH. La FFT describe propiedades acústicas, pero no mide atención, comprensión, rendimiento académico o síntomas clínicos.",
            "Las revisiones consultadas describen resultados prometedores en algunos contextos, pero también señalan heterogeneidad entre estudios, intervenciones y participantes. La evidencia no justifica elegir un género universalmente superior para todas las personas con TDAH.",
            "Para una conclusión académica, separa: (1) los datos medidos del fragmento —espectro, pico dominante, amplitud y error ADC/DAC—; y (2) el análisis documental, que debe considerar letra, volumen, preferencias individuales, tarea y diseño de cada estudio. Una prueba personal de concentración, por sí sola, tampoco demuestra un efecto clínico.",
            "",
            "REFERENCIAS DOCUMENTALES",
            "1. Exploring the Intersection of ADHD and Music: A Systematic Review (2025). PubMed: https://pubmed.ncbi.nlm.nih.gov/39851869/",
            "2. Outcomes of music therapy on children and adolescents with ADHD: a systematic review and meta-analysis (2025). PubMed: https://pubmed.ncbi.nlm.nih.gov/40680190/",
            "3. Effects of Music on ADHD and Potential Application in Serious Video Games: Systematic Review (2023). PubMed: https://pubmed.ncbi.nlm.nih.gov/37171837/",
        ]
        self.conclusion_text.configure(state="normal")
        self.conclusion_text.delete("1.0", "end")
        self.conclusion_text.insert("1.0", "\n\n".join(lines))
        for heading in (
            "LECTURA TÉCNICA DE LOS RESULTADOS",
            "RESULTADOS POR PISTA",
            "MÚSICA, CONCENTRACIÓN Y TDAH: INTERPRETACIÓN PRUDENTE",
            "REFERENCIAS DOCUMENTALES",
        ):
            idx = self.conclusion_text.search(heading, "1.0", stopindex="end")
            if idx:
                self.conclusion_text.tag_add("heading", idx, f"{idx} lineend")
        self.conclusion_text.configure(state="disabled")

    def _export_csv(self):
        path = filedialog.asksaveasfilename(
            title="Guardar tabla de resultados", defaultextension=".csv",
            filetypes=[("CSV", "*.csv")], initialfile="resumen_dsp.csv"
        )
        if not path:
            return
        try:
            export_csv(path, self.tracks, self.analysis_results)
            self.status_var.set(f"Tabla CSV guardada: {path}")
            messagebox.showinfo("Exportación terminada", "Se guardó el resumen en CSV.")
        except Exception as exc:
            messagebox.showerror("Error al exportar CSV", str(exc))

    def _export_txt(self):
        path = filedialog.asksaveasfilename(
            title="Guardar informe de análisis", defaultextension=".txt",
            filetypes=[("Texto", "*.txt")], initialfile="informe_analisis_dsp.txt"
        )
        if not path:
            return
        try:
            export_text_report(path, self.tracks, self.analysis_results, self.fs_processing, self.bits, self.vmin, self.vmax)
            self.status_var.set(f"Informe TXT guardado: {path}")
            messagebox.showinfo("Exportación terminada", "Se guardó el informe TXT con resultados y referencias.")
        except Exception as exc:
            messagebox.showerror("Error al exportar informe", str(exc))
