"""Interfaz Tkinter para la aplicación Señales DSP.

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

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

from dsp.audio_io import load_wav, filename_title, to_float_audio
from dsp.models import AudioTrack
from dsp.processing import analyze_audio, MAX_ANALYSIS_SECONDS
from dsp.reporting import export_csv, export_text_report
from ui.plots import PlotManager, style_axes, CARD, BORDER, MUTED


GENRES = ["Clásica / barroca", "Electrónica", "Salsa", "Música de cantina", "Reguetón"]
DEFAULT_GENRES = ["Clásica / barroca", "Electrónica", "Salsa", "Música de cantina", "Reguetón"]
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
    """Controlador y ventana principal de la aplicación Señales DSP."""

    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Señales DSP | ADC · DAC · FFT")
        self.root.geometry("1450x900")
        self.root.minsize(1120, 720)
        self.root.configure(bg=BG)

        self.fs_processing = 22050
        self.bits = 8
        self.vmin = -5.0
        self.vmax = 5.0
        self.analysis_results: dict[int, dict] = {}
        self.analysis_start_seconds = 0.0
        self.analysis_end_seconds = 6.0
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
        self._load_default_tracks_if_available()
        self._load_metadata_form(self.tracks[0])
        if self.tracks[0].loaded:
            self._draw_current()
        else:
            self._draw_empty_state()
            self._draw_full_song(self.tracks[0])
        self._draw_comparison()
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
        tk.Label(inner, text="Señales DPS", bg=NAVY, fg="#ffffff", font=("Segoe UI", 20, "bold")).pack(side="left")
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
        self.bits_combo = ttk.Combobox(panel, textvariable=self.bits_var, values=BIT_OPTIONS, width=7, state="readonly")
        self.bits_combo.grid(row=1, column=1, sticky="ew", padx=(0, 14), pady=(4, 0))
        self.bits_combo.bind("<<ComboboxSelected>>", self._on_bits_selected)
        self.fs_combo.bind("<<ComboboxSelected>>", self._on_fs_selected)
        ttk.Label(panel, text="V MÍNIMO (V)", style="Field.TLabel").grid(row=0, column=2, sticky="w", padx=(0, 7))
        self.vmin_entry = ttk.Entry(panel, textvariable=self.vmin_var, width=9)
        self.vmin_entry.grid(row=1, column=2, sticky="ew", padx=(0, 14), pady=(4, 0))
        self.vmin_entry.bind("<Return>", lambda e: self._apply_parameters())
        ttk.Label(panel, text="V MÁXIMO (V)", style="Field.TLabel").grid(row=0, column=3, sticky="w", padx=(0, 7))
        self.vmax_entry = ttk.Entry(panel, textvariable=self.vmax_var, width=9)
        self.vmax_entry.grid(row=1, column=3, sticky="ew", padx=(0, 14), pady=(4, 0))
        self.vmax_entry.bind("<Return>", lambda e: self._apply_parameters())
        ttk.Button(panel, text="Aplicar y recalcular", style="Accent.TButton", command=self._apply_parameters).grid(row=1, column=4, padx=(0, 10), pady=(4, 0))
        ttk.Button(panel, text="Exportar CSV", command=self._export_csv).grid(row=1, column=5, padx=(0, 8), pady=(4, 0))
        ttk.Button(panel, text="Guardar informe TXT", command=self._export_txt).grid(row=1, column=6, pady=(4, 0))
        for col in range(7):
            panel.grid_columnconfigure(col, weight=0)
        panel.grid_columnconfigure(0, weight=1)

        # Controles de reproducción y selección del intervalo de la canción a analizar.
        self.analysis_start_var = tk.StringVar(value="0")
        self.analysis_end_var = tk.StringVar(value="6")

        ttk.Separator(panel).grid(row=2, column=0, columnspan=7, sticky="ew", pady=(12, 0))
        interval_bar = ttk.Frame(panel, style="Card.TFrame")
        interval_bar.grid(row=3, column=0, columnspan=7, sticky="ew", pady=(10, 0))
        ttk.Label(interval_bar, text="Intervalo del audio (s):", style="Card.TLabel", font=("Segoe UI", 9, "bold")).pack(side="left", padx=(0, 10))
        ttk.Label(interval_bar, text="Inicio", style="Card.TLabel").pack(side="left", padx=(0, 4))
        self.start_entry = ttk.Entry(interval_bar, textvariable=self.analysis_start_var, width=8)
        self.start_entry.pack(side="left", padx=(0, 10))
        self.start_entry.bind("<Return>", lambda e: self._apply_time_range())
        ttk.Label(interval_bar, text="Fin", style="Card.TLabel").pack(side="left", padx=(0, 4))
        self.end_entry = ttk.Entry(interval_bar, textvariable=self.analysis_end_var, width=8)
        self.end_entry.pack(side="left", padx=(0, 8))
        self.end_entry.bind("<Return>", lambda e: self._apply_time_range())
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

        compare_table_columns = ("genre", "dominant", "nyquist", "bits", "levels")
        self.compare_tree = ttk.Treeview(compare_table_frame, columns=compare_table_columns, show="headings", height=6)
        headings = {"genre": "Género", "dominant": "Frecuencia dominante (Hz)", "nyquist": "Nyquist (Hz)", "bits": "Bits", "levels": "Niveles"}
        widths = {"genre": 130, "dominant": 145, "nyquist": 100, "bits": 50, "levels": 65}
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

        # Panel superior: histograma de bandas de frecuencia comparativo entre pistas
        self.fig_conclusion = Figure(figsize=(9.5, 3.2), dpi=90, constrained_layout=True, facecolor=CARD)
        self.ax_conclusion_hist = self.fig_conclusion.add_subplot(111)
        self.canvas_conclusion = FigureCanvasTkAgg(self.fig_conclusion, master=self.tab_conclusion)
        self.canvas_conclusion.get_tk_widget().configure(bg=CARD, highlightthickness=0)
        self.canvas_conclusion.get_tk_widget().pack(fill="x", pady=(0, 4))

        # Panel inferior: contenedor con scrollbar para el informe especializado de TDAH y DSP
        text_container = ttk.Frame(self.tab_conclusion, style="Card.TFrame")
        text_container.pack(fill="both", expand=True)

        scrollbar_conclusion = ttk.Scrollbar(text_container, orient="vertical")
        scrollbar_conclusion.pack(side="right", fill="y")

        self.conclusion_text = tk.Text(
            text_container, wrap="word", relief="flat", bg="#ffffff", fg="#29384d",
            padx=16, pady=10, font=("Segoe UI", 9), spacing1=2, spacing3=3, borderwidth=0,
            highlightthickness=1, highlightbackground=BORDER,
            yscrollcommand=scrollbar_conclusion.set
        )
        self.conclusion_text.pack(side="left", fill="both", expand=True)
        scrollbar_conclusion.config(command=self.conclusion_text.yview)

        self.conclusion_text.tag_configure(
            "heading", font=("Segoe UI", 10, "bold"), foreground=PRIMARY_DARK,
            spacing1=8, spacing3=3, background="#eef4ff"
        )
        self.conclusion_text.tag_configure(
            "subheading", font=("Segoe UI", 9, "bold"), foreground="#1e3a5f",
            spacing1=6, spacing3=2
        )
        self.conclusion_text.tag_configure(
            "winner", font=("Segoe UI", 9, "bold"), foreground="#15803d", background="#f0fdf4"
        )
        self.conclusion_text.tag_configure(
            "runnerup", font=("Segoe UI", 9, "bold"), foreground="#ea580c", background="#fff7ed"
        )
        self.conclusion_text.tag_configure(
            "bad", font=("Segoe UI", 9, "bold"), foreground="#dc2626"
        )
        self.conclusion_text.tag_configure(
            "limit", font=("Segoe UI", 9, "bold"), foreground="#b45309"
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
            if fs <= 0:
                raise ValueError("La frecuencia de procesamiento debe ser un número entero positivo mayor que cero.")
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

    # Las 5 bandas acústicas normalizadas para el análisis comparativo (Hz)
    _FREQ_BANDS = [
        ("Subgraves\n20-60 Hz",       20.0,   60.0),
        ("Graves\n60-250 Hz",         60.0,  250.0),
        ("Medios\n250-2k Hz",        250.0, 2000.0),
        ("Agudos\n2k-6k Hz",        2000.0, 6000.0),
        ("Muy agudos\n6k-20k Hz",   6000.0, 20000.0),
    ]
    # Paleta de colores por pista (igual que PlotManager)
    _SLOT_COLORS = ["#2563eb", "#ea7a1a", "#16a34a", "#8b5cf6", "#dc2626"]

    def _compute_band_energy(self, magnitudes: np.ndarray, frequencies: np.ndarray) -> list:
        """Calcula el porcentaje de energía en cada una de las 5 bandas acústicas obligatorias.
        
        Respeta los límites reales de frecuencia disponibles (hasta Nyquist o 20 kHz)
        y normaliza respecto a la energía audible total contenida en las 5 bandas (20 Hz a 20 000 Hz).
        """
        energies = []
        for _, f_lo, f_hi in self._FREQ_BANDS:
            mask = (frequencies >= f_lo) & (frequencies < (f_hi if f_hi is not None else np.inf))
            e = float(np.sum(np.square(magnitudes[mask])))
            energies.append(e)
        total = sum(energies)
        if total <= 0:
            return [0.0] * len(self._FREQ_BANDS)
        return [100.0 * e / total for e in energies]

    def _on_bits_selected(self, _event=None):
        """Aplica y recalcula inmediatamente cuando el usuario cambia la resolución en bits."""
        self._apply_parameters()

    def _on_fs_selected(self, _event=None):
        """Aplica y recalcula inmediatamente cuando el usuario cambia la frecuencia de muestreo."""
        self._apply_parameters()

    def _load_default_tracks_if_available(self):
        """Carga automáticamente las canciones del proyecto si existen en data/audio."""
        default_files = [
            (1, "data/audio/capricho_24.wav", "Capricho No. 24", "Niccolò Paganini", "1802-1817", "Violín solo"),
            (2, "data/audio/Alan-Walker-Faded.wav", "Fade", "Alan Walker", "2015", "NCS / Electrónica"),
            (3, "data/audio/gotas_de_lluvia.wav", "Gotas de Lluvia", "Grupo Niche", "1995", "Salsa"),
            (4, "data/audio/repitela.wav", "Repítela", "Jessi Uribe", "2015", "Música de cantina"),
            (5, "data/audio/diva_virtual.wav", "Diva Virtual", "Don Omar", "2009", "Reguetón"),
        ]
        loaded_any = False
        for slot, rel_path, title, artist, year, source in default_files:
            abs_path = os.path.abspath(rel_path)
            if not os.path.isfile(abs_path):
                continue
            track = self.tracks[slot - 1]
            if track.loaded:
                continue
            try:
                fs, raw, mono, channels, bit_depth = load_wav(abs_path)
                track.file_path = abs_path
                track.sample_rate = fs
                track.samples = raw
                track.mono = mono
                track.channels = channels
                track.bit_depth = bit_depth
                track.duration = len(mono) / float(fs)
                track.title = title
                track.artist = artist
                track.year = year
                track.source = source
                loaded_any = True
            except Exception:
                pass
        if loaded_any:
            self._calculate_all()
            self._refresh_track_table()

    def _classify_dominant_band(self, dom_f: float) -> str:
        """Determina a cuál de las 5 bandas acústicas pertenece la frecuencia dominante."""
        if dom_f < 20.0:
            return "Infrasónico / Residual (<20 Hz)"
        elif dom_f < 60.0:
            return "Subgraves (20–60 Hz)"
        elif dom_f < 250.0:
            return "Graves (60–250 Hz)"
        elif dom_f < 2000.0:
            return "Medios (250–2.000 Hz)"
        elif dom_f < 6000.0:
            return "Agudos (2.000–6.000 Hz)"
        else:
            return "Muy agudos (6.000–20.000 Hz)"

    def _update_conclusions(self):
        """Actualiza el histograma de 5 bandas y genera el informe interpretativo 100% dinámico sobre TDAH."""
        ax = self.ax_conclusion_hist
        ax.clear()

        loaded_results = [
            (track, self.analysis_results[track.slot])
            for track in self.tracks
            if track.loaded and track.slot in self.analysis_results
        ]

        band_labels = [b[0] for b in self._FREQ_BANDS]
        n_bands = len(band_labels)
        n_tracks = len(loaded_results)

        if n_tracks == 0:
            ax.text(0.5, 0.5, "Carga archivos WAV para ver el histograma de bandas acústicas",
                    ha="center", va="center", transform=ax.transAxes, color="#68788f", fontsize=11)
            ax.set_axis_off()
        else:
            bar_width = 0.8 / max(n_tracks, 1)
            x = np.arange(n_bands)
            for i, (track, result) in enumerate(loaded_results):
                energies = self._compute_band_energy(result["magnitudes"], result["frequencies"])
                offset = (i - (n_tracks - 1) / 2.0) * bar_width
                color = self._SLOT_COLORS[track.slot - 1] if track.slot - 1 < len(self._SLOT_COLORS) else "#555"
                ax.bar(
                    x + offset, energies, width=bar_width * 0.9,
                    color=color, alpha=0.85,
                    label=f"{track.genre} ({track.title[:16] + '…' if len(track.title) > 16 else track.title or 'WAV'})"
                )
            ax.set_xticks(x)
            ax.set_xticklabels(band_labels, fontsize=8)
            ax.set_ylabel("Energía relativa en banda (%)", fontsize=9)
            ax.set_title(
                f"Distribución de energía espectral en 5 bandas acústicas "
                f"[intervalo {self.analysis_start_seconds:g}–{self.analysis_end_seconds:g} s | fs={self.fs_processing} Hz]",
                fontsize=10, fontweight="bold", color="#14243a"
            )
            ax.set_ylim(0, 100)
            ax.legend(fontsize=8, loc="upper right", framealpha=0.92)
            style_axes(ax)

        self.canvas_conclusion.draw_idle()

        # ── Generación dinámica del informe simplificado, comprensible y directo ──
        delta_v_mv = (self.vmax - self.vmin) / (2 ** self.bits) * 1000.0
        lines = [
            "════════════════════════════════════════════════════════════════════════════════",
            "  CONCLUSIONES: ANÁLISIS ESPECTRAL DE FRECUENCIAS Y ESTUDIO CON TDAH",
            "════════════════════════════════════════════════════════════════════════════════",
            "",
            "PARÁMETROS VIGENTES DEL ANÁLISIS:",
            f"• Frecuencia de muestreo: fs = {self.fs_processing} Hz (Límite Nyquist = {self.fs_processing / 2:g} Hz)",
            f"• Resolución ADC: {self.bits} bits ({2 ** self.bits} niveles cuánticos) | Paso de voltaje ΔV = {delta_v_mv:.2f} mV",
            f"• Rango de voltaje configurado: [{self.vmin:g}, {self.vmax:g}] V",
            f"• Intervalo de tiempo procesado: {self.analysis_start_seconds:g} a {self.analysis_end_seconds:g} s ({self.analysis_end_seconds - self.analysis_start_seconds:g} s evaluados)",
            "",
            "────────────────────────────────────────────────────────────────────────────────",
            "1. RESUMEN: ¿QUÉ CANCIÓN ES MÁS ADECUADA PARA ESTUDIAR CON TDAH?",
            "────────────────────────────────────────────────────────────────────────────────",
            "No existe una 'frecuencia mágica' universal, pero según la distribución real de frecuencias:",
            "",
        ]

        if not loaded_results:
            lines.append("No hay pistas cargadas en el intervalo actual. Carga archivos WAV para ver las mediciones.")
            self._render_conclusion_lines(lines)
            return

        track_metrics = []
        for track, result in loaded_results:
            energies = self._compute_band_energy(result["magnitudes"], result["frequencies"])
            probs = [e / 100.0 for e in energies]
            ent = -sum(p * np.log(p) for p in probs if p > 0)
            entropy_norm = float(ent / np.log(len(self._FREQ_BANDS))) if len(self._FREQ_BANDS) > 1 else 0.0

            dom_f = float(result["dominant_frequency"])
            dom_band = self._classify_dominant_band(dom_f)

            max_pct = max(energies)
            max_idx = energies.index(max_pct)
            band_labels = ["Subgraves", "Graves", "Medios", "Agudos", "Muy agudos"]

            track_metrics.append({
                "slot": track.slot,
                "genre": track.genre or f"Pista {track.slot}",
                "title": track.title or os.path.basename(track.file_path or ""),
                "energies": energies,
                "entropy_norm": entropy_norm,
                "dom_f": dom_f,
                "dom_band": dom_band,
                "max_band": band_labels[max_idx],
                "max_pct": max_pct,
            })

        # Evaluación dinámica de los candidatos según los valores reales medidos en este intervalo:
        best_focus = max(
            track_metrics,
            key=lambda m: (m["energies"][1] + m["energies"][2]) - m["energies"][0] - (m["energies"][3] + m["energies"][4])
        )
        focus_mid_grav = best_focus["energies"][1] + best_focus["energies"][2]
        focus_high = best_focus["energies"][3] + best_focus["energies"][4]
        focus_sub = best_focus["energies"][0]

        best_mask = max(track_metrics, key=lambda m: m["energies"][0] + m["energies"][1])
        mask_low = best_mask["energies"][0] + best_mask["energies"][1]

        max_high = max(track_metrics, key=lambda m: m["energies"][3])
        max_mid = max(track_metrics, key=lambda m: m["energies"][2])

        lines += [
            f"🥇 MEJOR OPCIÓN PARA CONCENTRACIÓN Y LECTURA: {best_focus['genre']} ('{best_focus['title']}')",
            f"• ¿Por qué? El {focus_mid_grav:.1f}% de su energía en este fragmento está concentrada entre Graves ({best_focus['energies'][1]:.1f}%) y Medios ({best_focus['energies'][2]:.1f}%).",
            f"• Ventaja clave medida: Casi nula presencia en agudos punzantes ({focus_high:.1f}%) y mínima vibración subgrave ({focus_sub:.1f}%). Tono dominante en {best_focus['dom_f']:.1f} Hz.",
            f"• Efecto acústico: Al no presentar componentes agudos chillones ni sacudidas subgraves, el cerebro no dispara reflejos de alerta auditiva involuntaria.",
            "",
            f"🥈 BUENA OPCIÓN PARA TRABAJO CONTINUO O ENMASCARAMIENTO: {best_mask['genre']} ('{best_mask['title']}')",
            f"• ¿Por qué? El {mask_low:.1f}% de su energía medida se concentra en frecuencias bajas (Subgraves {best_mask['energies'][0]:.1f}% + Graves {best_mask['energies'][1]:.1f}%).",
            f"• Ventaja clave medida: Distribución espectral extendida (equilibrio H_norm = {best_mask['entropy_norm']:.3f}) y tono dominante en {best_mask['dom_f']:.1f} Hz.",
            f"• Efecto acústico: Actúa como suelo sonoro continuo de bajas frecuencias (análogo al ruido marrón), útil para amortiguar ruidos imprevisibles del ambiente que desconcentran a personas con TDAH.",
            "",
            f"⚠️ MENOS RECOMENDADAS PARA ESTUDIO PASIVO: {max_high['genre']} ('{max_high['title']}') y {max_mid['genre']} ('{max_mid['title']}')",
            f"• ¿Por qué? {max_high['genre']} registra un {max_high['energies'][3]:.1f}% de energía en Agudos (2k-6k Hz por metales/percusión), y {max_mid['genre']} concentra un {max_mid['energies'][2]:.1f}% en Medios.",
            f"• Efecto acústico: Los armónicos brillantes y punzantes llaman la atención auditiva consciente de forma involuntaria, interrumpiendo el hilo cognitivo de estudio.",
            "",
            "────────────────────────────────────────────────────────────────────────────────",
            "2. TABLA DE MEDICIONES REALES EN LAS 5 BANDAS ACÚSTICAS (FFT)",
            "────────────────────────────────────────────────────────────────────────────────",
        ]

        lines.append(f"{'Género / Canción':<22} | {'Sub (20-60)':<11} | {'Grav (60-250)':<12} | {'Med (250-2k)':<11} | {'Agu (2k-6k)':<11} | {'MAgu (6k-20k)':<12} | {'Pico Dom.':<10}")
        lines.append("-" * 105)
        for m in track_metrics:
            e = m["energies"]
            title_str = f"{m['genre'][:12]} ({m['title'][:6]})"
            lines.append(
                f"{title_str:<22} | {e[0]:8.1f} %  | {e[1]:8.1f} %   | {e[2]:8.1f} %  | {e[3]:8.1f} %  | {e[4]:8.1f} %   | {m['dom_f']:6.1f} Hz"
            )

        lines += [
            "",
            "────────────────────────────────────────────────────────────────────────────────",
            "3. RESPUESTAS CLARAS A LAS PREGUNTAS DEL ANÁLISIS",
            "────────────────────────────────────────────────────────────────────────────────",
            "",
            "A. ¿Qué canciones tienen una distribución más equilibrada?",
        ]
        sorted_entropy = sorted(track_metrics, key=lambda x: x["entropy_norm"], reverse=True)
        if len(sorted_entropy) >= 2:
            m_top = sorted_entropy[0]
            m_bot = sorted_entropy[-1]
            lines += [
                f"• La más equilibrada: {m_top['genre']} ('{m_top['title']}')",
                f"  -> Reparte su sonido en varias bandas a la vez: Subgraves ({m_top['energies'][0]:.1f}%), Graves ({m_top['energies'][1]:.1f}%) y Medios ({m_top['energies'][2]:.1f}%).",
                f"• La menos equilibrada (más concentrada): {m_bot['genre']} ('{m_bot['title']}')",
                f"  -> Concentra casi todo en un solo sector ({m_bot['max_band']} con {m_bot['max_pct']:.1f}%), dejando el resto de bandas con muy baja presencia.",
            ]

        lines += [
            "",
            "B. ¿Dónde concentra cada canción su mayor cantidad de energía?",
        ]
        for b_name, b_idx in [("Subgraves (20–60 Hz)", 0), ("Graves (60–250 Hz)", 1), ("Medios (250–2.000 Hz)", 2), ("Agudos (2.000–6.000 Hz)", 3)]:
            top = max(track_metrics, key=lambda x: x["energies"][b_idx])
            lines.append(f"• {b_name}: La mayor presencia está en {top['genre']} con {top['energies'][b_idx]:.1f}%.")

        lines += [
            "",
            "C. ¿Qué diferencias físicas reales hay entre los géneros?",
            "• Música acústica vs. Música electrónica/urbana:",
            f"  - En Subgraves (<60 Hz): Los instrumentos acústicos reales (violín, guitarra, voz) apenas generan energía en este rango (<1%) debido a sus límites físicos de resonancia. En cambio, pistas electrónicas o urbanas como {best_mask['genre']} registran hasta un {best_mask['energies'][0]:.1f}% por síntesis de bombos y sub-bajos.",
            f"  - En Agudos (2.000–6.000 Hz): Géneros como {max_high['genre']} alcanzan un {max_high['energies'][3]:.1f}% por metales y percusión brillante, mientras que piezas acústicas de cuerdas como {best_focus['genre']} decaen rápidamente ({best_focus['energies'][3]:.1f}%), ofreciendo un sonido más cálido y menos fatigante.",
            "",
            "D. ¿Qué nos indica la frecuencia dominante?",
            "Es el tono o golpe de mayor fuerza en la canción:",
        ]
        for m in track_metrics:
            lines.append(f"• {m['genre']}: {m['dom_f']:.1f} Hz (ubicado en {m['dom_band']}).")

        lines += [
            "",
            "────────────────────────────────────────────────────────────────────────────────",
            "4. LO QUE SE PUEDE Y NO SE PUEDE AFIRMAR (DIFERENCIACIÓN Y LÍMITES)",
            "────────────────────────────────────────────────────────────────────────────────",
            "1. Lo que SÍ demuestran los datos de la app:",
            "   - La potencia real de cada frecuencia calculada mediante la transformada de Fourier (FFT).",
            "   - Cómo cambia la fidelidad de la señal al modificar los bits (paso de tensión ΔV) y la tasa de muestreo.",
            "",
            "2. La hipótesis para el estudio:",
            "   - Canciones con energía suave en graves/medios (Clásica) o con base continua (Electrónica) son mejores candidatas para evitar sobresaltos atencionales.",
            ""
        ]

        self._render_conclusion_lines(lines)

    def _render_conclusion_lines(self, lines: list):
        """Renderiza las líneas del informe en el Text widget y aplica los estilos visuales."""
        self.conclusion_text.configure(state="normal")
        self.conclusion_text.delete("1.0", "end")
        self.conclusion_text.insert("1.0", "\n".join(lines))

        for heading in (
            "1. RESUMEN: ¿QUÉ CANCIÓN ES MÁS ADECUADA PARA ESTUDIAR CON TDAH?",
            "2. TABLA DE MEDICIONES EN LAS 5 BANDAS ACÚSTICAS (FFT)",
            "3. RESPUESTAS CLARAS A LAS PREGUNTAS DEL ANÁLISIS",
            "4. LO QUE SE PUEDE Y NO SE PUEDE AFIRMAR (DIFERENCIACIÓN Y LÍMITES)",
        ):
            idx = self.conclusion_text.search(heading, "1.0", stopindex="end")
            if idx:
                self.conclusion_text.tag_add("heading", idx, f"{idx} lineend")

        for subheading in (
            "A. ¿Qué canciones tienen una distribución más equilibrada?",
            "B. ¿Dónde concentra cada canción su mayor cantidad de energía?",
            "C. ¿Qué diferencias físicas reales hay entre los géneros?",
            "D. ¿Qué nos indica la frecuencia dominante?",
        ):
            idx = self.conclusion_text.search(subheading, "1.0", stopindex="end")
            if idx:
                self.conclusion_text.tag_add("subheading", idx, f"{idx} lineend")

        for winner in ("🥇 MEJOR OPCIÓN PARA CONCENTRACIÓN Y LECTURA", "🥈 BUENA OPCIÓN PARA TRABAJO CONTINUO O ENMASCARAMIENTO"):
            idx = self.conclusion_text.search(winner, "1.0", stopindex="end")
            if idx:
                self.conclusion_text.tag_add("winner", idx, f"{idx} lineend")

        for bad in ("⚠️ MENOS RECOMENDADAS PARA ESTUDIO PASIVO",):
            idx = self.conclusion_text.search(bad, "1.0", stopindex="end")
            if idx:
                self.conclusion_text.tag_add("bad", idx, f"{idx} lineend")

        for limit in ("1. Lo que SÍ demuestran los datos de la app:", "2. La hipótesis para el estudio:"):
            idx = self.conclusion_text.search(limit, "1.0", stopindex="end")
            if idx:
                self.conclusion_text.tag_add("limit", idx, f"{idx} lineend")

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
