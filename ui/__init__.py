"""Interfaz gráfica de la aplicación Mira / Señales DSP."""

from .app import AudioDSPApp
from .plots import PlotManager, style_axes, pack_plot

__all__ = ["AudioDSPApp", "PlotManager", "style_axes", "pack_plot"]
