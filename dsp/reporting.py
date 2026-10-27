"""Exportación de resultados en CSV y TXT."""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, List

from .models import AudioTrack


def summary_rows(tracks: List[AudioTrack], analyses: Dict[int, dict]) -> list[dict]:
    rows = []
    for track in tracks:
        result = analyses.get(track.slot)
        rows.append({
            "slot": track.slot,
            "genero": track.genre,
            "titulo": track.title,
            "interprete_compositor": track.artist,
            "ano": track.year,
            "archivo": track.file_path or "",
            "fuente": track.source,
            "fs_original_Hz": track.sample_rate or "",
            "canales": track.channels or "",
            "profundidad_bits_original": track.bit_depth or "",
            "duracion_s": round(track.duration, 6) if track.duration is not None else "",
            "fs_procesamiento_Hz": result["fs_processing"] if result else "",
            "Nyquist_Hz": result["nyquist"] if result else "",
            "frecuencia_dominante_Hz": result["dominant_frequency"] if result else "",
            "bits_ADC": result["adc"]["bits"] if result else "",
            "niveles_cuantizacion": result["adc"]["levels"] if result else "",
            "paso_cuantizacion_V": result["adc"]["delta_v"] if result else "",
            "RMSE_cuantizacion_normalizada": result["adc"]["rmse_normalized"] if result else "",
            "RMSE_reconstruccion_normalizada": result["reconstruction_rmse"] if result else "",
            "energia_original_sobre_Nyquist_pct": result["energy_above_nyquist_pct"] if result else "",
        })
    return rows


def export_csv(path: str, tracks: List[AudioTrack], analyses: Dict[int, dict]) -> None:
    rows = summary_rows(tracks, analyses)
    if not rows:
        raise ValueError("No hay pistas para exportar.")
    with open(path, "w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def export_text_report(path: str, tracks: List[AudioTrack], analyses: Dict[int, dict],
                       fs_processing: int, bits: int, vmin: float, vmax: float) -> None:
    lines = [
        "INFORME DE ANÁLISIS DSP DE AUDIO: ADC Y DAC",
        "=" * 48,
        "",
        "Parámetros de procesamiento",
        f"Frecuencia de muestreo de procesamiento: {fs_processing} Hz",
        f"Frecuencia de Nyquist: {fs_processing / 2:.2f} Hz",
        f"Resolución ADC: {bits} bits ({2 ** bits} niveles)",
        f"Rango de entrada: [{vmin:g}, {vmax:g}] V",
        f"Paso de cuantización uniforme: {(vmax-vmin)/(2**bits):.8g} V",
        "Convención: cuantizador uniforme mid-rise; ΔV=(Vmax−Vmin)/2^B.",
        "Reconstrucción DAC de referencia: retención de orden cero (ZOH).",
        "Intervalo de análisis configurable según el rango seleccionado en la aplicación.",
        "",
    ]
    loaded = [track for track in tracks if track.loaded]
    if not loaded:
        lines.append("No se cargaron archivos WAV; todavía no hay resultados numéricos.")
    for track in tracks:
        lines += ["-" * 48, f"Pista {track.slot}: {track.genre}",
                  f"Título: {track.title or '(sin título)'}",
                  f"Compositor/intérprete: {track.artist or '(sin dato)'}",
                  f"Año: {track.year or '(sin dato)'}",
                  f"Fuente: {track.source or track.file_path or '(sin dato)'}"]
        if not track.loaded:
            lines.append("Estado: no se cargó un WAV.")
            lines.append("")
            continue
        result = analyses.get(track.slot)
        lines += [f"Archivo: {track.file_path}",
                  f"Frecuencia original: {track.sample_rate} Hz",
                  f"Canales: {track.channels}",
                  f"Profundidad original: {track.bit_depth if track.bit_depth else 'no disponible'} bits",
                  f"Duración total: {track.duration:.4f} s"]
        if result:
            adc = result["adc"]
            snr_str = f"{adc['snr_quantization_db']:.2f} dB" if adc['snr_quantization_db'] != float('inf') else "∞ (sin error medible)"
            lines += [f"Duración analizada: {result['analysis_duration']:.4f} s",
                      f"Frecuencia de procesamiento: {result['fs_processing']} Hz",
                      f"Nyquist: {result['nyquist']:.2f} Hz",
                      f"Frecuencia dominante estimada por FFT: {result['dominant_frequency']:.3f} Hz",
                      f"Niveles ADC: {adc['levels']}",
                      f"Paso de cuantización ΔV: {adc['delta_v']:.8g} V",
                      f"Factor de escala: {adc['scale_v_per_unit']:.8g} V por unidad normalizada",
                      f"Relación señal/ruido de cuantización (SNR): {snr_str}",
                      f"RMSE de cuantización (escala normalizada): {adc['rmse_normalized']:.8g}",
                      f"RMSE de cuantización (voltios): {adc['rmse_volts']:.8g} V",
                      f"Error máximo de cuantización: {adc['max_error_volts']:.8g} V",
                      f"RMSE de reconstrucción DAC (escala normalizada): {result['reconstruction_rmse']:.8g}",
                      f"Energía espectral original estimada sobre Nyquist: {result['energy_above_nyquist_pct']:.3f}%"]
            if result["fs_processing"] < result["fs_original"]:
                lines.append("Nota: se reduce la frecuencia de muestreo. El contenido por encima de Nyquist no se conserva al remuestrear.")
        lines.append("")
    lines += [
        "CONCLUSIONES Y LIMITACIONES",
        "- La FFT caracteriza el contenido frecuencial del fragmento analizado; no identifica por sí sola el efecto de la música en la concentración.",
        "- La frecuencia dominante es el pico espectral más alto distinto de DC; no equivale necesariamente a la nota musical principal ni al ritmo.",
        "- La reconstrucción dibujada es una aproximación mediante retención de orden cero, no una salida analógica física.",
        "- No se puede concluir que un género sea terapéutico o mejore el TDAH únicamente a partir de sus frecuencias. La respuesta puede variar entre personas y tareas.",
        "- Para estudiar concentración se necesitarían medidas conductuales controladas y revisar la evidencia clínica/documental.",
        "",
        "FUENTES DE REFERENCIA SOBRE MÚSICA Y TDAH",
        "1. Exploring the Intersection of ADHD and Music: A Systematic Review (2025).",
        "   https://pubmed.ncbi.nlm.nih.gov/39851869/",
        "2. Outcomes of music therapy on children and adolescents with ADHD: systematic review and meta-analysis (2025).",
        "   https://pubmed.ncbi.nlm.nih.gov/40680190/",
        "3. Effects of Music on ADHD and Potential Application in Serious Video Games: Systematic Review (2023).",
        "   https://pubmed.ncbi.nlm.nih.gov/37171837/",
        "",
    ]
    Path(path).write_text("\n".join(lines), encoding="utf-8")
