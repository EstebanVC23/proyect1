"""Módulo de reproducción de audio para AudioLab DSP.

Usa pygame.mixer para reproducir WAV y MP3 sin conversión previa.
Soporta reproducción, pausa, reanudación y detención.
"""
from __future__ import annotations

import threading
from pathlib import Path
from typing import Optional, Callable

# pygame se importa de forma diferida para no bloquear si no está disponible.
_pygame_ok = False
try:
    import pygame
    _pygame_ok = True
except ImportError:
    pass

_lock = threading.Lock()
_current_path: Optional[str] = None
_on_finish_cb: Optional[Callable] = None
_MIXER_READY = False


def _ensure_mixer() -> bool:
    """Inicializa pygame.mixer si todavía no está listo."""
    global _MIXER_READY
    if not _pygame_ok:
        return False
    with _lock:
        if _MIXER_READY:
            return True
        try:
            if not pygame.get_init():
                pygame.init()
            if not pygame.mixer.get_init():
                pygame.mixer.init(frequency=44100, size=-16, channels=2, buffer=2048)
            _MIXER_READY = True
            return True
        except Exception:
            return False


def is_available() -> bool:
    """True si pygame.mixer está disponible."""
    return _ensure_mixer()


def play(path: str, on_finish: Optional[Callable] = None) -> bool:
    """Reproduce el archivo de audio (WAV o MP3).

    Args:
        path: Ruta al archivo de audio.
        on_finish: Callback opcional a ejecutar cuando termina la reproducción.

    Returns:
        True si se inició la reproducción, False si hubo un error.
    """
    global _current_path, _on_finish_cb
    if not _ensure_mixer():
        return False
    try:
        stop()  # Detiene cualquier reproducción activa
        pygame.mixer.music.load(path)
        pygame.mixer.music.play()
        _current_path = path
        _on_finish_cb = on_finish
        if on_finish:
            # Monitorea el fin en un hilo daemon
            threading.Thread(target=_watch_finish, daemon=True).start()
        return True
    except Exception as exc:
        print(f"[AudioPlayer] Error al reproducir '{path}': {exc}")
        return False


def _watch_finish():
    """Hilo que espera a que termine la reproducción y llama al callback."""
    if not _pygame_ok:
        return
    import time
    while pygame.mixer.music.get_busy():
        time.sleep(0.1)
    cb = _on_finish_cb
    if cb:
        try:
            cb()
        except Exception:
            pass


def pause() -> None:
    """Pausa la reproducción activa."""
    if _ensure_mixer():
        pygame.mixer.music.pause()


def resume() -> None:
    """Reanuda la reproducción pausada."""
    if _ensure_mixer():
        pygame.mixer.music.unpause()


def stop() -> None:
    """Detiene la reproducción."""
    global _current_path, _on_finish_cb
    if _ensure_mixer():
        try:
            pygame.mixer.music.stop()
        except Exception:
            pass
    _current_path = None
    _on_finish_cb = None


def is_playing() -> bool:
    """True si hay audio reproduciéndose en este momento."""
    if not _ensure_mixer():
        return False
    return bool(pygame.mixer.music.get_busy())


def is_paused() -> bool:
    """True si la reproducción está en pausa (heurístico: mixer init pero no busy)."""
    if not _ensure_mixer():
        return False
    # pygame no expone directamente un estado "pausado"; lo inferimos desde el path
    return _current_path is not None and not pygame.mixer.music.get_busy()


def get_position_ms() -> int:
    """Posición actual de reproducción en milisegundos."""
    if not _ensure_mixer():
        return 0
    return pygame.mixer.music.get_pos()


def set_volume(level: float) -> None:
    """Ajusta el volumen (0.0 – 1.0)."""
    if _ensure_mixer():
        pygame.mixer.music.set_volume(max(0.0, min(1.0, float(level))))


def cleanup() -> None:
    """Libera recursos de pygame.mixer. Llamar al cerrar la aplicación."""
    global _MIXER_READY
    stop()
    if _pygame_ok and _MIXER_READY:
        try:
            pygame.mixer.quit()
        except Exception:
            pass
        _MIXER_READY = False

