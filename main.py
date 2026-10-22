"""Punto de entrada de AudioLab DSP."""
import tkinter as tk
from tkinter import messagebox


def main() -> None:
    try:
        from ui.app import AudioDSPApp
        root = tk.Tk()
        AudioDSPApp(root)
        root.mainloop()
    except ImportError as exc:
        message = (
            "No se encontraron algunas dependencias. Instálalas con:\n\n"
            "python -m pip install -r requirements.txt\n\n"
            f"Detalle: {exc}"
        )
        try:
            messagebox.showerror("Dependencias faltantes", message)
        except tk.TclError:
            print(message)
        raise


if __name__ == "__main__":
    main()
