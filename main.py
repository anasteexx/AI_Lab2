
import tkinter as tk
from tkinter import ttk

from task_1.perceptron_ui import PerceptronTab

from task_2.hopfield_ui import HopfieldTab

from task_2.hamming_ui import HammingTab


def main():
    root = tk.Tk()
    root.title("Лабораторна робота №2 — Нейронні мережі")
    root.geometry("1200x800")
    root.minsize(950, 650)

    style = ttk.Style()
    style.theme_use("clam")
    style.configure(
        "TNotebook.Tab",
        font=("Segoe UI", 11),
        padding=(18, 12)
    )

    notebook = ttk.Notebook(root)
    notebook.pack(fill="both", expand=True, padx=15, pady=15)

    perceptron_tab = PerceptronTab(notebook)
    notebook.add(
        perceptron_tab,
        text="Перцептрон Розенблата"
    )

    hopfield_tab = HopfieldTab(notebook)
    notebook.add(hopfield_tab, text="Мережа Хопфілда")

    hamming_tab = HammingTab(notebook)
    notebook.add(hamming_tab, text="Мережа Хемінга")

    root.mainloop()


if __name__ == "__main__":
    main()
