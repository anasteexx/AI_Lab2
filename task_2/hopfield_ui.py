
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from tkinter.scrolledtext import ScrolledText

from PIL import Image, ImageTk

from common.features import extract_features
from task_2.hopfield import HopfieldNetwork


class HopfieldTab(ttk.Frame):
    def __init__(self, parent):
        super().__init__(parent, padding=15)

        self.network = HopfieldNetwork()
        self.references = {}
        self.unknown = None
        self.images = {}
        self.result = None

        self.columnconfigure(1, weight=1)
        self.rowconfigure(0, weight=1)

        container = ttk.Frame(self)
        container.grid(row=0, column=0, sticky="nsew", padx=(0, 15))
        container.rowconfigure(0, weight=1)
        container.columnconfigure(0, weight=1)

        canvas = tk.Canvas(
            container, width=370, highlightthickness=0
        )
        canvas.grid(row=0, column=0, sticky="nsew")

        scrollbar = ttk.Scrollbar(
            container, orient="vertical", command=canvas.yview
        )
        scrollbar.grid(row=0, column=1, sticky="ns")
        canvas.configure(yscrollcommand=scrollbar.set)

        left = ttk.Frame(canvas)
        window_id = canvas.create_window(
            (0, 0), window=left, anchor="nw"
        )

        left.bind(
            "<Configure>",
            lambda event: canvas.configure(
                scrollregion=canvas.bbox("all")
            )
        )

        canvas.bind(
            "<Configure>",
            lambda event: canvas.itemconfigure(
                window_id, width=event.width
            )
        )

        ttk.Label(
            left,
            text="Мережа Хопфілда",
            font=("Segoe UI", 17, "bold")
        ).pack(anchor="w", pady=(0, 15))

        for digit in range(3):
            group = ttk.LabelFrame(
                left, text=f"Еталонний клас {digit}", padding=10
            )
            group.pack(fill="x", pady=(0, 10))

            label = ttk.Label(
                group, text="Зображення не вибрано", anchor="center"
            )
            label.pack(pady=5)

            self.images[f"label_{digit}"] = label

            ttk.Button(
                group,
                text=f"Завантажити цифру {digit}",
                command=lambda d=digit: self.load_reference(d)
            ).pack(fill="x")

        ttk.Button(
            left,
            text="Сформувати матрицю ваг",
            command=self.train_network
        ).pack(fill="x", pady=(5, 15))

        unknown_group = ttk.LabelFrame(
            left, text="Невідоме зображення", padding=10
        )
        unknown_group.pack(fill="x", pady=(0, 10))

        self.unknown_label = ttk.Label(
            unknown_group,
            text="Зображення не вибрано",
            anchor="center"
        )
        self.unknown_label.pack(pady=5)

        ttk.Button(
            unknown_group,
            text="Завантажити BMP",
            command=self.load_unknown
        ).pack(fill="x")

        ttk.Button(
            left,
            text="Розпізнати зображення",
            command=self.recognize
        ).pack(fill="x", pady=10)

        right = ttk.Frame(self)
        right.grid(row=0, column=1, sticky="nsew")
        right.columnconfigure(0, weight=1)
        right.rowconfigure(1, weight=1)

        ttk.Label(
            right,
            text="Параметри та результати мережі",
            font=("Segoe UI", 16, "bold")
        ).grid(row=0, column=0, sticky="w", pady=(0, 12))

        self.output = ScrolledText(
            right, font=("Consolas", 10), wrap="word"
        )
        self.output.grid(row=1, column=0, sticky="nsew")

        self.status = tk.StringVar(value="Еталонів завантажено: 0 із 3")

        ttk.Label(
            right, textvariable=self.status
        ).grid(row=2, column=0, sticky="w", pady=10)

        self.update_output()

    def display_image(self, path, label, key):
        with Image.open(path) as source:
            image = source.convert("RGB")
            image.thumbnail((160, 160))
            photo = ImageTk.PhotoImage(image)

        self.images[key] = photo
        label.configure(image=photo, text="")

    def load_reference(self, digit):
        path = filedialog.askopenfilename(
            title=f"Еталон цифри {digit}",
            filetypes=[("BMP-зображення", "*.bmp")]
        )

        if not path:
            return

        try:
            absolute, normalized, bipolar = extract_features(path)

            self.display_image(
                path, self.images[f"label_{digit}"], f"image_{digit}"
            )

            self.references[str(digit)] = {
                "absolute": absolute,
                "normalized": normalized,
                "bipolar": bipolar,
            }

            self.network = HopfieldNetwork()
            self.result = None
            self.update_output()

        except (ValueError, OSError) as error:
            messagebox.showerror("Помилка", str(error))

    def load_unknown(self):
        path = filedialog.askopenfilename(
            title="Невідомий образ",
            filetypes=[("BMP-зображення", "*.bmp")]
        )

        if not path:
            return

        try:
            absolute, normalized, bipolar = extract_features(path)

            self.display_image(
                path, self.unknown_label, "unknown_image"
            )

            self.unknown = {
                "absolute": absolute,
                "normalized": normalized,
                "bipolar": bipolar,
            }

            self.result = None
            self.update_output()

        except (ValueError, OSError) as error:
            messagebox.showerror("Помилка", str(error))

    def train_network(self):
        if len(self.references) != 3:
            messagebox.showwarning(
                "Увага", "Завантажте еталони всіх трьох класів."
            )
            return

        patterns = {
            name: data["bipolar"]
            for name, data in self.references.items()
        }

        try:
            self.network.train(patterns)
            self.result = None
            self.update_output()

        except ValueError as error:
            messagebox.showerror("Помилка", str(error))

    def recognize(self):
        if not self.network.weights:
            messagebox.showwarning(
                "Увага", "Спочатку сформуйте матрицю ваг."
            )
            return

        if self.unknown is None:
            messagebox.showwarning(
                "Увага", "Завантажте невідоме зображення."
            )
            return

        try:
            self.result = self.network.recognize(
                self.unknown["bipolar"]
            )
            self.update_output()

        except ValueError as error:
            messagebox.showerror("Помилка", str(error))

    def update_output(self):
        lines = [
            "МЕРЕЖА ХОПФІЛДА",
            "Кількість нейронів: 25",
            f"Кількість еталонів: {len(self.references)}",
            "",
        ]

        for name, data in sorted(self.references.items()):
            lines.extend([
                f"ЕТАЛОННИЙ КЛАС {name}",
                "Абсолютний вектор:",
                str(data["absolute"]),
                "Біполярний вектор:",
                str(data["bipolar"]),
                "",
            ])

        if self.network.weights:
            lines.append("МАТРИЦЯ ВАГ (25 x 25)")
            lines.append("")

            for row in self.network.weights:
                lines.append(
                    " ".join(f"{value:3d}" for value in row)
                )

            lines.append("")

        if self.unknown is not None:
            lines.extend([
                "НЕВІДОМИЙ ОБРАЗ",
                "Абсолютний вектор:",
                str(self.unknown["absolute"]),
                "Біполярний вектор:",
                str(self.unknown["bipolar"]),
                "",
            ])

        if self.result is not None:
            result = self.result

            lines.extend([
                "РЕЗУЛЬТАТ РОЗПІЗНАВАННЯ",
                f"Кількість ітерацій: {result['iterations']}",
                f"Стабілізація: {'Так' if result['converged'] else 'Ні'}",
                "",
                "Вихідний вектор:",
                str(result["output"]),
                "",
            ])

            if result["predicted"] is not None:
                lines.append(
                    f"Розпізнаний клас: {result['predicted']}"
                )
            else:
                lines.append(
                    "Точного збігу з еталонами не знайдено."
                )

            lines.extend([
                "",
                "ВІДСТАНІ ХЕМІНГА ДО ЕТАЛОНІВ",
            ])

            for name, distance in result["distances"].items():
                lines.append(f"Клас {name}: {distance}")

            lines.extend([
                "",
                "ІСТОРІЯ ІТЕРАЦІЙ",
            ])

            for step, vector in enumerate(result["history"]):
                lines.append(
                    f"Ітерація {step}: {vector}"
                )

        self.output.configure(state="normal")
        self.output.delete("1.0", "end")
        self.output.insert("1.0", "\n".join(lines))
        self.output.configure(state="disabled")

        self.status.set(
            f"Еталонів завантажено: {len(self.references)} із 3 | "
            f"Мережа навчена: {'Так' if self.network.weights else 'Ні'}"
        )
