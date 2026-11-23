
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from tkinter.scrolledtext import ScrolledText

from PIL import Image, ImageTk

from common.features import extract_features
from task_1.perceptron import RosenblattPerceptron


class PerceptronTab(ttk.Frame):
    def __init__(self, parent):
        super().__init__(parent, padding=15)

        self.model = RosenblattPerceptron()
        self.samples = []
        self.current_features = None
        self.current_absolute = None
        self.current_path = None
        self.preview_image = None
        self.history = []

        self.class_var = tk.IntVar(value=-1)
        self.rate_var = tk.StringVar(value="0.1")
        self.threshold_var = tk.StringVar(
            value=f"{self.model.threshold:.6f}"
        )
        self.status_var = tk.StringVar()

        self.columnconfigure(1, weight=1)
        self.rowconfigure(0, weight=1)


        left_container = ttk.Frame(self)
        left_container.grid(
            row=0, column=0, sticky="nsew", padx=(0, 20)
        )

        left_container.rowconfigure(0, weight=1)
        left_container.columnconfigure(0, weight=1)

        canvas = tk.Canvas(
            left_container,
            width=370,
            highlightthickness=0
        )
        canvas.grid(row=0, column=0, sticky="nsew")

        scrollbar = ttk.Scrollbar(
            left_container,
            orient="vertical",
            command=canvas.yview
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


        right = ttk.Frame(self)
        right.grid(row=0, column=1, sticky="nsew")
        right.columnconfigure(0, weight=1)
        right.rowconfigure(1, weight=1)

        ttk.Label(
            left,
            text="Перцептрон Розенблата",
            font=("Segoe UI", 17, "bold")
        ).pack(anchor="w", pady=(0, 15))

        image_group = ttk.LabelFrame(
            left, text="Графічний образ", padding=12
        )
        image_group.pack(fill="x", pady=(0, 12))

        self.image_label = ttk.Label(
            image_group,
            text="Зображення не вибрано",
            anchor="center",
            width=32
        )
        self.image_label.pack(pady=10)

        ttk.Button(
            image_group,
            text="Завантажити BMP",
            command=self.load_image
        ).pack(fill="x")

        class_group = ttk.LabelFrame(
            left, text="Навчання з учителем", padding=12
        )
        class_group.pack(fill="x", pady=(0, 12))

        ttk.Label(
            class_group,
            text="Правильний клас зображення:"
        ).pack(anchor="w")

        ttk.Radiobutton(
            class_group,
            text="Клас 0 (−1)",
            variable=self.class_var,
            value=-1
        ).pack(anchor="w", pady=(8, 3))

        ttk.Radiobutton(
            class_group,
            text="Клас 1 (+1)",
            variable=self.class_var,
            value=1
        ).pack(anchor="w")

        ttk.Button(
            class_group,
            text="Підтвердити / скоригувати",
            command=self.train_sample
        ).pack(fill="x", pady=(12, 5))

        ttk.Button(
            class_group,
            text="Навчити до збіжності",
            command=self.train_all
        ).pack(fill="x")

        parameters = ttk.LabelFrame(
            left, text="Параметри перцептрона", padding=12
        )
        parameters.pack(fill="x")

        ttk.Label(
            parameters, text="Швидкість навчання:"
        ).pack(anchor="w")

        ttk.Entry(
            parameters, textvariable=self.rate_var
        ).pack(fill="x", pady=(4, 10))

        ttk.Label(
            parameters, text="Поріг активації:"
        ).pack(anchor="w")

        ttk.Entry(
            parameters, textvariable=self.threshold_var
        ).pack(fill="x", pady=(4, 10))

        ttk.Label(
            parameters, text="Початкові ваги (25 значень):"
        ).pack(anchor="w")

        self.weights_entry = ScrolledText(
            parameters, height=3, width=30, wrap="word"
        )
        self.weights_entry.pack(fill="x", pady=(4, 10))
        self.fill_weights()

        ttk.Button(
            parameters,
            text="Встановити параметри",
            command=self.apply_parameters
        ).pack(fill="x", pady=(0, 5))

        ttk.Button(
            parameters,
            text="Нові випадкові ваги",
            command=self.reset_model
        ).pack(fill="x")

        ttk.Label(
            right,
            text="Стан і результати роботи мережі",
            font=("Segoe UI", 16, "bold")
        ).grid(row=0, column=0, sticky="w", pady=(0, 12))

        self.output = ScrolledText(
            right, font=("Consolas", 10), wrap="word"
        )
        self.output.grid(row=1, column=0, sticky="nsew")

        ttk.Label(
            right, textvariable=self.status_var
        ).grid(row=2, column=0, sticky="w", pady=10)

        self.show_state()

    def fill_weights(self):
        self.weights_entry.delete("1.0", "end")
        self.weights_entry.insert(
            "1.0",
            ", ".join(f"{weight:.6f}" for weight in self.model.weights)
        )

    def load_image(self):
        path = filedialog.askopenfilename(
            title="Виберіть зображення",
            filetypes=[("BMP-зображення", "*.bmp")]
        )

        if not path:
            return

        try:
            absolute, normalized, _ = extract_features(path)

            with Image.open(path) as image:
                preview = image.convert("RGB")
                preview.thumbnail((220, 220))
                self.preview_image = ImageTk.PhotoImage(preview)

            self.image_label.configure(
                image=self.preview_image, text=""
            )

            self.current_absolute = absolute
            self.current_features = normalized
            self.current_path = path

            predicted = self.model.predict(normalized)
            self.history.append(
                f"Завантажено: {path}\nПоточний вихід: {predicted}"
            )
            self.show_state()

        except (OSError, ValueError) as error:
            messagebox.showerror("Помилка", str(error))

    def apply_parameters(self):
        try:
            rate = float(self.rate_var.get())
            threshold = float(self.threshold_var.get())

            weights = [
                float(value.strip())
                for value in self.weights_entry.get(
                    "1.0", "end"
                ).replace("\n", ",").split(",")
                if value.strip()
            ]

            if not 0 < rate < 1:
                raise ValueError(
                    "Швидкість навчання має бути між 0 і 1."
                )

            if not 0 < threshold < 1:
                raise ValueError(
                    "Поріг активації має бути між 0 і 1."
                )

            if len(weights) != 25:
                raise ValueError("Потрібно ввести рівно 25 ваг.")

            if any(not 0 < weight < 1 for weight in weights):
                raise ValueError(
                    "Початкові ваги мають бути між 0 і 1."
                )

            self.model = RosenblattPerceptron(
                learning_rate=rate
            )
            self.model.weights = weights
            self.model.threshold = threshold

            self.samples.clear()
            self.history.clear()
            self.show_state()

        except ValueError as error:
            messagebox.showerror("Помилка", str(error))

    def reset_model(self):
        try:
            rate = float(self.rate_var.get())
            self.model = RosenblattPerceptron(
                learning_rate=rate
            )
        except ValueError as error:
            messagebox.showerror("Помилка", str(error))
            return

        self.threshold_var.set(f"{self.model.threshold:.6f}")
        self.fill_weights()
        self.samples.clear()
        self.history.clear()
        self.show_state()

    def train_sample(self):
        if self.current_features is None:
            messagebox.showwarning(
                "Увага", "Спочатку завантажте зображення."
            )
            return

        desired = self.class_var.get()
        result = self.model.train_sample(
            self.current_features, desired
        )

        self.samples.append(
            (self.current_features.copy(), desired)
        )

        decision = (
            "Рішення підтверджено"
            if result["correct"]
            else "Рішення спростовано, ваги скориговано"
        )

        self.history.append(
            f"Крок {result['step']}: "
            f"Y={result['predicted']}, D={desired}\n"
            f"{decision}; помилка={result['error']}"
        )

        self.show_state()

    def train_all(self):
        if not self.samples:
            messagebox.showwarning(
                "Увага", "Спочатку додайте навчальні зразки."
            )
            return

        result = self.model.train_until_convergence(
            self.samples, max_epochs=500
        )

        errors = [
            f"Епоха {item['epoch']}: "
            f"{item['errors']} помилок"
            for item in result["history"]
        ]

        conclusion = (
            "Навчання збіглося"
            if result["converged"]
            else "Збіжності за 500 епох не досягнуто"
        )

        self.history.append(
            f"{conclusion}\n"
            f"Кількість епох: {result['epochs']}\n"
            + "\n".join(errors)
        )
        self.show_state()

    def show_state(self):
        lines = [
            "ПАРАМЕТРИ МЕРЕЖІ",
            f"Крок навчання: {self.model.step}",
            f"Швидкість: {self.model.learning_rate}",
            f"Поріг: {self.model.threshold:.6f}",
            f"Навчальних зразків: {len(self.samples)}",
            "",
            "СИНАПТИЧНІ ВАГИ",
        ]

        lines.extend(
            f"w{i + 1} = {weight:.6f}"
            for i, weight in enumerate(self.model.weights)
        )

        if self.current_features is not None:
            lines.extend([
                "",
                "АБСОЛЮТНИЙ ВЕКТОР",
                str(self.current_absolute),
                "",
                "НОРМОВАНИЙ ВЕКТОР",
                str([round(x, 6) for x in self.current_features]),
                "",
                "КЛАСИФІКАЦІЯ",
                f"Y = ({self.model.predict(self.current_features)})",
            ])

        if self.history:
            lines.extend([
                "",
                "ІСТОРІЯ НАВЧАННЯ",
                "\n\n".join(self.history),
            ])

        self.output.configure(state="normal")
        self.output.delete("1.0", "end")
        self.output.insert("1.0", "\n".join(lines))
        self.output.configure(state="disabled")

        self.status_var.set(
            f"Поточний крок: {self.model.step} | "
            f"Навчальних зразків: {len(self.samples)}"
        )
