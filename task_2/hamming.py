
class HammingNetwork:
    def __init__(self, epsilon=0.2):
        self.epsilon = epsilon
        self.patterns = {}
        self.weights = {}
        self.bias = 0
        self.size = 0

    def train(self, patterns):
        if len(patterns) < 2:
            raise ValueError("Потрібно щонайменше два еталони.")

        if not 0 < self.epsilon < 1 / len(patterns):
            raise ValueError("Неправильний коефіцієнт гальмування.")

        size = len(next(iter(patterns.values())))

        for vector in patterns.values():
            if len(vector) != size:
                raise ValueError("Розміри векторів не збігаються.")

            if any(value not in (-1, 1) for value in vector):
                raise ValueError("Ознаки повинні дорівнювати -1 або 1.")

        self.size = size
        self.patterns = {
            str(name): list(vector)
            for name, vector in patterns.items()
        }

        self.weights = {
            name: [value / 2 for value in vector]
            for name, vector in self.patterns.items()
        }

        self.bias = size / 2

        return self.weights

    def recognize(self, vector, max_iterations=500):
        if not self.weights:
            raise ValueError("Спочатку потрібно навчити мережу.")

        if len(vector) != self.size:
            raise ValueError("Неправильна кількість ознак.")

        if any(value not in (-1, 1) for value in vector):
            raise ValueError("Вектор повинен містити лише -1 та 1.")

        names = list(self.patterns)

        initial = [
            sum(
                weight * value
                for weight, value in zip(self.weights[name], vector)
            ) + self.bias
            for name in names
        ]

        outputs = initial.copy()
        history = [outputs.copy()]
        converged = False

        for _ in range(max_iterations):
            new_outputs = []

            for i, value in enumerate(outputs):
                inhibition = self.epsilon * sum(
                    outputs[j]
                    for j in range(len(outputs))
                    if j != i
                )

                new_outputs.append(max(0.0, value - inhibition))

            history.append(new_outputs.copy())

            if all(
                abs(a - b) < 1e-9
                for a, b in zip(outputs, new_outputs)
            ):
                converged = True
                outputs = new_outputs
                break

            outputs = new_outputs

        maximum = max(outputs)

        winners = [
            names[i]
            for i, value in enumerate(outputs)
            if maximum > 1e-9 and abs(value - maximum) < 1e-9
        ]

        predicted = (
            winners[0]
            if converged and len(winners) == 1
            else None
        )

        distances = {
            name: sum(
                a != b
                for a, b in zip(vector, pattern)
            )
            for name, pattern in self.patterns.items()
        }

        return {
            "predicted": predicted,
            "initial": dict(zip(names, initial)),
            "outputs": dict(zip(names, outputs)),
            "distances": distances,
            "iterations": len(history) - 1,
            "converged": converged,
            "history": history,
        }
