
class HopfieldNetwork:
    def __init__(self):
        self.patterns = {}
        self.weights = []
        self.size = 0

    def train(self, patterns):
        if not patterns:
            raise ValueError("Необхідно додати еталонні образи.")

        self.size = len(next(iter(patterns.values())))

        for vector in patterns.values():
            if len(vector) != self.size:
                raise ValueError("Розміри векторів повинні збігатися.")

            if any(value not in (-1, 1) for value in vector):
                raise ValueError("Ознаки повинні мати значення -1 або 1.")

        self.patterns = {
            str(name): list(vector)
            for name, vector in patterns.items()
        }

        self.weights = [
            [
                sum(
                    vector[i] * vector[j]
                    for vector in self.patterns.values()
                ) if i != j else 0
                for j in range(self.size)
            ]
            for i in range(self.size)
        ]

        return self.weights

    def recognize(self, vector, max_iterations=100):
        if not self.weights:
            raise ValueError("Спочатку потрібно навчити мережу.")

        if len(vector) != self.size:
            raise ValueError("Неправильна кількість ознак.")

        if any(value not in (-1, 1) for value in vector):
            raise ValueError("Вектор повинен містити лише -1 та 1.")

        state = list(vector)
        history = [state.copy()]
        converged = False

        for _ in range(max_iterations):
            changed = False

            for j in range(self.size):
                signal = sum(
                    self.weights[j][i] * state[i]
                    for i in range(self.size)
                )

                new_value = (
                    1 if signal > 0
                    else -1 if signal < 0
                    else state[j]
                )

                if new_value != state[j]:
                    state[j] = new_value
                    changed = True

            history.append(state.copy())

            if not changed:
                converged = True
                break

        distances = {
            name: sum(
                a != b for a, b in zip(state, pattern)
            )
            for name, pattern in self.patterns.items()
        }

        matches = [
            name
            for name, pattern in self.patterns.items()
            if state == pattern
        ]

        predicted = matches[0] if len(matches) == 1 else None

        return {
            "input": list(vector),
            "output": state,
            "predicted": predicted,
            "iterations": len(history) - 1,
            "converged": converged,
            "history": history,
            "distances": distances,
        }
