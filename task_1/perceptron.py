
import random


class RosenblattPerceptron:
    def __init__(self, feature_count=25, learning_rate=0.1):
        if not 0 < learning_rate < 1:
            raise ValueError("Швидкість навчання повинна бути між 0 і 1.")

        self.weights = [
            random.uniform(0.01, 0.99)
            for _ in range(feature_count)
        ]
        self.threshold = random.uniform(0.01, 0.99)
        self.learning_rate = learning_rate
        self.step = 0

    def weighted_sum(self, features):
        if len(features) != len(self.weights):
            raise ValueError("Неправильна кількість ознак.")

        return sum(
            weight * value
            for weight, value in zip(self.weights, features)
        )

    def predict(self, features):
        signal = self.weighted_sum(features)
        return 1 if signal > self.threshold else -1

    def train_sample(self, features, desired):
        if desired not in (-1, 1):
            raise ValueError("Бажаний вихід повинен дорівнювати -1 або 1.")

        signal = self.weighted_sum(features)
        predicted = 1 if signal > self.threshold else -1
        error = desired - predicted

        if error != 0:
            self.weights = [
                weight + self.learning_rate * error * value
                for weight, value in zip(self.weights, features)
            ]

        self.step += 1

        return {
            "step": self.step,
            "desired": desired,
            "predicted": predicted,
            "correct": error == 0,
            "signal": signal,
            "error": error,
            "weights": self.weights.copy(),
            "threshold": self.threshold,
        }

    def train_until_convergence(self, samples, max_epochs=100):
        if not samples:
            raise ValueError("Навчальна послідовність порожня.")

        history = []

        for epoch in range(1, max_epochs + 1):
            errors = 0

            for features, desired in samples:
                result = self.train_sample(features, desired)

                if not result["correct"]:
                    errors += 1

            history.append({
                "epoch": epoch,
                "errors": errors,
            })

            if errors == 0:
                break

        return {
            "converged": history[-1]["errors"] == 0,
            "epochs": len(history),
            "history": history,
        }
