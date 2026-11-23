
import unittest

from task_1.perceptron import RosenblattPerceptron


class TestPerceptron(unittest.TestCase):
    def setUp(self):
        self.model = RosenblattPerceptron(learning_rate=0.1)
        self.model.weights = [0.0] * 25
        self.model.threshold = 0.05

    def test_classification(self):
        positive = [1.0] + [0.0] * 24
        negative = [0.0] * 25

        self.model.weights[0] = 0.2

        self.assertEqual(self.model.predict(positive), 1)
        self.assertEqual(self.model.predict(negative), -1)

    def test_weight_correction(self):
        features = [1.0] + [0.0] * 24

        result = self.model.train_sample(features, 1)

        self.assertEqual(result["predicted"], -1)
        self.assertAlmostEqual(self.model.weights[0], 0.2)
        self.assertEqual(self.model.step, 1)

    def test_convergence(self):
        samples = [
            ([1.0] + [0.0] * 24, 1),
            ([0.0, 1.0] + [0.0] * 23, -1),
        ]

        result = self.model.train_until_convergence(samples)

        self.assertTrue(result["converged"])
        self.assertEqual(result["history"][-1]["errors"], 0)


if __name__ == "__main__":
    unittest.main()
