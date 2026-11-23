
import unittest

from task_2.hopfield import HopfieldNetwork


class TestHopfield(unittest.TestCase):
    def setUp(self):
        self.patterns = {
            "0": [1 if i % 2 == 0 else -1 for i in range(24)] + [1],
            "1": [1 if (i // 2) % 2 == 0 else -1 for i in range(24)] + [1],
            "2": [1 if (i // 4) % 2 == 0 else -1 for i in range(24)] + [1],
        }

        self.network = HopfieldNetwork()
        self.network.train(self.patterns)

    def test_weight_matrix(self):
        weights = self.network.weights

        self.assertEqual(len(weights), 25)

        for i in range(25):
            self.assertEqual(weights[i][i], 0)

            for j in range(25):
                self.assertEqual(weights[i][j], weights[j][i])

    def test_recognition(self):
        for name, pattern in self.patterns.items():
            result = self.network.recognize(pattern)

            self.assertTrue(result["converged"])
            self.assertEqual(result["predicted"], name)
            self.assertEqual(result["output"], pattern)

    def test_damaged_pattern(self):
        for name, pattern in self.patterns.items():
            damaged = pattern.copy()
            damaged[4] *= -1

            result = self.network.recognize(damaged)

            self.assertTrue(result["converged"])
            self.assertEqual(result["predicted"], name)
            self.assertEqual(result["output"], pattern)


if __name__ == "__main__":
    unittest.main()
