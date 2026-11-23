
import unittest

from task_2.hamming import HammingNetwork


class TestHamming(unittest.TestCase):
    def setUp(self):
        self.patterns = {
            "0": [-1] * 25,
            "1": [1] * 25,
            "2": [1] * 12 + [-1] * 13,
        }

        self.network = HammingNetwork(epsilon=0.2)
        self.network.train(self.patterns)

    def test_weights(self):
        self.assertEqual(self.network.size, 25)
        self.assertEqual(self.network.bias, 12.5)

        for name, vector in self.patterns.items():
            expected = [value / 2 for value in vector]
            self.assertEqual(self.network.weights[name], expected)

    def test_recognition(self):
        for name, vector in self.patterns.items():
            result = self.network.recognize(vector)

            self.assertTrue(result["converged"])
            self.assertEqual(result["predicted"], name)
            self.assertEqual(result["distances"][name], 0)

    def test_damaged_pattern(self):
        damaged = self.patterns["2"].copy()
        damaged[4] *= -1

        result = self.network.recognize(damaged)

        self.assertTrue(result["converged"])
        self.assertEqual(result["predicted"], "2")
        self.assertEqual(result["distances"]["2"], 1)


if __name__ == "__main__":
    unittest.main()
