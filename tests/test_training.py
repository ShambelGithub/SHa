import unittest

from bus_network.training import run_training


class TestTraining(unittest.TestCase):
    def test_training_returns_summary(self):
        summary = run_training(episodes=2)
        self.assertIn("best_reward", summary)
        self.assertIn("best_state", summary)
        self.assertIn("reward_history", summary)
        self.assertEqual(len(summary["reward_history"]), 2)


if __name__ == "__main__":
    unittest.main()
