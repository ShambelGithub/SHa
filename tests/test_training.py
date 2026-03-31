import unittest

from bus_network.agent import GreedyTransitAgent, QLearningAgent
from bus_network.environment import BusTransitEnv
from bus_network.training import run_training


class TestTraining(unittest.TestCase):
    def test_training_returns_summary(self):
        summary = run_training(episodes=2)
        self.assertIn("best_reward", summary)
        self.assertIn("best_state", summary)
        self.assertIn("reward_history", summary)
        self.assertEqual(len(summary["reward_history"]), 2)


class TestQLearningAgent(unittest.TestCase):
    def _make_env(self) -> BusTransitEnv:
        return BusTransitEnv(
            demand=[100.0, 80.0],
            target_frequencies=[5, 4],
            initial_frequencies=[3, 3],
            max_steps=10,
        )

    def test_qlearning_returns_result(self):
        env = self._make_env()
        agent = QLearningAgent(env)
        result = agent.train(episodes=5, seed=42)
        self.assertEqual(len(result.reward_history), 5)
        self.assertIsInstance(result.best_reward, float)
        self.assertIsInstance(result.best_state, tuple)

    def test_epsilon_decays_over_training(self):
        env = self._make_env()
        agent = QLearningAgent(env, epsilon_start=1.0, epsilon_end=0.05, epsilon_decay=0.5)
        initial_epsilon = agent.epsilon
        agent.train(episodes=10, seed=0)
        self.assertLess(agent.epsilon, initial_epsilon)

    def test_qlearning_converges_over_episodes(self):
        """Q-learning reward should improve from early to late training episodes."""
        env = self._make_env()
        qlearner = QLearningAgent(env, epsilon_start=1.0, epsilon_end=0.05, epsilon_decay=0.99)
        result = qlearner.train(episodes=200, seed=42)
        early_avg = sum(result.reward_history[:20]) / 20
        late_avg = sum(result.reward_history[-20:]) / 20
        self.assertGreater(late_avg, early_avg)

    def test_q_table_populated_after_training(self):
        env = self._make_env()
        agent = QLearningAgent(env)
        agent.train(episodes=5, seed=1)
        self.assertGreater(len(agent.q_table), 0)


if __name__ == "__main__":
    unittest.main()
