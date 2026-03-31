import unittest

from bus_network.environment import BusTransitEnv


class TestBusTransitEnv(unittest.TestCase):
    def test_step_updates_reward(self):
        env = BusTransitEnv(
            demand=[100.0, 80.0],
            target_frequencies=[5, 4],
            initial_frequencies=[3, 3],
            max_steps=1,
        )
        state = env.reset()
        self.assertEqual(state, (3, 3))
        next_state, reward, done, info = env.step(0)
        self.assertEqual(next_state, (4, 3))
        self.assertTrue(done)
        self.assertIn("reward", info)
        self.assertIsInstance(reward, float)


if __name__ == "__main__":
    unittest.main()
