import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from rl_mass import train


class TrainEpisodeStatsTest(unittest.TestCase):
    def test_summarize_episode_lengths(self):
        stats = train.summarize_episode_lengths([10, 20, 30], control_hz=30)

        self.assertEqual(stats["avg_episode_steps"], 20.0)
        self.assertEqual(stats["max_episode_steps"], 30)
        self.assertAlmostEqual(stats["avg_episode_seconds"], 20.0 / 30.0)
        self.assertAlmostEqual(stats["max_episode_seconds"], 1.0)

    def test_reward_csv_columns_include_episode_duration_stats(self):
        self.assertIn("avg_episode_steps", train.REWARD_CSV_COLUMNS)
        self.assertIn("avg_episode_seconds", train.REWARD_CSV_COLUMNS)
        self.assertIn("max_episode_steps", train.REWARD_CSV_COLUMNS)
        self.assertIn("max_episode_seconds", train.REWARD_CSV_COLUMNS)

    def test_reward_csv_header_mismatch_raises(self):
        with TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "train_rewards.csv"
            path.write_text("iter,elapsed_s,num_episode\n")

            with self.assertRaises(RuntimeError):
                train.ensure_reward_csv_header(str(path))


if __name__ == "__main__":
    unittest.main()
