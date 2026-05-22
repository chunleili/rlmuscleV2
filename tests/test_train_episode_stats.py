import os
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

    def test_resolve_output_paths_from_outdir(self):
        run_paths = train.resolve_run_paths(outdir="output/overhead-squat")

        self.assertEqual(run_paths["reward_csv_path"], "output/overhead-squat/train_rewards.csv")
        self.assertEqual(run_paths["nn_dir"], "output/overhead-squat/nn")

    def test_resolve_resume_metadata_from_checkpoint_path(self):
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir) / "output" / "overhead-squat"
            nn_dir = root / "nn"
            nn_dir.mkdir(parents=True)
            (nn_dir / "current.pt").write_text("")
            (nn_dir / "current_muscle.pt").write_text("")
            reward_csv = root / "train_rewards.csv"
            reward_csv.write_text(
                ",".join(train.REWARD_CSV_COLUMNS) + "\n"
                "10,1,1,1,1,1,1,1,1,1,1,1,1,1\n"
                "25,2,1,1,1,1,1,1,1,1,1,1,1,1\n"
            )

            run_paths = train.resolve_run_paths(
                resume=str(nn_dir / "current.pt"),
            )

            self.assertEqual(run_paths["nn_dir"], str(nn_dir))
            self.assertEqual(run_paths["reward_csv_path"], str(reward_csv))
            self.assertEqual(run_paths["resume_prefix"], str(nn_dir / "current"))
            self.assertEqual(run_paths["resume_iter"], 25)

    def test_infer_resume_iter_from_reward_csv(self):
        with TemporaryDirectory() as tmpdir:
            reward_csv = Path(tmpdir) / "train_rewards.csv"
            reward_csv.write_text(
                ",".join(train.REWARD_CSV_COLUMNS) + "\n"
                "3,1,1,1,1,1,1,1,1,1,1,1,1,1\n"
                "8,2,1,1,1,1,1,1,1,1,1,1,1,1\n"
            )

            self.assertEqual(train.infer_resume_iter(str(reward_csv)), 8)

    def test_resolve_bare_resume_name_with_outdir(self):
        with TemporaryDirectory() as tmpdir:
            cwd = Path.cwd()
            try:
                os.chdir(tmpdir)
                root = Path("output") / "overhead-squat"
                nn_dir = root / "nn"
                nn_dir.mkdir(parents=True)
                (nn_dir / "current.pt").write_text("")
                (nn_dir / "current_muscle.pt").write_text("")
                (root / "train_rewards.csv").write_text(
                    ",".join(train.REWARD_CSV_COLUMNS) + "\n"
                    "12,1,1,1,1,1,1,1,1,1,1,1,1,1\n"
                )

                run_paths = train.resolve_run_paths(
                    outdir="output/overhead-squat",
                    resume="current",
                )
            finally:
                os.chdir(cwd)

        self.assertEqual(run_paths["nn_dir"], "output/overhead-squat/nn")
        self.assertEqual(run_paths["reward_csv_path"], "output/overhead-squat/train_rewards.csv")
        self.assertEqual(run_paths["resume_prefix"], "output/overhead-squat/nn/current")
        self.assertEqual(run_paths["resume_iter"], 12)


if __name__ == "__main__":
    unittest.main()
