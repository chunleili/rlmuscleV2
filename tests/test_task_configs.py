import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class TaskConfigTests(unittest.TestCase):
    def test_overhead_squat_uses_pose_cutoff_instead_of_height_termination(self):
        config_path = ROOT / "data" / "rl_mass" / "configs" / "overhead-squat.json"
        config = json.loads(config_path.read_text())

        self.assertIs(config.get("height_termination"), False)
        self.assertEqual(config.get("pose_error_threshold"), 15.0)


if __name__ == "__main__":
    unittest.main()
