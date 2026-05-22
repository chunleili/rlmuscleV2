import importlib.util
import sys
import unittest
from pathlib import Path


_SCRIPT = Path(__file__).resolve().parents[1] / "tools" / "profile_warp.py"
_SPEC = importlib.util.spec_from_file_location("profile_warp", _SCRIPT)
profile_warp = importlib.util.module_from_spec(_SPEC)
assert _SPEC.loader is not None
sys.modules[_SPEC.name] = profile_warp
_SPEC.loader.exec_module(profile_warp)


class ProfileWarpMetricSelectionTests(unittest.TestCase):
    def test_select_env_metric_names_preserves_declared_order(self):
        selected = profile_warp.select_env_metric_names(
            ["steps_at_once", "get_states"]
        )
        self.assertEqual(selected, ["get_states", "steps_at_once"])

    def test_select_env_metric_names_rejects_unknown_metric(self):
        with self.assertRaises(ValueError):
            profile_warp.select_env_metric_names(["not_a_metric"])


if __name__ == "__main__":
    unittest.main()
