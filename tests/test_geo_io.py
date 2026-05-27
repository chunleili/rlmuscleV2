"""Tests for muscle.coupleOneWay.geo_io against real lion data."""
from pathlib import Path

import numpy as np
import pytest

from muscle.coupleOneWay.geo_io import GeoFile, read_cons_frame

DATA = Path(__file__).resolve().parents[1] / "data" / "model" / "lion"

skip_no_data = pytest.mark.skipif(
    not (DATA / "restpos.geo").exists(),
    reason="lion .geo blobs not present (cp them in from realtimeMuscle/data/model/lion)",
)


@skip_no_data
def test_restpos_reads():
    g = GeoFile(str(DATA / "restpos.geo"))
    assert g.positions.dtype == np.float32
    assert g.positions.ndim == 2 and g.positions.shape[1] == 3
    assert g.tet_indices.dtype == np.int32
    assert g.tet_indices.ndim == 2 and g.tet_indices.shape[1] == 4
    assert g.gluetoanimation.dtype == np.bool_
    assert g.gluetoanimation.shape[0] == g.positions.shape[0]
    assert g.tet_indices.min() >= 0
    assert g.tet_indices.max() < g.positions.shape[0]


@skip_no_data
def test_cons_frame_reads():
    g = GeoFile(str(DATA / "cons_1.geo"))
    assert g.pts.dtype == np.int32
    assert g.target_pos.dtype == np.float32
    assert g.target_pos.shape == (g.pts.shape[0], 3)


@skip_no_data
def test_read_cons_frame_matches_full_parse():
    full = GeoFile(str(DATA / "cons_45.geo")).target_pos
    fast = read_cons_frame(str(DATA), 45)
    np.testing.assert_array_equal(full, fast)


@skip_no_data
def test_parity_with_reference_reader():
    """Round-trip parity check: our reader vs. realtimeMuscle/engine/geo.py."""
    import sys

    ref_root = Path.home() / "Dev" / "realtimeMuscle"
    if not (ref_root / "engine" / "geo.py").exists():
        pytest.skip("realtimeMuscle reference not available at ~/Dev/realtimeMuscle")
    sys.path.insert(0, str(ref_root))
    try:
        from engine.geo import Geo as RefGeo
    finally:
        sys.path.pop(0)

    ref = RefGeo(str(DATA / "cons_1.geo"))
    ours = GeoFile(str(DATA / "cons_1.geo"))

    ref_pos = np.asarray(ref.get_pos(), dtype=np.float32)
    ref_pts = np.asarray(ref.get_pts(), dtype=np.int32)
    ref_tp = np.asarray(ref.get_target_pos(), dtype=np.float32)

    np.testing.assert_array_equal(ours.positions, ref_pos)
    np.testing.assert_array_equal(ours.pts, ref_pts)
    np.testing.assert_array_equal(ours.target_pos, ref_tp)
