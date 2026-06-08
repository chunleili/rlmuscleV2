"""Tests for geo_io fascia (mixed tet+poly, named attrs, point groups)."""
from pathlib import Path

import numpy as np
import pytest

from muscle.util.geo_io import GeoFile

FASCIA = Path(__file__).resolve().parents[1] / "data" / "humanV2" / "model" / "fascia" / "fascia.geo"

skip_no_data = pytest.mark.skipif(
    not FASCIA.exists(),
    reason="fascia.geo not present (cp ~/Dev/fascia.geo data/humanV2/model/fascia/)",
)


@skip_no_data
def test_fascia_tets_and_positions():
    g = GeoFile(str(FASCIA))
    assert g.positions.shape == (15348, 3)
    assert g.positions.dtype == np.float32
    # Tetrahedron_run only: 66785 tets, polygons excluded.
    assert g.tet_indices.shape == (66785, 4)
    assert g.tet_indices.min() >= 0
    assert g.tet_indices.max() < 15348


@skip_no_data
def test_fascia_named_point_attrs():
    g = GeoFile(str(FASCIA))
    d = g.point_attr("tissuetomuscledistance")
    assert d.shape == (15348,)
    assert np.allclose(d, 0.01, atol=1e-6)
    mat = g.point_attr("materialW")
    assert mat.shape == (15348, 3)
    assert g.point_attr("solidshapestiffness").mean() == pytest.approx(5000.0, rel=1e-3)


@skip_no_data
def test_fascia_point_groups():
    g = GeoFile(str(FASCIA))
    rigid = g.point_group("rigid")          # ordered list
    inner = g.point_group("tissue_inner_points")  # boolRLE
    bone = g.point_group("tissue_bone_points")     # boolRLE
    for grp in (rigid, inner, bone):
        assert grp.dtype == np.int32
        assert grp.ndim == 1
        assert grp.min() >= 0 and grp.max() < 15348
        assert len(np.unique(grp)) == len(grp)  # no dupes
    assert len(rigid) > 0 and len(inner) > 0
