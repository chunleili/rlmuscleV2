"""Per-tet muscle activation sources.

An *activation source* is any object exposing ``per_tet(frame) -> (NT,) float32`` —
the per-(muscle)-tet activation for that frame, ready for
``MultiGridSim.set_fiber_activation()``. The run loop is source-agnostic: it just
asks the source each frame, so a USD curve asset, a linear ramp, an RL policy, a
constant, etc. are interchangeable. Two concrete sources live here:

  * ``UsdActivationSource`` — driven by a USD per-muscle curve asset (e.g. walk.usd).
  * ``RampActivationSource`` — a fixed mask ramped 0->1 over the frame range.

USD curve asset (walk.usd) carries line-of-action muscles under
``/Human/Muscles/<Name>``, each with a per-frame scalar ``primvars:muscle:activation``
on its ``curve`` (BasisCurves). The volumetric otto solver instead has TetMesh muscles
keyed by a camelCase ``muscle_id`` (e.g. ``L_bicepsBrachialis``).
``muscle_mapping.json`` maps every curve to ONE muscle (many-to-one): several lines
(``Bicep_Femoris_Longus``, ``_Short``, ``_Short1``) share one muscle, so their
activations are aggregated (mean by default; ``max`` optional). Unmapped entries
(authored as empty ``{}``) are skipped; muscles with no source stay passive (0).
"""

import json
from pathlib import Path

import numpy as np


class RampActivationSource:
    """Linearly ramp a fixed per-tet ``mask`` from 0 to 1 across the frame range —
    drives one named muscle's activation against the passive baseline (demo /
    single-muscle contraction test). ``per_tet(frame) = clamp((frame-start)/span,
    0, 1) * mask``."""

    def __init__(self, mask, start_frame, end_frame):
        self._mask = np.asarray(mask, dtype=np.float32)
        self.start_frame = int(start_frame)
        self.end_frame = int(end_frame)
        self._span = max(1, self.end_frame - self.start_frame)

    def per_tet(self, frame: int) -> np.ndarray:
        a = (int(frame) - self.start_frame) / self._span
        a = min(1.0, max(0.0, a))                    # clamp outside [start, end]
        return (a * self._mask).astype(np.float32)


class UsdActivationSource:
    """Maps a USD per-muscle activation curve asset (e.g. walk.usd) onto per-tet
    activation.

    Parameters
    ----------
    usd_path : str           the curve-muscle asset with per-frame activation
    mapping_path : str       muscle_mapping.json ({"mappings": {"<name>/curve": {"value": "<muscle_id>"}}})
    muscle_id_np : (NT,) int per-(muscle)-tet global index into ``muscle_id_strings``
    muscle_id_strings : list global muscle-id name table (camelCase names)
    start_frame, end_frame   inclusive frame range to read
    muscles_root : str       scope holding the curve muscles
    aggregate : "mean"|"max" how to merge several curves onto one muscle
    """

    def __init__(self, usd_path, mapping_path, muscle_id_np, muscle_id_strings,
                 start_frame, end_frame, muscles_root="/Human/Muscles",
                 aggregate="mean", activation_attr="primvars:muscle:activation"):
        from pxr import Usd

        self.start_frame = int(start_frame)
        self.end_frame = int(end_frame)
        self._n_frames = self.end_frame - self.start_frame + 1
        self._muscle_id_np = np.asarray(muscle_id_np, dtype=np.int64)
        self._nt = int(self._muscle_id_np.shape[0])
        self.aggregate = str(aggregate).lower()

        if not Path(mapping_path).is_file():
            raise FileNotFoundError(f"muscle mapping not found: {mapping_path}")
        mapping = json.loads(Path(mapping_path).read_text()).get("mappings", {})

        stage = Usd.Stage.Open(str(usd_path))
        if stage is None:
            raise FileNotFoundError(f"cannot open activation USD: {usd_path}")

        # 1) read each mapped walk curve's per-frame activation, grouped by otto name
        otto_lines: dict[str, list] = {}
        n_walk = n_unmapped = n_missing = 0
        for key, m in mapping.items():               # key = "<MuscleName>/curve"
            otto = m.get("value") if isinstance(m, dict) else None
            if not otto:
                n_unmapped += 1                      # authored empty {} -> no otto target
                continue
            prim = stage.GetPrimAtPath(f"{muscles_root}/{key}")
            attr = prim.GetAttribute(activation_attr) if prim and prim.IsValid() else None
            if attr is None or not attr.HasValue():
                n_missing += 1
                continue
            arr = np.empty(self._n_frames, dtype=np.float32)
            for i, f in enumerate(range(self.start_frame, self.end_frame + 1)):
                v = attr.Get(float(f))
                arr[i] = 0.0 if v is None else float(v)
            otto_lines.setdefault(otto, []).append(arr)
            n_walk += 1

        # 2) aggregate many walk lines -> one curve per otto muscle
        otto_acti: dict[str, np.ndarray] = {}
        for name, lines in otto_lines.items():
            stacked = np.stack(lines, axis=0)        # (n_lines, n_frames)
            merged = stacked.max(axis=0) if self.aggregate == "max" else stacked.mean(axis=0)
            otto_acti[name] = merged.astype(np.float32)

        # 3) scatter onto a [n_strings, n_frames] table indexed by global muscle-id
        name_to_idx = {s: i for i, s in enumerate(muscle_id_strings)}
        n_strings = len(muscle_id_strings)
        self._table = np.zeros((n_strings, self._n_frames), dtype=np.float32)
        matched, missing_otto = [], []
        for name, arr in otto_acti.items():
            si = name_to_idx.get(name)
            if si is None:
                missing_otto.append(name)            # mapping value not in asset
                continue
            self._table[si] = arr
            matched.append(name)

        # otto muscles actually present in the tet asset but with no walk drive
        driven = {name_to_idx[n] for n in matched}
        present = {int(i) for i in np.unique(self._muscle_id_np)}
        no_source = sorted(muscle_id_strings[i] for i in present
                           if i not in driven and 0 <= i < n_strings)

        print(f"[usd-activation] {usd_path}  agg={self.aggregate}  "
              f"frames {self.start_frame}..{self.end_frame}")
        print(f"  curves read: {n_walk}  unmapped(empty): {n_unmapped}  "
              f"missing prim/attr: {n_missing}")
        print(f"  muscles driven: {len(matched)}  "
              f"mapping-name absent from asset: {len(missing_otto)}  "
              f"present-but-no-source: {len(no_source)}")
        if missing_otto:
            print(f"    absent: {sorted(missing_otto)[:12]}"
                  f"{' ...' if len(missing_otto) > 12 else ''}")
        peak = float(self._table.max()) if self._table.size else 0.0
        print(f"  activation peak over all muscles/frames: {peak:.3f}")

    def per_tet(self, frame: int) -> np.ndarray:
        """Per-(muscle)-tet activation at integer ``frame`` (clamped to range)."""
        fi = int(frame) - self.start_frame
        fi = max(0, min(self._n_frames - 1, fi))
        return self._table[:, fi][self._muscle_id_np].astype(np.float32)
