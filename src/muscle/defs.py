"""Shared enum definitions for the muscle simulation.

``ConstraintType`` — per-constraint category tag. Replaces the old loose
module-level integer "macros". The integer *values* are preserved exactly so
any geometry / config that encoded them keeps matching.

``FemFlag`` — bitwise FEM solver flags (was ``LINEARENERGY`` / ``NORMSTIFFNESS``).

Both subclass the int-flavoured enums on purpose: members *are* ints, so they
drop straight into ``np.int32`` arrays (``c_type``), ``dict`` keys
(``cons_ranges``), Warp kernel arguments and ``==`` comparisons, and JSON —
no call-site arithmetic breaks. A plain ``Enum`` would not be int-compatible.
"""

from __future__ import annotations

from enum import IntEnum, IntFlag


class ConstraintType(IntEnum):
    PIN             =  157323
    ATTACH          =  1650556
    PINTOTARGET     =  771624311
    TETVOLUME       =  -215389979
    TETFIBERNORM    =  -303462111
    DISTANCELINE    =  1621136047
    TETARAP         =  -92199131
    TETARAPSVD      =  -92199132
    TETARAPNORM     =  -885573303
    TETARAPANISO    =  -92199133   # anisotropic (along-fiber) ARAP, passive HOBAK
    TETFIBERDGF     =  -403562211
    TETSNH          =  -503462311
    TETFIBERMILLARD =  -503662411
    DISTANCE        =  1621136100   # two-vertex edge spring (MassSpring)
    TETVOLUMEHARD   =  -215389990   # hard det(F)=1 volume (MassSpring)
    TETINVERSION    =  -215389991   # inverted-tet untangle (MassSpring)


class FemFlag(IntFlag):
    """ARAP / FEM solver flags (from pbd_types.h)."""
    LINEARENERGY  = 1 << 0
    NORMSTIFFNESS = 1 << 1
