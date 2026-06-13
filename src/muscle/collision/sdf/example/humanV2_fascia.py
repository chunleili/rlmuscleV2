"""humanV2 with BOTH fascia squeeze AND animated-bone SDF collision.

The flagship ``humanV2`` SDF demo (``humanV2_gmg.json``) handles muscle-vs-bone
collision but no fascia; the one-way ``humanV2_fascia.json`` adds the two-way
fascia shell that squeezes the muscle inward but has no collision. This entry
runs both at once: the same animated-bone collider on top of the fascia config,
so the muscle is squeezed by the fascia AND pushed off the moving skeleton.

No new solver code — it reuses the ``humanV2`` SDF demo, which auto-detects the
fascia tets (``sim.nvm < sim.NV``) and renders muscle + fascia as separate USD
prims.

Run::

    PYTHONPATH=src python -m muscle.collision.sdf.example.humanV2_fascia \\
        --end_frame 50
    # -> output/humanV2_fascia_coll.usd
"""
from __future__ import annotations

import sys

from muscle.collision.sdf.example.humanV2 import main as _humanV2_main


def main():
    # Fascia + collision defaults; user-supplied flags (appended last) win,
    # since argparse keeps the final value for a repeated option.
    defaults = [
        "--config", "data/humanV2/config/humanV2_fascia.json",
        "--usd", "output/humanV2_fascia_coll.usd",
    ]
    _humanV2_main(defaults + sys.argv[1:])


if __name__ == "__main__":
    main()
