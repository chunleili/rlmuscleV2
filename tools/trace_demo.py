"""Trace which engine.py symbols a demo actually exercises.

Run ONE demo per subprocess:
    PYTHONPATH=src python tools/trace_demo.py <out.json> <module> -- <demo argv...>

Records (union, written to out.json):
  - "kernels":  every Warp kernel key passed to wp.launch
  - "funcs":    every engine.py Python function/method that executed (sys.setprofile)
Warp @wp.func/@wp.kernel *bodies* never run as Python, so kernel usage comes
from the wp.launch hook; @wp.func usage is derived statically afterwards.
"""
import importlib
import json
import os
import runpy
import sys

import warp as wp

ENGINE_TAG = os.sep + os.path.join("muscle", "engine.py")

used_kernels = set()
used_funcs = set()

# ---- hook wp.launch to log dispatched kernel keys ----
_orig_launch = wp.launch
def _traced_launch(kernel, *a, **k):
    key = getattr(kernel, "key", None) or getattr(kernel, "__name__", None) or str(kernel)
    used_kernels.add(key)
    return _orig_launch(kernel, *a, **k)
wp.launch = _traced_launch

# ---- profile engine.py python funcs/methods ----
def _profiler(frame, event, arg):
    if event == "call":
        co = frame.f_code
        if co.co_filename.endswith(ENGINE_TAG):
            used_funcs.add(co.co_name)
    return None

def main():
    out_path = sys.argv[1]
    module = sys.argv[2]
    assert sys.argv[3] == "--"
    demo_argv = sys.argv[4:]

    # present argv to the demo as if invoked directly
    sys.argv = [module] + demo_argv

    sys.setprofile(_profiler)
    try:
        runpy.run_module(module, run_name="__main__", alter_sys=True)
    finally:
        sys.setprofile(None)
        with open(out_path, "w") as f:
            json.dump({"kernels": sorted(used_kernels),
                       "funcs": sorted(used_funcs)}, f, indent=2)
        print(f"[trace] {module}: {len(used_kernels)} kernels, "
              f"{len(used_funcs)} funcs -> {out_path}")

if __name__ == "__main__":
    main()
