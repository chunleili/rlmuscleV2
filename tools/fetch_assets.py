"""Download per-case art assets (model.zip) from the public HF dataset.

Assets live in chunleili/rlmuscleV2-assets as <case>/model.zip; each zip holds a
model/ prefix, so it extracts to data/<case>/model/. Cases are discovered from
the repo, so new cases are picked up automatically.

Prereqs (one-time):
    pip install huggingface_hub

    python tools/fetch_assets.py                 # all cases
    python tools/fetch_assets.py humanV2 lion     # only these cases
    python tools/fetch_assets.py --force          # re-extract even if model/ exists
"""
import os
import sys
import zipfile

from huggingface_hub import HfApi, hf_hub_download

REPO = "chunleili/rlmuscleV2-assets"

args = [a for a in sys.argv[1:] if not a.startswith("-")]
force = "--force" in sys.argv[1:]

api = HfApi()
files = [f for f in api.list_repo_files(REPO, repo_type="dataset") if f.endswith("/model.zip")]
cases = {f.split("/")[0]: f for f in files}
if args:
    missing = [c for c in args if c not in cases]
    if missing:
        sys.exit(f"unknown case(s): {missing}. available: {sorted(cases)}")
    cases = {c: cases[c] for c in args}

for case, remote in cases.items():
    out = f"data/{case}"
    if os.path.isdir(f"{out}/model") and not force:
        print(f"skip {case} (data/{case}/model exists; --force to redo)")
        continue
    print(f"fetch {case} ...")
    z = hf_hub_download(REPO, remote, repo_type="dataset")
    os.makedirs(out, exist_ok=True)
    zipfile.ZipFile(z).extractall(out)  # zip carries model/ prefix
    print(f"  -> {out}/model/")

print("done")
