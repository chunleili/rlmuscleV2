# rlmuscleV2

## Art assets

Per-case art assets (the large `model/` geometry) are **not** in git. They live in
a private Hugging Face dataset: [`chunleili/rlmuscleV2-assets`](https://huggingface.co/datasets/chunleili/rlmuscleV2-assets).

Layout: one `<case>/model.zip` per case; each zip carries a `model/` prefix and
extracts to `data/<case>/model/`. Available cases: `humanV2`, `lion`, `minotaur`,
`minotaur_leg`.

### Download

One-time setup:

```bash
pip install huggingface_hub
hf auth login        # paste a read token from https://huggingface.co/settings/tokens
```

The repo is private — ask the owner to add you as a collaborator first, otherwise
the files are invisible.

Then fetch (cases are auto-discovered from the repo):

```bash
python tools/fetch_assets.py                # all cases
python tools/fetch_assets.py humanV2 lion   # only these
python tools/fetch_assets.py --force        # re-extract even if model/ exists
```

### Upload (maintainer)

```bash
hf upload chunleili/rlmuscleV2-assets data/<case>/model.zip <case>/model.zip --repo-type dataset
```

Build `model.zip` from a `model/` dir with `cd data/<case> && zip -r model.zip model/`
(keep the `model/` prefix so it extracts in place).
