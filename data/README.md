# data/

Per-case asset layout: `data/<case>/{model.zip, config/}`.

## Unzip before first run

Model geometry (restpos, restpos_lowres, cons_*.geo) ships as **zips**
to save Git LFS storage. Extract once per case:

```bash
cd data/lion          && unzip -o model.zip
cd data/minotaur      && unzip -o model.zip
cd data/minotaur_leg  && unzip -o model.zip
```

Re-running `unzip -o` is idempotent. The solver raises a
`FileNotFoundError` with the exact extract command if `model/` is
missing.

## Why zipped

`.geo` files are ASCII and compress ~60–80%. Trade-off: edits to any
file inside `model.zip` force a full LFS reupload (no per-file dedup).
Use this layout for **stable** assets — if you iterate on cons
sequences, work in the unzipped `model/` and re-zip when stable.
