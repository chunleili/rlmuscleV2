# Moreira 2021 — real adult lower-limb sEMG (walk-vs-EMG GT)

Ground-truth surface EMG for the humanoid walk activation comparison.

**Source:** Moreira, Figueiredo, Fonseca, Vilas-Boas, Santos (2021), *Lower limb
kinematic, kinetic, and EMG data from young healthy humans during walking at
controlled speeds*, **Scientific Data**.
figshare collection `10.6084/m9.figshare.c.4923162` (ASCII files article
`13169240`, file id `26902613`). Re-download the full 1.3 GB ASCII zip from
there if `raw/moreira_ascii.zip` is missing (it is git-ignored — too big to
commit).

**Subjects:** 16 healthy adults (8 M / 8 F), 7 treadmill speeds
V1..V7 = 1.0/1.5/2.0/2.5/3.0/3.5/4.0 km/h. Per-subject anthropometrics in
`metadata/Participant*.txt`.

**EMG:** 4 surface muscles — `VL` Vastus Lateralis, `BF` Biceps Femoris,
`TA` Tibialis Anterior, `GAL` Gastrocnemius Lateralis. Already
rectified + linear-enveloped, gait-cycle-normalized (0 = right heel strike).

## What is committed here

- `metadata/Participant<N>.txt` — all 16 subjects (id, age, sex, height, mass, leg/foot length).
- `P8_V4_R/T*.txt` — the default comparison subset: **Participant 8** (male,
  **1.72 m, 75.8 kg** — the closest body match to the SNU humanoid, an adult
  male ≈1.7 m), speed **V4 = 2.5 km/h** (the highest speed with processed R-leg
  envelopes for this subject), **right** leg, 10 trials.
- `raw/` — the full ASCII zip, **git-ignored** (1.3 GB). Persisted on disk for
  re-extraction; not committed.

## EMG envelope file format (`P8_V4_R/T*.txt`)

Tab-separated, header row then ~1001 rows (one gait cycle, 0→100%):

```
St1_VL  St2_VL  St1_BF  St2_BF  St1_TA  St2_TA  St1_GAL  St2_GAL
```

8 columns = 4 muscles × 2 strides (`St1`, `St2`) per trial. Amplitudes are
envelope μV (NOT MVC-normalized) — peak-normalize per muscle for shape/timing
comparison. 10 trials × 2 strides = 20 stride envelopes per muscle.

## SNU(284-muscle humanoid) → Moreira muscle map (right leg)

| Moreira | rl_mass SNU |
|---|---|
| VL  | `R_Vastus_Lateralis` |
| BF  | `R_Bicep_Femoris_Longus` |
| TA  | `R_Tibialis_Anterior` |
| GAL | `R_Gastrocnemius_Lateral_Head` |

To extract another subject/speed/side from the raw zip:
`unzip -qjo raw/moreira_ascii.zip "ASCII files/Participant<N>/Processed_Data/V<k>/<L|R>/EMG envelope/*" -d <dest>`
