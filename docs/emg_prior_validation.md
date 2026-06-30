# EMG-Prior Walk: Matching rl_mass Muscle Activations to Real sEMG

**Branch:** `finger` · **Date:** 2026-06-30

## Goal

Make the rl_mass humanoid's *walk* muscle activations match real surface EMG
(Moreira et al. 2021), as a physiological-plausibility validation and a paper
contribution. Baseline (tracking-only policy) activations were non-physiological:
they sat outside the human EMG band ~82 % of the gait cycle.

## Result

| Metric | Baseline (no prior) | **With EMG prior** |
|---|---|---|
| mean Pearson (shape) | −0.19 | **0.97** |
| mean in-band (mean ± 1 std) | 17.6 % | **91 %** |
| dead muscles | — | **0 / 4** |
| timing peaks vs EMG | off (GAL 39 % lag) | **all aligned** |
| gait | stable | stable (return ↑) |

Per channel (final, `nn/max`): VL r=0.89 / BF 0.99 / TA 0.99 / GAL 0.99; timing
lag 0; in-band 88 / 100 / 100 / 75 %.

Figure: `output/walk_vs_emg_emgprior_final/with_prior.pdf`.
Animation: `output/anim/walk_emgprior.usd` (skeleton + 284 muscle-activation curves).

## How it works — division of labour

The MASS framework uses two networks:

- **Policy** `SimulationNN` → decides the **motion** (target joint poses / desired
  torques), trained by imitation (PPO) to track the reference walk.
- **Muscle network** `MuscleNN` → decides **how to actuate** that motion, mapping
  `(muscle torques, desired torques) → 284 activations`. These activations drive
  the physics (`set_activation_levels` → muscle forces → sim step).

The activation→torque map is **redundant**: many activation patterns produce the
same joint torques. The tracking-only solver picks a non-physiological tie-break.
The **EMG prior is added to the MuscleNN loss only** — it does not change the
locomotion objective. It selects, among the redundant activation solutions that
realize the motion, the one whose surface muscles match real EMG. The matched
activations genuinely actuate the walk (not a cosmetic overlay).

> Motion is set by the imitation policy → muscle redundancy leaves a null space →
> the EMG prior picks the physiological activation in that null space → those
> activations drive the simulated walk.

## The prior-injection formula

Per muscle-training minibatch (`src/rl_mass/train.py`, `OptimizeMuscleNN`):

$$
L = 0.01\,L_{\text{reg}} + L_{\text{torque}} + \lambda\,L_{\text{emg}}
$$

- $L_{\text{reg}} = \operatorname{mean}(a^2)$ — minimum-activation regularizer
  ($\Sigma a^2$ family, same as Static Optimization).
- $L_{\text{torque}} = \operatorname{mean}\!\left(\left(\dfrac{\tau - \tau_{\text{des}}}{100}\right)^{2}\right)$
  — the activations must produce the joint torques the motion requires
  ($\tau = L\,a + b$).
- **$L_{\text{emg}}$** — the EMG prior (new):

$$
L_{\text{emg}} = \operatorname*{mean}_{\substack{c \in \text{channels}\\ b \in \text{samples}}}
\Big[\, w_c \,\big(\bar a_c(b) - s\,\hat e_c(\varphi_b)\big)^{2} \,\Big]
$$

with

$$
\bar a_c(b) = \sum_{m \in G_c} \omega_{c,m}\, a_m(b)
\qquad
\omega_{c,m} = \frac{f0_m}{\sum_{m' \in G_c} f0_{m'}}
$$

$$
\varphi_b = \frac{\operatorname{fmod}(t_b,\, T_{\text{cycle}})}{T_{\text{cycle}}}
\qquad
\mathrm{gait\%}(\varphi) = \big((\varphi - \varphi_{\text{hs}}) \bmod 1\big)\cdot 100
$$

- $\bar a_c(b)$ — f0 / PCSA-weighted group activation; the weights $\omega_{c,m}$
  sum to 1 over each group.
- $\hat e_c(\varphi)$ — peak-normalised Moreira cohort EMG of channel $c$ at gait
  phase $\varphi$.
- $\varphi_b$ — gait phase tagged on each tuple; $\mathrm{gait\%}$ is the
  heel-strike-aligned phase lookup.

`c ∈ {VL, BF, TA, GAL}` are the four Moreira surface channels. Each maps to a
**synergist group** `G_c` (a surface electrode detects a region), and the model
signal `ā_c` is the f0-weighted mean activation of that group:

| Channel | Group `G_c` |
|---|---|
| VL  | quadriceps (vasti lat/med/int + rectus femoris) — 9 |
| BF  | hamstrings (biceps femoris l/s + semitendinosus + semimembranosus) — 6 |
| TA  | dorsiflexors (tib. anterior + EDL + EHL + peroneus tertius) — 9 |
| GAL | triceps surae (gastroc lat/med + soleus ×2) — 4 |

### Why groups (the dead-unit pitfall)

`MuscleNN`'s output is `Tanh() → ReLU()` — a zero-gradient dead zone for any
pre-activation ≤ 0. The Σa² solver zeros whichever synergist is weaker
(R gastroc-lateral f0=606 loses to soleus f0=1793). Once a unit's output hits 0
the EMG-prior gradient vanishes → the muscle is **permanently dead** (observed
GAL, then TA on the next run). Mapping each channel to its synergist group makes
the comparison robust: the prior shapes the group's f0-weighted mean, the live
muscles (e.g. soleus) carry the signal, and a dead weak head is harmless.

## Parameters

Config block `emg_prior` in `data/rl_mass/configs/walk_moreira_emgprior.json`:

| Symbol | Key | Value | Meaning |
|---|---|---|---|
| λ | `lambda` | **1.0** | EMG-prior weight vs torque / Σa² |
| s | `scale` | **0.3** | EMG peak (=1) → target activation level |
| w_c | `muscle_lambda` | **[1,1,1,1]** | per-channel weight (VL,BF,TA,GAL) |
| — | `enabled` | true | gate (false ⇒ stock training, byte-identical) |
| — | `target_npz` | `emg_prior_V4.npz` | EMG target + group indices/weights |

Calibration constants (baked into `emg_prior_V4.npz` by `tools/build_emg_prior.py`):

| Symbol | Value | Source |
|---|---|---|
| φ_hs (heel-strike phase offset) | **0.254** | FK on the reference foot (Zeni 2008) |
| n_phase (lookup resolution) | **101** | gait-cycle grid |
| ê_c (EMG target) | (101, 4) peak-norm | Moreira V4 cohort mean (13 subj) |

Only **λ=1.0** and **s=0.3** are real tunables; both worked on first try (no sweep).

## Pipeline (prerequisite: fair kinematics)

The EMG comparison is only fair if the *motion* matches real gait. A kinematics
fairness gate (`tools/validate_walk_kinematics.py`) found hip/knee already match
but the stock `walk.bvh` ankle over-rotated 2× (49° vs real 25°) — the policy
faithfully tracked a **bad reference**. Fix: `tools/make_moreira_ref_bvh.py`
authored `walk_moreira.bvh`, replacing the ankle channel with the Moreira ankle
(ref|r| 0.63→0.96). The EMG prior is then trained on this corrected reference.

## Files & commands

| Purpose | Path |
|---|---|
| Final training config | `data/rl_mass/configs/walk_moreira_emgprior.json` |
| Final EMG validation (nn/max) | `data/rl_mass/configs/walk_vs_emg_emgprior_final.json` |
| Baseline (before) | `data/rl_mass/configs/walk_vs_emg_cohort_validate.json` |
| Corrected reference motion | `data/rl_mass/motion/walk_moreira.bvh` |
| EMG target + groups | `data/rl_mass/emg_moreira/emg_prior_V4.npz` |
| Build prior target | `tools/build_emg_prior.py` |
| EMG validation tool | `tools/validate_walk_vs_emg.py` |
| Paper figure | `tools/plot_walk_vs_emg.py` |

```bash
# 1. corrected reference + EMG target
uv run python tools/make_moreira_ref_bvh.py
uv run python tools/build_emg_prior.py

# 2. fine-tune from walk_max (corrected ref + EMG prior), GPU0
mkdir -p output/walk_emgprior/nn
cp data/rl_mass/checkpoints/walk_max{,_muscle}.pt output/walk_emgprior/nn/   # seed → max{,_muscle}.pt
PYTHONPATH=src CUDA_VISIBLE_DEVICES=0 uv run python -m rl_mass.train \
    -d data/rl_mass/configs/walk_moreira_emgprior.json \
    --resume output/walk_emgprior/nn/max.pt --num_envs 16 --max_iter 600

# 3. validate + figure
uv run python tools/validate_walk_vs_emg.py -c data/rl_mass/configs/walk_vs_emg_emgprior_final.json
uv run python tools/plot_walk_vs_emg.py \
    -i output/walk_vs_emg_emgprior_final/compare.csv \
    -o output/walk_vs_emg_emgprior_final/with_prior.pdf
```

## Caveats

- The match is **supervised**: the prior directly pulls activations toward EMG. It
  is the intended "inject the prior" mechanism, not an emergent unsupervised match.
  State this plainly in the paper.
- Magnitudes are not comparable (EMG is %MVC, activation is neural drive); the
  comparison is **shape/timing** in peak-normalised space.
- GAL is reported as the **triceps-surae group**, not the isolated lateral head,
  because the model's redundancy carries push-off on soleus.
- The corrected reference fixes only the ankle; the knee still slightly
  under-flexes via tracking loss (minor).
