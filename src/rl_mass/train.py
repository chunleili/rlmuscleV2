"""PPO training loop for MASS-MuJoCo.

Ported from external/MASS/python/main.py.
All hyperparameters and logic kept identical to the original.

Usage:
    PYTHONPATH=. uv run python rl_mass/train.py -d data/rl_mass/config.json --outdir output/walk
    PYTHONPATH=. uv run python rl_mass/train.py -d data/rl_mass/config.json --outdir output/walk --resume current
    PYTHONPATH=. uv run python rl_mass/train.py -d data/rl_mass/config.json --resume output/walk/nn/max.pt
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import time
from collections import deque, namedtuple
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.optim as optim

from rl_mass.model import MuscleNN, SimulationNN, Tensor, use_cuda
from rl_mass.vec_env import MassVecEnv, make_vec_env

DEFAULT_REWARD_CSV_PATH = "output/rl_mass/train_rewards.csv"
DEFAULT_NN_DIR = "nn"

REWARD_CSV_COLUMNS = (
    "iter",
    "elapsed_s",
    "num_episode",
    "num_tuple",
    "avg_return",
    "avg_reward_per_transition",
    "avg_episode_steps",
    "avg_episode_seconds",
    "max_episode_steps",
    "max_episode_seconds",
    "max_return",
    "loss_actor",
    "loss_critic",
    "loss_muscle",
)


def summarize_episode_lengths(
    episode_lengths: list[int], control_hz: int
) -> dict[str, float | int]:
    """Aggregate episode duration stats in control steps and seconds."""
    if not episode_lengths:
        return {
            "avg_episode_steps": 0.0,
            "avg_episode_seconds": 0.0,
            "max_episode_steps": 0,
            "max_episode_seconds": 0.0,
        }

    avg_steps = float(np.mean(episode_lengths))
    max_steps = int(np.max(episode_lengths))
    hz = float(control_hz)
    return {
        "avg_episode_steps": avg_steps,
        "avg_episode_seconds": avg_steps / hz,
        "max_episode_steps": max_steps,
        "max_episode_seconds": max_steps / hz,
    }


def ensure_reward_csv_header(path: str) -> bool:
    """Return True when the CSV header should be written for this path."""
    if not os.path.exists(path):
        return True

    with open(path, "r") as f:
        header = f.readline().strip()

    expected = ",".join(REWARD_CSV_COLUMNS)
    if header != expected:
        raise RuntimeError(
            f"Reward CSV schema mismatch for {path}. "
            "Move/remove the old CSV before resuming training with episode "
            "duration stats enabled."
        )
    return False


def normalize_checkpoint_prefix(path: str) -> str:
    """Strip checkpoint file suffixes and return the shared prefix path."""
    if path.endswith("_muscle.pt"):
        return path[: -len("_muscle.pt")]
    if path.endswith(".pt"):
        return path[:-3]
    return path


def infer_resume_iter(path: str) -> int:
    """Read the last recorded evaluation iteration from the reward CSV."""
    if not os.path.exists(path):
        return 0

    ensure_reward_csv_header(path)
    last_iter = 0
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            last_iter = int(row["iter"])
    return last_iter


def resolve_run_paths(
    *,
    outdir: Optional[str] = None,
    resume: Optional[str] = None,
) -> dict[str, Optional[str] | int]:
    """Resolve output paths and optional resume metadata from CLI inputs."""
    if outdir:
        base_dir = Path(outdir)
        resolved_reward_csv = str(base_dir / "train_rewards.csv")
        resolved_nn_dir = str(base_dir / "nn")
    else:
        resolved_reward_csv = DEFAULT_REWARD_CSV_PATH
        resolved_nn_dir = DEFAULT_NN_DIR

    resume_prefix = None
    resume_iter = 0
    if resume:
        normalized = normalize_checkpoint_prefix(resume)
        normalized_path = Path(normalized)
        if normalized_path.parent == Path(".") and not normalized_path.is_absolute():
            resume_prefix = str(Path(resolved_nn_dir) / normalized)
        else:
            resume_prefix = str(normalized_path)
            resolved_nn_dir = str(normalized_path.parent)
            if normalized_path.parent.name == "nn":
                resolved_reward_csv = str(
                    normalized_path.parent.parent / "train_rewards.csv"
                )

        resume_iter = infer_resume_iter(resolved_reward_csv)

    return {
        "reward_csv_path": resolved_reward_csv,
        "nn_dir": resolved_nn_dir,
        "resume_prefix": resume_prefix,
        "resume_iter": resume_iter,
    }

# ---------------------------------------------------------------------------
# Data structures (identical to MASS main.py)
# ---------------------------------------------------------------------------

Episode = namedtuple("Episode", ("s", "a", "r", "value", "logprob"))


class EpisodeBuffer:
    def __init__(self):
        self.data: list[Episode] = []

    def Push(self, *args):
        self.data.append(Episode(*args))

    def Pop(self):
        self.data.pop()

    def GetData(self):
        return self.data


Transition = namedtuple("Transition", ("s", "a", "logprob", "TD", "GAE"))


class ReplayBuffer:
    def __init__(self, buff_size: int = 10000):
        super().__init__()
        self.buffer: deque = deque(maxlen=buff_size)

    def Push(self, *args):
        self.buffer.append(Transition(*args))

    def Clear(self):
        self.buffer.clear()


# ---------------------------------------------------------------------------
# PPO
# ---------------------------------------------------------------------------


class PPO:
    """PPO trainer, ported from MASS main.py with MassEnv backend."""

    def __init__(
        self,
        config_path: str,
        num_envs: int = 16,
        vec_env_mode: str = "subproc",
        seed: Optional[int] = None,
        packed_muscle_loop: bool = True,
        reward_csv_path: Optional[str] = None,
        nn_dir: str = "nn",
    ):
        self.reward_csv_path = reward_csv_path
        self.nn_dir = nn_dir
        if seed is None:
            seed = int(time.time())
        np.random.seed(seed=seed)

        # Create vectorized environment
        self.num_envs = num_envs
        self.vec_env_mode = vec_env_mode
        self.env: MassVecEnv = make_vec_env(
            config_path, num_envs=num_envs, mode=vec_env_mode, seed=seed
        )
        # Packed path enables worker-internal muscle inner loop (P2).
        # Only meaningful in subproc mode with muscles on; Serial is always batch-API.
        self._packed_path = (
            packed_muscle_loop and vec_env_mode == "subproc" and self.env.use_muscle
        )

        self.use_muscle: bool = self.env.use_muscle
        self.num_state: int = self.env.num_state
        self.num_action: int = self.env.num_action
        self.num_muscles: int = self.env.num_muscles

        self.num_epochs = 10
        self.num_epochs_muscle = 3
        self.num_evaluation = 0
        self.num_tuple_so_far = 0
        self.num_episode = 0
        self.num_tuple = 0
        self.avg_episode_steps = 0.0
        self.avg_episode_seconds = 0.0
        self.max_episode_steps = 0
        self.max_episode_seconds = 0.0
        self.num_simulation_Hz: int = self.env.simulation_hz
        self.num_control_Hz: int = self.env.control_hz
        self.num_simulation_per_control: int = self.num_simulation_Hz // self.num_control_Hz

        self.gamma = 0.99
        self.lb = 0.99

        self.buffer_size = 2048
        self.batch_size = 128
        self.muscle_batch_size = 128
        self.replay_buffer = ReplayBuffer(30000)
        self.muscle_buffer: dict = {}

        self.model = SimulationNN(self.num_state, self.num_action)

        self.num_total_muscle_related_dofs: int = self.env.num_total_related_dofs
        self.muscle_model = MuscleNN(
            self.num_total_muscle_related_dofs, self.num_action, self.num_muscles
        )
        if use_cuda:
            self.model.cuda()
            self.muscle_model.cuda()

        self.default_learning_rate = 1e-4
        self.default_clip_ratio = 0.2
        self.learning_rate = self.default_learning_rate
        self.clip_ratio = self.default_clip_ratio
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.optimizer_muscle = optim.Adam(
            self.muscle_model.parameters(), lr=self.learning_rate
        )
        self.max_iteration = 50000

        # Optional EMG activation prior (config-gated; default off → behaviour
        # identical to stock training). Pulls the 4 Moreira surface muscles'
        # activations toward the peak-normalised EMG envelope, indexed by gait
        # phase. See tools/build_emg_prior.py for the target .npz.
        self.emg_prior = None
        try:
            with open(config_path) as _f:
                _ep = json.load(_f).get("emg_prior")
        except (OSError, ValueError):
            _ep = None
        if _ep and _ep.get("enabled", False) and self.use_muscle:
            _d = np.load(_ep["target_npz"])
            _tgt = torch.FloatTensor(_d["target"])          # (P, 4) peak-norm EMG
            _idx = torch.LongTensor(_d["snu_idx"])          # (4,) primary SNU cols
            # Group aggregation: each EMG channel maps to one or more SNU muscles
            # (a surface electrode detects a region). The model signal is the
            # f0-weighted mean activation of the group. Singletons → weight 1.
            _gidx = torch.LongTensor(_d["group_idx"])       # (4, G) SNU indices
            _gw   = torch.FloatTensor(_d["group_w"])        # (4, G) weights (rows sum 1)
            # Per-muscle weight: boosts muscles the torque solver tends to zero
            # (e.g. R gastroc-lateral, which dies in the Tanh→ReLU output's dead
            # zone under equal weighting — see model.MuscleNN). Default all 1.
            _mw = np.asarray(_ep.get("muscle_lambda", [1.0] * len(_idx)), dtype=np.float32)
            _mw_t = torch.FloatTensor(_mw)
            if use_cuda:
                _tgt = _tgt.cuda(); _idx = _idx.cuda(); _mw_t = _mw_t.cuda()
                _gidx = _gidx.cuda(); _gw = _gw.cuda()
            self.emg_prior = {
                "lam":   float(_ep.get("lambda", 1.0)),
                "scale": float(_ep.get("scale", 0.3)),       # EMG-peak → activation level
                "n_phase": int(_tgt.shape[0]),
                "snu_idx": _idx,
                "group_idx": _gidx,
                "group_w":   _gw,
                "target":  _tgt,
                "mw":      _mw_t,                            # (4,) per-channel weight
            }
            print(f"[emg_prior] enabled: lambda={self.emg_prior['lam']} "
                  f"scale={self.emg_prior['scale']} muscles={_d['snu_idx'].tolist()} "
                  f"muscle_lambda={_mw.tolist()}")

        self.w_entropy = -0.001

        self.loss_actor = 0.0
        self.loss_critic = 0.0
        self.loss_muscle = 0.0
        self.rewards: list[float] = []
        self.sum_return = 0.0
        self.max_return = -1.0
        self.max_return_epoch = 1
        self.tic = time.time()

        self.episodes: list[EpisodeBuffer] = [
            EpisodeBuffer() for _ in range(self.num_envs)
        ]

        # Reset all envs with RSI
        self.env.reset(rsi=True)

    # ------------------------------------------------------------------
    # Save / Load
    # ------------------------------------------------------------------

    def SaveModel(self, nn_dir: str = "nn"):
        self.model.save(os.path.join(nn_dir, "current.pt"))
        self.muscle_model.save(os.path.join(nn_dir, "current_muscle.pt"))

        if self.max_return_epoch == self.num_evaluation:
            self.model.save(os.path.join(nn_dir, "max.pt"))
            self.muscle_model.save(os.path.join(nn_dir, "max_muscle.pt"))
        if self.num_evaluation % 100 == 0:
            tag = str(self.num_evaluation // 100)
            self.model.save(os.path.join(nn_dir, f"{tag}.pt"))
            self.muscle_model.save(os.path.join(nn_dir, f"{tag}_muscle.pt"))

    def LoadModel(self, path: str, nn_dir: str = "nn"):
        self.model.load(os.path.join(nn_dir, path + ".pt"))
        self.muscle_model.load(os.path.join(nn_dir, path + "_muscle.pt"))

    # ------------------------------------------------------------------
    # GenerateTransitions  (ported from main.py lines 173-232)
    # ------------------------------------------------------------------

    def GenerateTransitions(self):
        self.total_episodes: list[EpisodeBuffer] = []
        # Packed path: broadcast current MuscleNN weights to workers once per iter.
        if self._packed_path:
            self.env.update_muscle_nn(self.muscle_model.state_dict())

        states = self.env.get_states()  # (num_envs, num_state)
        local_step = 0
        counter = 0

        while True:
            counter += 1
            if counter % 10 == 0:
                print(f"SIM : {local_step}", end="\r")

            # Forward pass through policy
            states_tensor = Tensor(states.astype(np.float32))
            a_dist, v = self.model(states_tensor)

            actions = a_dist.sample().cpu().detach().numpy()
            logprobs = (
                a_dist.log_prob(Tensor(actions.astype(np.float32)))
                .cpu()
                .detach()
                .numpy()
                .reshape(-1)
            )
            values = v.cpu().detach().numpy().reshape(-1)

            # Step environments
            if self._packed_path:
                # P2: each worker runs the full control step internally
                # (set_action → get_muscle_torques → muscle inner loop → substeps).
                results = self.env.run_control_step(actions)
                next_states = np.array([r[0] for r in results])
                step_rewards = np.array([r[1] for r in results])
                step_eoes = [r[2] for r in results]
            else:
                # P0/P1 batch-API path: main process drives MuscleNN, workers step.
                self.env.set_actions(actions)
                if self.use_muscle:
                    mt = Tensor(self.env.get_muscle_torques())
                    for _ in range(self.num_simulation_per_control // 2):
                        dt = Tensor(self.env.get_desired_torques())
                        activations = self.muscle_model(mt, dt).cpu().detach().numpy()
                        self.env.set_activation_levels(activations)
                        self.env.steps(2)
                else:
                    self.env.steps_at_once()
                next_states = None  # computed after episode bookkeeping
                step_rewards = None
                step_eoes = None

            # Check for episode end, collect rewards
            for j in range(self.num_envs):
                nan_occur = False
                terminated_state = True

                if (
                    np.any(np.isnan(states[j]))
                    or np.any(np.isnan(actions[j]))
                    or np.any(np.isnan(values[j]))
                    or np.any(np.isnan(logprobs[j]))
                ):
                    nan_occur = True
                else:
                    eoe = step_eoes[j] if step_eoes is not None else self.env.is_end_of_episode(j)
                    if not eoe:
                        terminated_state = False
                        reward = (
                            step_rewards[j] if step_rewards is not None
                            else self.env.get_reward(j)
                        )
                        self.episodes[j].Push(
                            states[j], actions[j], reward, values[j], logprobs[j]
                        )
                        local_step += 1

                if terminated_state or nan_occur:
                    if nan_occur:
                        self.episodes[j].Pop()
                    self.total_episodes.append(self.episodes[j])
                    self.episodes[j] = EpisodeBuffer()
                    new_state = self.env.reset(rsi=True, idx=j)
                    if self._packed_path and new_state is not None:
                        next_states[j] = new_state

            if local_step >= self.buffer_size:
                break

            # Next states: packed path already has them (with reset fixups above);
            # batch path gathers in a single round-trip.
            if self._packed_path:
                states = next_states
            else:
                states = self.env.get_states()

    # ------------------------------------------------------------------
    # ComputeTDandGAE
    # ------------------------------------------------------------------

    def ComputeTDandGAE(self):
        self.replay_buffer.Clear()
        self.muscle_buffer = {}
        self.sum_return = 0.0

        for epi in self.total_episodes:
            data = epi.GetData()
            size = len(data)
            if size == 0:
                continue
            states, actions, rewards, values, logprobs = zip(*data)

            values = np.concatenate((values, np.zeros(1)), axis=0)
            advantages = np.zeros(size)
            ad_t = 0

            epi_return = 0.0
            for i in reversed(range(size)):
                epi_return += rewards[i]
                delta = rewards[i] + values[i + 1] * self.gamma - values[i]
                ad_t = delta + self.gamma * self.lb * ad_t
                advantages[i] = ad_t
            self.sum_return += epi_return
            TD = values[:size] + advantages

            for i in range(size):
                self.replay_buffer.Push(
                    states[i], actions[i], logprobs[i], TD[i], advantages[i]
                )

        self.num_episode = len(self.total_episodes)
        self.num_tuple = len(self.replay_buffer.buffer)
        episode_lengths = [len(epi.GetData()) for epi in self.total_episodes]
        episode_stats = summarize_episode_lengths(
            episode_lengths, control_hz=self.num_control_Hz
        )
        self.avg_episode_steps = float(episode_stats["avg_episode_steps"])
        self.avg_episode_seconds = float(episode_stats["avg_episode_seconds"])
        self.max_episode_steps = int(episode_stats["max_episode_steps"])
        self.max_episode_seconds = float(episode_stats["max_episode_seconds"])
        print(f"SIM : {self.num_tuple}")
        self.num_tuple_so_far += self.num_tuple

        # Collect muscle tuples from all envs
        all_tuples = self.env.gather_muscle_tuples()

        if len(all_tuples) > 0:
            self.muscle_buffer["JtA"] = np.stack(
                [t.JtA for t in all_tuples]
            )
            self.muscle_buffer["TauDes"] = np.stack(
                [t.tau_des for t in all_tuples]
            )
            self.muscle_buffer["L"] = np.stack(
                [t.L for t in all_tuples]
            )
            self.muscle_buffer["b"] = np.stack(
                [t.b for t in all_tuples]
            )
            self.muscle_buffer["phase"] = np.array(
                [t.phase for t in all_tuples], dtype=np.float32
            )
            print(self.muscle_buffer["JtA"].shape)
        else:
            # Edge case: no muscle tuples collected
            print("WARNING: No muscle tuples collected")

    # ------------------------------------------------------------------
    # OptimizeSimulationNN
    # ------------------------------------------------------------------

    def OptimizeSimulationNN(self):
        all_transitions = list(self.replay_buffer.buffer)
        for j in range(self.num_epochs):
            np.random.shuffle(all_transitions)
            for i in range(len(all_transitions) // self.batch_size):
                transitions = all_transitions[
                    i * self.batch_size : (i + 1) * self.batch_size
                ]
                batch = Transition(*zip(*transitions))

                stack_s = np.vstack(batch.s).astype(np.float32)
                stack_a = np.vstack(batch.a).astype(np.float32)
                stack_lp = np.vstack(batch.logprob).astype(np.float32)
                stack_td = np.vstack(batch.TD).astype(np.float32)
                stack_gae = np.vstack(batch.GAE).astype(np.float32)

                a_dist, v = self.model(Tensor(stack_s))

                # Critic loss
                loss_critic = ((v - Tensor(stack_td)).pow(2)).mean()

                # Actor loss
                ratio = torch.exp(
                    a_dist.log_prob(Tensor(stack_a)) - Tensor(stack_lp)
                )
                stack_gae = (stack_gae - stack_gae.mean()) / (
                    stack_gae.std() + 1e-5
                )
                stack_gae = Tensor(stack_gae)
                surrogate1 = ratio * stack_gae
                surrogate2 = (
                    torch.clamp(
                        ratio, min=1.0 - self.clip_ratio, max=1.0 + self.clip_ratio
                    )
                    * stack_gae
                )
                loss_actor = -torch.min(surrogate1, surrogate2).mean()

                # Entropy loss
                loss_entropy = -self.w_entropy * a_dist.entropy().mean()

                self.loss_actor = loss_actor.cpu().detach().numpy().tolist()
                self.loss_critic = loss_critic.cpu().detach().numpy().tolist()

                loss = loss_actor + loss_entropy + loss_critic

                self.optimizer.zero_grad()
                loss.backward(retain_graph=True)
                for param in self.model.parameters():
                    if param.grad is not None:
                        param.grad.data.clamp_(-0.5, 0.5)
                self.optimizer.step()
            print(
                f"Optimizing sim nn : {j + 1}/{self.num_epochs}", end="\r"
            )
        print("")

    # ------------------------------------------------------------------
    # OptimizeMuscleNN
    # ------------------------------------------------------------------

    def generate_shuffle_indices(self, batch_size, minibatch_size):
        n = batch_size
        m = minibatch_size
        p = np.random.permutation(n)

        r = m - n % m
        # MASS original: `if r > 0` — when n%m==0, r=m and it still pads
        # one extra minibatch of random samples. Replicate this exactly.
        if r > 0:
            p = np.hstack([p, np.random.randint(0, n, r)])

        p = p.reshape(-1, m)
        return p

    def OptimizeMuscleNN(self):
        if "JtA" not in self.muscle_buffer:
            print("No muscle tuples to optimize")
            return

        for j in range(self.num_epochs_muscle):
            minibatches = self.generate_shuffle_indices(
                self.muscle_buffer["JtA"].shape[0], self.muscle_batch_size
            )

            for minibatch in minibatches:
                stack_JtA = self.muscle_buffer["JtA"][minibatch].astype(np.float32)
                stack_tau_des = self.muscle_buffer["TauDes"][minibatch].astype(
                    np.float32
                )
                stack_L = self.muscle_buffer["L"][minibatch].astype(np.float32)
                stack_L = stack_L.reshape(
                    self.muscle_batch_size, self.num_action, self.num_muscles
                )
                stack_b = self.muscle_buffer["b"][minibatch].astype(np.float32)

                stack_JtA = Tensor(stack_JtA)
                stack_tau_des = Tensor(stack_tau_des)
                stack_L = Tensor(stack_L)
                stack_b = Tensor(stack_b)

                activation = self.muscle_model(stack_JtA, stack_tau_des)
                tau = torch.einsum("ijk,ik->ij", (stack_L, activation)) + stack_b

                loss_reg = (activation).pow(2).mean()
                loss_target = (((tau - stack_tau_des) / 100.0).pow(2)).mean()

                loss = 0.01 * loss_reg + loss_target

                # EMG activation prior: pull the 4 surface muscles toward the
                # peak-normalised EMG envelope at this minibatch's gait phases.
                if self.emg_prior is not None:
                    ep = self.emg_prior
                    ph = self.muscle_buffer["phase"][minibatch]      # (B,) in [0,1)
                    pidx = (ph * ep["n_phase"]).astype(np.int64) % ep["n_phase"]
                    tgt = ep["target"][torch.as_tensor(pidx, device=ep["target"].device)]
                    # f0-weighted group activation per EMG channel: (B,4,G)*(4,G)→(B,4)
                    a_grp = activation[:, ep["group_idx"]]           # (B, 4, G)
                    a_sel = (a_grp * ep["group_w"].unsqueeze(0)).sum(dim=2)  # (B, 4)
                    loss_emg = (ep["mw"] * (a_sel - ep["scale"] * tgt).pow(2)).mean()
                    loss = loss + ep["lam"] * loss_emg

                self.optimizer_muscle.zero_grad()
                loss.backward(retain_graph=True)
                for param in self.muscle_model.parameters():
                    if param.grad is not None:
                        param.grad.data.clamp_(-0.5, 0.5)
                self.optimizer_muscle.step()

            print(
                f"Optimizing muscle nn : {j + 1}/{self.num_epochs_muscle}",
                end="\r",
            )
        self.loss_muscle = loss.cpu().detach().numpy().tolist()
        print("")

    # ------------------------------------------------------------------
    # OptimizeModel
    # ------------------------------------------------------------------

    def OptimizeModel(self):
        self.ComputeTDandGAE()
        self.OptimizeSimulationNN()
        if self.use_muscle:
            self.OptimizeMuscleNN()

    # ------------------------------------------------------------------
    # Train
    # ------------------------------------------------------------------

    def Train(self):
        self.GenerateTransitions()
        self.OptimizeModel()

    # ------------------------------------------------------------------
    # Evaluate
    # ------------------------------------------------------------------

    def Evaluate(self) -> np.ndarray:
        self.num_evaluation += 1
        elapsed = time.time() - self.tic
        h = int(elapsed // 3600.0)
        m = int(elapsed // 60.0) - h * 60
        s = int(elapsed) - h * 3600 - m * 60

        if self.num_episode == 0:
            self.num_episode = 1
        if self.num_tuple == 0:
            self.num_tuple = 1
        if self.max_return < self.sum_return / self.num_episode:
            self.max_return = self.sum_return / self.num_episode
            self.max_return_epoch = self.num_evaluation

        print(f"# {self.num_evaluation} === {h}h:{m}m:{s}s ===")
        print(f"||Loss Actor               : {self.loss_actor:.4f}")
        print(f"||Loss Critic              : {self.loss_critic:.4f}")
        print(f"||Loss Muscle              : {self.loss_muscle:.4f}")
        print(
            f"||Noise                    : {self.model.log_std.exp().mean():.3f}"
        )
        print(f"||Num Transition So far    : {self.num_tuple_so_far}")
        print(f"||Num Transition           : {self.num_tuple}")
        print(f"||Num Episode              : {self.num_episode}")
        print(
            f"||Avg Step per episode     : {self.avg_episode_steps:.1f}"
        )
        print(
            f"||Avg Time per episode     : {self.avg_episode_seconds:.3f}s"
        )
        print(
            f"||Max Step per episode     : {self.max_episode_steps}"
        )
        print(
            f"||Max Time per episode     : {self.max_episode_seconds:.3f}s"
        )
        print(
            f"||Avg Return per episode   : {self.sum_return / self.num_episode:.3f}"
        )
        print(
            f"||Avg Reward per transition: {self.sum_return / self.num_tuple:.3f}"
        )
        print(
            f"||Max Avg Return So far    : {self.max_return:.3f} at #{self.max_return_epoch}"
        )
        avg_return = self.sum_return / self.num_episode
        self.rewards.append(avg_return)

        # Persist reward curve to CSV (append) — one row per evaluation.
        if self.reward_csv_path is not None:
            write_header = ensure_reward_csv_header(self.reward_csv_path)
            os.makedirs(os.path.dirname(self.reward_csv_path) or ".", exist_ok=True)
            with open(self.reward_csv_path, "a") as f:
                if write_header:
                    f.write(",".join(REWARD_CSV_COLUMNS) + "\n")
                f.write(
                    f"{self.num_evaluation},{int(elapsed)},{self.num_episode},"
                    f"{self.num_tuple},{avg_return:.6f},"
                    f"{self.sum_return / self.num_tuple:.6f},"
                    f"{self.avg_episode_steps:.6f},{self.avg_episode_seconds:.6f},"
                    f"{self.max_episode_steps},{self.max_episode_seconds:.6f},"
                    f"{self.max_return:.6f},"
                    f"{self.loss_actor:.6f},{self.loss_critic:.6f},{self.loss_muscle:.6f}\n"
                )

        self.SaveModel(nn_dir=self.nn_dir)

        print("=============================================")
        return np.array(self.rewards)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PPO training for MASS-MuJoCo")
    parser.add_argument("-d", "--meta", help="Config JSON file", default="data/rl_mass/config.json")
    parser.add_argument(
        "--outdir",
        type=str,
        default=None,
        help="Output directory; uses <outdir>/train_rewards.csv and <outdir>/nn/",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Checkpoint prefix or .pt path to resume from; auto-infers iter from reward CSV",
    )
    parser.add_argument("--num_envs", type=int, default=16, help="Parallel env count")
    parser.add_argument(
        "--vec_env_mode",
        type=str,
        default="subproc",
        choices=["serial", "subproc"],
        help="VecEnv backend: 'serial' (in-process) or 'subproc' (multi-process, uses packed "
             "run_control_step when muscles are on)",
    )
    parser.add_argument("--seed", type=int, default=None, help="Random seed (default: time-based)")
    parser.add_argument("--max_iter", type=int, default=None, help="Override max iterations (debug)")
    parser.add_argument(
        "--no_packed",
        action="store_true",
        help="Disable worker-internal muscle loop (P1 naive batch-API path, for benchmarking)",
    )
    args = parser.parse_args()

    run_paths = resolve_run_paths(
        outdir=args.outdir,
        resume=args.resume,
    )
    reward_csv_path = run_paths["reward_csv_path"]
    nn_dir = str(run_paths["nn_dir"])
    resume_prefix = run_paths["resume_prefix"]
    resume_iter = int(run_paths["resume_iter"])

    ppo = PPO(
        args.meta,
        num_envs=args.num_envs,
        vec_env_mode=args.vec_env_mode,
        seed=args.seed,
        packed_muscle_loop=not args.no_packed,
        reward_csv_path=reward_csv_path if reward_csv_path else None,
        nn_dir=nn_dir,
    )
    if args.max_iter is not None:
        ppo.max_iteration = args.max_iter + 5  # loop runs max_iteration - 5

    if not os.path.exists(nn_dir):
        os.makedirs(nn_dir)

    if resume_prefix is not None:
        ppo.LoadModel(Path(resume_prefix).name, nn_dir=nn_dir)
    else:
        ppo.SaveModel(nn_dir=nn_dir)

    if resume_iter > 0:
        ppo.num_evaluation = resume_iter
        ppo.max_return_epoch = resume_iter

    print(
        f"num states: {ppo.num_state}, num actions: {ppo.num_action}, "
        f"num muscles: {ppo.num_muscles}, "
        f"num total muscle related dofs: {ppo.num_total_muscle_related_dofs}"
    )
    print(
        f"simulation_hz: {ppo.num_simulation_Hz}, control_hz: {ppo.num_control_Hz}, "
        f"num_sim_per_control: {ppo.num_simulation_per_control}"
    )
    print(f"use_muscle: {ppo.use_muscle}")

    for i in range(ppo.max_iteration - 5):
        ppo.Train()
        rewards = ppo.Evaluate()
