"""Neural network models ported from MASS (external/MASS/python/Model.py).

Contains:
  - SimulationNN: Policy + value network for RL
  - MuscleNN: Muscle activation network

Architecture, initialization, and forward pass are kept identical to MASS.

Usage:
    from rl_mass.model import SimulationNN, MuscleNN

    policy = SimulationNN(num_states=136, num_actions=50)
    muscle_nn = MuscleNN(num_total_muscle_related_dofs=1098, num_dofs=50, num_muscles=284)
"""

import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

MultiVariateNormal = torch.distributions.Normal
temp = MultiVariateNormal.log_prob
MultiVariateNormal.log_prob = lambda self, val: temp(self, val).sum(-1, keepdim=True)

temp2 = MultiVariateNormal.entropy
MultiVariateNormal.entropy = lambda self: temp2(self).sum(-1)
MultiVariateNormal.mode = lambda self: self.mean

# Device selection with optional override via environment variable MASS_DEVICE
# Values: 'cpu' to force CPU, 'cuda' to force CUDA (if available), or unset for auto
_force = os.environ.get("MASS_DEVICE", "").lower()
if _force == "cpu":
    use_cuda = False
elif _force == "cuda":
    use_cuda = torch.cuda.is_available()
else:
    use_cuda = torch.cuda.is_available()

FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if use_cuda else torch.LongTensor
ByteTensor = torch.cuda.ByteTensor if use_cuda else torch.ByteTensor
Tensor = FloatTensor


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find("Linear") != -1:
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.zero_()


class MuscleNN(nn.Module):
    """Muscle activation network.

    Input: concatenation of muscle_tau (JtA, normalized) and tau (desired torques, normalized)
    Output: activation levels (non-negative, via Tanh -> ReLU)
    """

    def __init__(self, num_total_muscle_related_dofs, num_dofs, num_muscles):
        super(MuscleNN, self).__init__()
        self.num_total_muscle_related_dofs = num_total_muscle_related_dofs
        self.num_dofs = num_dofs
        self.num_muscles = num_muscles

        num_h1 = 1024
        num_h2 = 512
        num_h3 = 512
        self.fc = nn.Sequential(
            nn.Linear(num_total_muscle_related_dofs + num_dofs, num_h1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(num_h1, num_h2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(num_h2, num_h3),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(num_h3, num_muscles),
            nn.Tanh(),
            nn.ReLU(),
        )
        self.std_muscle_tau = torch.zeros(self.num_total_muscle_related_dofs)
        self.std_tau = torch.zeros(self.num_dofs)

        for i in range(self.num_total_muscle_related_dofs):
            self.std_muscle_tau[i] = 200.0

        for i in range(self.num_dofs):
            self.std_tau[i] = 200.0
        if use_cuda:
            self.std_tau = self.std_tau.cuda()
            self.std_muscle_tau = self.std_muscle_tau.cuda()
            self.cuda()
        self.fc.apply(weights_init)

    def forward(self, muscle_tau, tau):
        muscle_tau = muscle_tau / self.std_muscle_tau
        tau = tau / self.std_tau
        out = self.fc.forward(torch.cat([muscle_tau, tau], dim=1))
        return out

    def load(self, path):
        print("load muscle nn {}".format(path))
        if use_cuda:
            self.load_state_dict(torch.load(path))
            self.cuda()
        else:
            self.load_state_dict(torch.load(path, map_location="cpu"))
            self.cpu()

    def save(self, path):
        print("save muscle nn {}".format(path))
        torch.save(self.state_dict(), path)

    def get_activation(self, muscle_tau, tau):
        """Compute activation from numpy arrays.

        Args:
            muscle_tau: numpy array of shape (num_total_muscle_related_dofs,)
            tau: numpy array of shape (num_dofs,)

        Returns:
            numpy array of shape (num_muscles,), all non-negative.
        """
        mt = Tensor(muscle_tau.reshape(1, -1).astype(np.float32))
        t = Tensor(tau.reshape(1, -1).astype(np.float32))
        act = self.forward(mt, t)
        result = act.cpu().detach().numpy().squeeze()
        return np.ascontiguousarray(result, dtype=np.float64)


class SimulationNN(nn.Module):
    """Policy (actor-critic) network.

    Policy: 3-layer MLP -> Normal distribution (mean, learned log_std)
    Value: separate 3-layer MLP -> scalar
    """

    def __init__(self, num_states, num_actions):
        super(SimulationNN, self).__init__()

        num_h1 = 256
        num_h2 = 256

        self.p_fc1 = nn.Linear(num_states, num_h1)
        self.p_fc2 = nn.Linear(num_h1, num_h2)
        self.p_fc3 = nn.Linear(num_h2, num_actions)
        self.log_std = nn.Parameter(torch.zeros(num_actions))

        self.v_fc1 = nn.Linear(num_states, num_h1)
        self.v_fc2 = nn.Linear(num_h1, num_h2)
        self.v_fc3 = nn.Linear(num_h2, 1)

        torch.nn.init.xavier_uniform_(self.p_fc1.weight)
        torch.nn.init.xavier_uniform_(self.p_fc2.weight)
        torch.nn.init.xavier_uniform_(self.p_fc3.weight)

        self.p_fc1.bias.data.zero_()
        self.p_fc2.bias.data.zero_()
        self.p_fc3.bias.data.zero_()

        torch.nn.init.xavier_uniform_(self.v_fc1.weight)
        torch.nn.init.xavier_uniform_(self.v_fc2.weight)
        torch.nn.init.xavier_uniform_(self.v_fc3.weight)

        self.v_fc1.bias.data.zero_()
        self.v_fc2.bias.data.zero_()
        self.v_fc3.bias.data.zero_()

    def forward(self, x):
        p_out = F.relu(self.p_fc1(x))
        p_out = F.relu(self.p_fc2(p_out))
        p_out = self.p_fc3(p_out)

        p_out = MultiVariateNormal(p_out, self.log_std.exp())

        v_out = F.relu(self.v_fc1(x))
        v_out = F.relu(self.v_fc2(v_out))
        v_out = self.v_fc3(v_out)
        return p_out, v_out

    def load(self, path):
        print("load simulation nn {}".format(path))
        if use_cuda:
            self.load_state_dict(torch.load(path))
            self.cuda()
        else:
            self.load_state_dict(torch.load(path, map_location="cpu"))
            self.cpu()

    def save(self, path):
        print("save simulation nn {}".format(path))
        torch.save(self.state_dict(), path)

    def get_action(self, s):
        """Get deterministic action (mean) from numpy state.

        Args:
            s: numpy array of shape (num_states,) or (1, num_states)

        Returns:
            numpy array of shape (num_actions,)
        """
        ts = torch.tensor(s.astype(np.float32))
        if use_cuda:
            ts = ts.cuda()
        p, _ = self.forward(ts)
        result = p.loc.cpu().detach().numpy().squeeze()
        return np.ascontiguousarray(result, dtype=np.float64)

    def get_random_action(self, s):
        """Get stochastic action (sample) from numpy state.

        Args:
            s: numpy array of shape (num_states,) or (1, num_states)

        Returns:
            numpy array of shape (num_actions,)
        """
        ts = torch.tensor(s.astype(np.float32))
        p, _ = self.forward(ts)
        return p.sample().cpu().detach().numpy().squeeze()


# ---------------------------------------------------------------------------
# Quick test
# ---------------------------------------------------------------------------

def _test_models() -> None:
    """Verify output shapes of both networks."""
    print("=" * 60)
    print("Model Quick Test")
    print("=" * 60)

    # SimulationNN
    num_states = 136
    num_actions = 50
    sim_nn = SimulationNN(num_states, num_actions)
    if use_cuda:
        sim_nn.cuda()

    # Single sample
    s = np.random.randn(num_states).astype(np.float32)
    action = sim_nn.get_action(s)
    print(f"  SimulationNN({num_states}, {num_actions}):")
    print(f"    get_action shape: {action.shape} (expected ({num_actions},))")
    assert action.shape == (num_actions,), f"Expected ({num_actions},), got {action.shape}"

    # Batch
    batch_size = 8
    s_batch = torch.randn(batch_size, num_states)
    if use_cuda:
        s_batch = s_batch.cuda()
    p_dist, v = sim_nn(s_batch)
    print(f"    Batch policy loc shape: {p_dist.loc.shape} (expected ({batch_size}, {num_actions}))")
    print(f"    Batch value shape: {v.shape} (expected ({batch_size}, 1))")
    assert p_dist.loc.shape == (batch_size, num_actions)
    assert v.shape == (batch_size, 1)

    # MuscleNN
    num_total_related_dofs = 1098
    num_dofs = 50
    num_muscles = 284
    muscle_nn = MuscleNN(num_total_related_dofs, num_dofs, num_muscles)

    mt = np.random.randn(num_total_related_dofs).astype(np.float32)
    tau = np.random.randn(num_dofs).astype(np.float32)
    act = muscle_nn.get_activation(mt, tau)
    print(f"\n  MuscleNN({num_total_related_dofs}, {num_dofs}, {num_muscles}):")
    print(f"    get_activation shape: {act.shape} (expected ({num_muscles},))")
    assert act.shape == (num_muscles,), f"Expected ({num_muscles},), got {act.shape}"

    # Non-negativity (Tanh -> ReLU)
    assert np.all(act >= 0), f"Activations should be non-negative, min={act.min()}"
    print(f"    All activations >= 0: True (min={act.min():.6f}, max={act.max():.6f})")

    # Batch
    mt_batch = torch.randn(batch_size, num_total_related_dofs)
    tau_batch = torch.randn(batch_size, num_dofs)
    if use_cuda:
        mt_batch = mt_batch.cuda()
        tau_batch = tau_batch.cuda()
    act_batch = muscle_nn(mt_batch, tau_batch)
    print(f"    Batch activation shape: {act_batch.shape} (expected ({batch_size}, {num_muscles}))")
    assert act_batch.shape == (batch_size, num_muscles)
    assert torch.all(act_batch >= 0), "Batch activations should be non-negative"

    print("\n" + "=" * 60)
    print("Model Quick Test PASSED")
    print("=" * 60)


if __name__ == "__main__":
    _test_models()
