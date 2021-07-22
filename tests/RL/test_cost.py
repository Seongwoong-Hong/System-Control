import os
import pytest
import numpy as np
import torch as th
from scipy import io

from imitation.data.rollout import make_sample_until, generate_trajectories, flatten_trajectories

from algos.torch.ppo import PPO
from IRL.scripts.project_policies import def_policy
from common.util import make_env


@pytest.fixture
def rl_path():
    return os.path.abspath(os.path.join("..", "..", "RL"))


def test_expt_cost(rl_path):
    def expt_fn(inp):
        return inp[:, :2].square().sum() + inp[:, 2:4].square().sum() + 1e-5 * (200 * inp[:, 4:]).square().sum()
    env_type = "IDP"
    env_id = "IDP_custom"
    proj_path = os.path.join(rl_path, env_type, "tmp", "log", env_id, "ppo", "policies_3")
    pltqs = []
    if env_type == "HPC":
        for i in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]:
            file = os.path.join("..", "..", "IRL", "demos", env_type, "sub01", f"sub01i{i+1}.mat")
            pltqs += [io.loadmat(file)['pltq']]
        test_len = len(pltqs)
    venv = make_env(f"{env_type}_custom-v1", use_vec_env=True, num_envs=1, pltqs=pltqs)
    sample_until = make_sample_until(n_timesteps=None, n_episodes=10)
    i = 3.25e6
    # agent = PPO.load(os.path.join(proj_path, f"{int(i):012d}", "model.pkl"), device='cpu')
    agent = def_policy("IDP", venv)
    venv.reset()
    agent_trajs = generate_trajectories(agent, venv, sample_until=sample_until, deterministic_policy=False)
    agent_trans = flatten_trajectories(agent_trajs)
    th_input = th.from_numpy(np.concatenate([agent_trans.obs, agent_trans.acts], axis=1))
    print("Cost:", expt_fn(th_input).item() / 10)
