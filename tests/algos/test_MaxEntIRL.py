import os
import pickle

import numpy as np
import torch as th
from scipy import io
import pytest
from imitation.data import rollout
from stable_baselines3.common.vec_env import VecNormalize

from algos.torch.MaxEntIRL.algorithm import MaxEntIRL, GuidedCostLearning, APIRL
from common.callbacks import SaveCallback
from common.util import make_env
from common.wrappers import *

env_op = 1
subj = "sub01"
env_name = "DiscretizedHuman"
env_id = f"{env_name}"


@pytest.fixture
def demo_dir():
    proj_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    return os.path.join(proj_path, "IRL", "demos")


@pytest.fixture
def expert(demo_dir):
    expert_dir = os.path.join(demo_dir, env_name, f"{subj}_1.pkl")
    with open(expert_dir, "rb") as f:
        expert_trajs = pickle.load(f)
    return expert_trajs


@pytest.fixture
def env(demo_dir):
    subpath = os.path.join(demo_dir, "HPC", f"{subj}_cropped", subj)
    init_states = []
    for i in range(1):
        for j in range(6):
            bsp = io.loadmat(subpath + f"i{i + 1}_{j}.mat")['bsp']
            init_states += [io.loadmat(subpath + f"i{i + 1}_{j}.mat")['state'][0, :4]]
    return make_env(f"{env_id}-v2", subpath=subpath, N=[11, 21, 21, 21], bsp=bsp)


@pytest.fixture
def eval_env(demo_dir):
    subpath = os.path.join(demo_dir, "HPC", f"{subj}_cropped", subj)
    init_states = []
    for i in range(1):
        for j in range(6):
            bsp = io.loadmat(subpath + f"i{i + 1}_{j}.mat")['bsp']
            init_states += [io.loadmat(subpath + f"i{i + 1}_{j}.mat")['state'][0, :4]]
    return make_env(f"{env_id}-v0", subpath=subpath, N=[11, 21, 21, 21], bsp=bsp, init_states=init_states)


@pytest.fixture
def learner(env, expert, eval_env):
    from imitation.util import logger
    from IRL.scripts.project_policies import def_policy
    logger.configure("tmp/log", format_strs=["stdout"])

    def feature_fn(x):
        # if len(x.shape) == 1:
        #     x = x.reshape(1, -1)
        # ft = th.zeros([x.shape[0], env_op], dtype=th.float32)
        # for i, row in enumerate(x):
        #     idx = int(row.item())
        #     ft[i, idx] = 1
        # return ft
        # return x
        return th.cat([x, x ** 2], dim=1)

    agent = def_policy("finitesoftqiter", env, device='cuda:3', verbose=1)

    return MaxEntIRL(
        env,
        eval_env=eval_env,
        agent=agent,
        feature_fn=feature_fn,
        expert_trajectories=expert,
        use_action_as_input=True,
        rew_arch=[],
        device=agent.device,
        env_kwargs={'vec_normalizer': None, 'reward_wrapper': ActionRewardWrapper},
        rew_kwargs={'type': 'ann', 'scale': 1, 'norm_coeff': 0.0, 'lr': 1e-2},
    )


def test_callback(learner):
    from stable_baselines3.common import callbacks
    from imitation.policies import serialize
    save_policy_callback = serialize.SavePolicyCallback(f"tmp/log", None)
    save_policy_callback = callbacks.EveryNTimesteps(int(1e3), save_policy_callback)
    save_reward_callback = SaveCallback(cycle=1, dirpath=f"tmp/log")
    learner.learn(
        total_iter=10,
        agent_learning_steps=10000,
        gradient_steps=1,
        n_episodes=8,
        max_agent_iter=1,
        callback=save_reward_callback.net_save
    )


def test_validity(learner, expert):
    learner.learn(
        total_iter=10,
        agent_learning_steps=5000,
        n_episodes=len(expert),
        max_agent_iter=1,
        min_agent_iter=1,
        max_gradient_steps=1,
        min_gradient_steps=1,
        early_stop=True,
    )


def test_GCL(env, expert, eval_env):
    from imitation.util import logger
    from IRL.scripts.project_policies import def_policy
    logger.configure("tmp/log", format_strs=["stdout"])

    def feature_fn(x):
        # return x
        return th.cat([x, x**2], dim=1)

    agent = def_policy("softqlearning", env, device='cpu', verbose=1)

    learner = GuidedCostLearning(
        env,
        agent=agent,
        feature_fn=feature_fn,
        expert_trajectories=expert,
        use_action_as_input=False,
        rew_arch=[],
        device='cpu',
        eval_env=eval_env,
        env_kwargs={'reward_wrapper': ActionRewardWrapper, "num_envs": 10},
        rew_kwargs={'type': 'ann'},
    )

    learner.learn(
        total_iter=50,
        agent_learning_steps=1e3,
        n_episodes=len(expert),
        max_agent_iter=1,
        min_agent_iter=1,
        max_gradient_steps=1,
        min_gradient_steps=1,
        early_stop=True,
    )


def test_state_visitation(env, expert, learner):
    from algos.tabular.viter import FiniteSoftQiter
    policy = FiniteSoftQiter(env, gamma=0.8, alpha=0.01, device='cpu')
    policy.learn(1000)
    learner.agent = policy
    Ds = learner.state_visitation()
    learner.get_whole_states_from_env()
    d1 = th.dot(Ds, learner.reward_net(learner.whole_state).flatten())
    d2, _ = learner.mean_transition_reward(expert)
    assert th.abs((d2 - d1) / d2).item() < 0.1
    print(th.abs((d2 - d1) / d2).item())
