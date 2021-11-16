import os
import pickle
import torch as th

import pytest
from imitation.data import rollout
from stable_baselines3.common.vec_env import VecNormalize

from algos.torch.MaxEntIRL.algorithm import MaxEntIRL, GuidedCostLearning
from common.callbacks import SaveCallback
from common.util import make_env
from common.wrappers import *


@pytest.fixture
def demo_dir():
    return os.path.abspath(os.path.join("..", "..", "IRL", "demos"))


@pytest.fixture
def expert(demo_dir):
    expert_dir = os.path.join(demo_dir, "2DTarget", "softqlearning_disc.pkl")
    with open(expert_dir, "rb") as f:
        expert_trajs = pickle.load(f)
    return expert_trajs


@pytest.fixture
def env(demo_dir):
    subpath = os.path.join(demo_dir, "HPC", "sub01", "sub01")
    return make_env("2DTarget_disc-v2", subpath=subpath)


@pytest.fixture
def eval_env(demo_dir):
    subpath = os.path.join(demo_dir, "HPC", "sub01", "sub01")
    return make_env("2DTarget_disc-v0", subpath=subpath)


@pytest.fixture
def learner(env, expert, eval_env):
    from imitation.util import logger
    from IRL.scripts.project_policies import def_policy
    logger.configure("tmp/log", format_strs=["stdout"])

    def feature_fn(x):
        return th.cat([x, x**2], dim=1)

    agent = def_policy("softqlearning", env, device='cpu', verbose=1)

    return MaxEntIRL(
        env,
        eval_env=eval_env,
        agent=agent,
        feature_fn=feature_fn,
        expert_trajectories=expert,
        use_action_as_input=True,
        rew_arch=[],
        device=agent.device,
        env_kwargs={'vec_normalizer': None, 'reward_wrapper': RewardWrapper},
        rew_kwargs={'type': 'ann', 'scale': 1, 'alpha': 0.1},
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
        total_iter=50,
        agent_learning_steps=1000,
        n_episodes=len(expert),
        max_agent_iter=1,
        min_agent_iter=1,
        max_gradient_steps=200,
        min_gradient_steps=100,
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
        env_kwargs={'reward_wrapper': RewardWrapper},
        rew_kwargs={'type': 'ann'},
    )

    learner.learn(
        total_iter=50,
        agent_learning_steps=1e5,
        n_episodes=len(expert),
        max_agent_iter=1,
        min_agent_iter=1,
        max_gradient_steps=1200,
        min_gradient_steps=500,
        early_stop=True,
    )
