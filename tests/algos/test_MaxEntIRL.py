import os
import pickle
import torch as th

import pytest
from imitation.data import rollout

from algos.torch.MaxEntIRL.algorithm import MaxEntIRL, GuidedCostLearning
from common.callbacks import SaveCallback
from common.util import make_env


@pytest.fixture
def demo_dir():
    return os.path.abspath(os.path.join("..", "..", "IRL", "demos"))


@pytest.fixture
def expert(demo_dir):
    expert_dir = os.path.join(demo_dir, "HPC", "sub01.pkl")
    with open(expert_dir, "rb") as f:
        expert_trajs = pickle.load(f)
    return rollout.flatten_trajectories(expert_trajs)


@pytest.fixture
def env(demo_dir):
    subpath = os.path.join(demo_dir, "HPC", "sub01")
    return make_env("HPC_custom-v1", use_vec_env=False, num_envs=1, subpath=subpath + "/sub01")


@pytest.fixture
def learner(env, expert):
    from imitation.util import logger
    from IRL.scripts.project_policies import def_policy
    logger.configure("tmp/log", format_strs=["stdout", "tensorboard"])

    def feature_fn(x):
        return x

    agent = def_policy("sac", env, device='cpu', verbose=1)

    return MaxEntIRL(
        env,
        agent=agent,
        feature_fn=feature_fn,
        expert_transitions=expert,
        rew_arch=[8, 8],
        device='cpu',
        env_kwargs={'vec_normalizer': None},
        rew_kwargs={'type': 'ann', 'scale': 1},
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


def test_validity(learner):
    learner.learn(
        total_iter=10,
        agent_learning_steps=10000,
        gradient_steps=50,
        n_episodes=5,
        max_agent_iter=5,
    )


def test_GCL(env, expert):
    from imitation.util import logger
    from IRL.scripts.project_policies import def_policy
    logger.configure("tmp/log", format_strs=["stdout", "tensorboard"])

    def feature_fn(x):
        return x

    agent = def_policy("ppo", env, device='cpu', verbose=1)

    learner = GuidedCostLearning(
        env,
        agent=agent,
        feature_fn=feature_fn,
        expert_transitions=expert,
        rew_arch=[8, 8],
        device='cuda:0',
        env_kwargs={},
        rew_kwargs={'type': 'ann'}
    )

    learner.learn(
        total_iter=10,
        agent_learning_steps=10000,
        gradient_steps=10,
        n_episodes=8,
        max_agent_iter=1
    )
