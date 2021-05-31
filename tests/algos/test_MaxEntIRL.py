import pytest
import os
import pickle

from common.util import make_env
from common.callbacks import SaveCallback
from algos.torch.MaxEntIRL.algorithm import MaxEntIRL
from imitation.data import rollout


@pytest.fixture
def expert():
    expert_dir = os.path.join("..", "..", "IRL", "demos", "IP", "expert.pkl")
    with open(expert_dir, "rb") as f:
        expert_trajs = pickle.load(f)
    return rollout.flatten_trajectories(expert_trajs)


@pytest.fixture
def env():
    return make_env("IP_custom-v2", use_vec_env=False)


@pytest.fixture
def learner(env, expert):
    from imitation.util import logger
    logger.configure("tmp/log", format_strs=["stdout", "tensorboard"])

    def feature_fn(x):
        return x

    learning = MaxEntIRL(env,
                         agent_learning_steps_per_one_loop=10000,
                         expert_transitions=expert,
                         rew_lr=1e-3,
                         rew_arch=[8, 8],
                         device='cuda:0',
                         sac_kwargs={'verbose': 1},
                         rew_kwargs={'feature_fn': feature_fn}
                         )
    return learning


def test_logger(env, expert):
    from imitation.util import logger
    logger.configure("tmp/log", format_strs=["stdout", "tensorboard"])
    learning = MaxEntIRL(env,
                         agent_learning_steps_per_one_loop=10000,
                         expert_transitions=expert,
                         rew_lr=1e-5,
                         rew_arch=[8, 8],
                         device='cuda:0',
                         sac_kwargs={'verbose': 1},
                         )
    learning.learn(total_iter=10, gradient_steps=1, n_episodes=8)


def test_callback(learner):
    from stable_baselines3.common import callbacks
    from imitation.policies import serialize
    save_policy_callback = serialize.SavePolicyCallback(f"tmp/log", None)
    save_policy_callback = callbacks.EveryNTimesteps(int(1e3), save_policy_callback)
    save_reward_callback = SaveCallback(cycle=1, dirpath=f"tmp/log")
    learner.learn(total_iter=10, gradient_steps=100, n_episodes=8, max_sac_iter=10, callback=save_reward_callback.net_save)


def test_wrapper(env, expert):
    from imitation.util import logger
    from common.wrappers import RewardWrapper
    logger.configure("tmp/log", format_strs=["stdout", "tensorboard"])
    learner = MaxEntIRL(env,
                        agent_learning_steps_per_one_loop=1e4,
                        expert_transitions=expert,
                        rew_lr=1e-4,
                        rew_arch=[],
                        device='cpu',
                        sac_kwargs={'verbose': 1,
                                    'reward_wrapper': RewardWrapper,
                                    }
                        )
    learner.learn(total_iter=1, gradient_steps=1, n_episodes=8, max_sac_iter=1)


def test_feature(env, expert):
    from imitation.util import logger
    import torch as th
    logger.configure("tmp/log", format_strs=["stdout", "tensorboard"])
    learner = MaxEntIRL(
        env,
        agent_learning_steps_per_one_loop=1e4,
        expert_transitions=expert,
        rew_lr=1e-4,
        rew_arch=[],
        device='cpu',
        sac_kwargs={'verbose': 1},
        rew_kwargs={'feature_fn': lambda x: th.square(x)}
    )
    learner.learn(total_iter=1, gradient_steps=1, n_episodes=8, max_sac_iter=1)


def test_validity(learner):
    learner.learn(total_iter=10, gradient_steps=50, n_episodes=8, max_sac_iter=10)
