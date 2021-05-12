import pytest
import os
import pickle

from common.util import make_env
from algo.torch.MaxEntIRL.algorithm import MaxEntIRL
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


def test_make_obj(expert, env):
    learning = MaxEntIRL(env,
                         agent_learning_steps=1000,
                         expert_transitions=expert,
                         rew_kwargs={'lr': 1e-3, 'arch': [4, 8, 8]},
                         )
    # Test rollout from agent
    rollouts = learning.rollout_from_agent(n_episodes=8)
    # Test calculation of loss
    loss = learning.cal_loss(rollouts)
    assert isinstance(loss.item(), float)


def test_learn_agent(env):
    learning = MaxEntIRL(env,
                         agent_learning_steps=1000,
                         expert_transitions=expert,
                         rew_kwargs={'lr': 1e-3, 'arch': [4, 8, 8]},
                         sac_kwargs={'verbose': 1}
                         )
    learning.agent.learn(total_timesteps=1e4)


def test_process(env, expert):
    learning = MaxEntIRL(env,
                         agent_learning_steps=1000,
                         expert_transitions=expert,
                         rew_lr=1e-5,
                         rew_arch=[3, 8, 8],
                         device='cuda:0',
                         sac_kwargs={'verbose': 1},
                         )
    learning.learn(total_iter=10, gradient_steps=1)
