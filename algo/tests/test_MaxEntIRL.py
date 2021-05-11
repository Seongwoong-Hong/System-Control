import pytest
import os
import pickle

from common.util import make_env
from algo.torch.MaxEntIRL.algorithm import MaxEntIRL
from imitation.data import rollout


@pytest.fixture
def expert():
    expert_dir = os.path.join("..", "..", "IRL", "demos", "IDP", "expert.pkl")
    with open(expert_dir, "rb") as f:
        expert_trajs = pickle.load(f)
    return rollout.flatten_trajectories(expert_trajs)


@pytest.fixture
def env():
    return make_env("IDP_custom-v2", use_vec_env=False)


def test_make_obj(expert, env):
    learning = MaxEntIRL(env,
                         agent_learning_steps=1000,
                         expert_transitions=expert,
                         rew_kwargs={'lr': 1e-3, 'arch': [2, 4, 4]},
                         )
    # Test rollout from agent
    rollouts = learning.rollout_from_agent()
    # Test calculation of loss
