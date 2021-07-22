import os
import pickle
import pytest
from scipy import io

from imitation.data import rollout
from imitation.util import logger
from imitation.algorithms import bc
from matplotlib import pyplot as plt

from common.util import make_env


@pytest.fixture
def irl_path():
    return os.path.abspath("../../IRL")


@pytest.fixture
def pltqs(irl_path):
    pltqs = []
    for i in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]:
        file = f"{irl_path}/demos/HPC/sub01/sub01" + f"i{i + 1}.mat"
        pltqs += [io.loadmat(file)['pltq']]
    return pltqs


@pytest.fixture
def expert(irl_path):
    expert_dir = os.path.join(irl_path, "demos", "HPC", "sub01_1&2.pkl")
    with open(expert_dir, "rb") as f:
        expert_trajs = pickle.load(f)
    return rollout.flatten_trajectories(expert_trajs)


@pytest.fixture
def env(pltqs):
    return make_env("HPC_custom-v1", use_vec_env=False, num_envs=1, pltqs=pltqs)


def test_learn(env, expert):
    from algos.torch.ppo import MlpPolicy
    policy_kwargs = {
        'log_std_range': [None, 1.8],
        'net_arch': [{'pi': [32, 32], 'vf': [32, 32]}],
    }

    # Setup Logger
    logger.configure("tmp/log/BC", format_strs=["stdout", "tensorboard"])
    learner = bc.BC(
        observation_space=env.observation_space,
        action_space=env.action_space,
        policy_class=MlpPolicy,
        policy_kwargs=policy_kwargs,
        expert_data=expert,
        device='cpu',
        ent_weight=1e-5,
        l2_weight=1e-2,
    )

    learner.train(n_epochs=300)

    learner.save_policy("policy")


def test_policy(env, expert):
    policy = bc.reconstruct_policy("policy")
    learned_acts = []
    for obs in expert.obs:
        act, _ = policy.predict(obs)
        learned_acts.append(act)
    plt.plot(learned_acts)
    plt.show()
    plt.plot(expert.acts)
    plt.show()
