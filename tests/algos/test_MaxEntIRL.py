import os
import pickle

import torch as th
from scipy import io
import pytest

from algos.torch.MaxEntIRL.algorithm import MaxEntIRL, GuidedCostLearning, APIRL
from algos.tabular.viter import FiniteSoftQiter
from common.callbacks import SaveCallback
from common.util import make_env
from common.wrappers import *

subj = "sub06"
env_name = "DiscretizedHuman"
env_id = f"{env_name}"


@pytest.fixture
def demo_dir():
    proj_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    return os.path.join(proj_path, "IRL", "demos")


@pytest.fixture
def expert(demo_dir):
    expert_dir = os.path.join(demo_dir, env_name, "19191919", f"{subj}_1.pkl")
    with open(expert_dir, "rb") as f:
        expert_trajs = pickle.load(f)
    return expert_trajs


@pytest.fixture
def env(demo_dir):
    subpath = os.path.join(demo_dir, "HPC", subj, subj)
    bsp = io.loadmat(subpath + f"i1.mat")['bsp']
    return make_env(f"{env_id}-v2", bsp=bsp, N=[19, 19, 19, 19], NT=[11, 11])


@pytest.fixture
def eval_env(expert, demo_dir):
    subpath = os.path.join(demo_dir, "HPC", subj, subj)
    bsp = io.loadmat(subpath + f"i1.mat")['bsp']
    init_states = []
    for traj in expert:
        init_states += [traj.obs[0]]
    return make_env(f"{env_id}-v0", bsp=bsp, N=[19, 19, 19, 19], NT=[11, 11], init_states=init_states)


@pytest.fixture
def learner(env, expert, eval_env):
    from imitation.util import logger
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
        return x ** 2
        # return th.cat([x, x ** 2], dim=1)

    agent = FiniteSoftQiter(env, gamma=1, alpha=0.001, device='cuda:1')

    return MaxEntIRL(
        env,
        eval_env=eval_env,
        agent=agent,
        feature_fn=feature_fn,
        expert_trajectories=expert,
        use_action_as_input=True,
        rew_arch=[],
        device=agent.device,
        env_kwargs={'vec_normalizer': None, 'reward_wrapper': RewardInputNormalizeWrapper},
        rew_kwargs={'type': 'ann', 'scale': 1,
                    'optim_kwargs': {'weight_decay': 0.0, 'lr': 1e-1, 'betas': (0.9, 0.999)}
                    },
    )


def test_callback(expert, learner):
    from stable_baselines3.common import callbacks
    from imitation.policies import serialize
    save_policy_callback = serialize.SavePolicyCallback(f"tmp/log", None)
    save_policy_callback = callbacks.EveryNTimesteps(int(1e3), save_policy_callback)
    save_reward_callback = SaveCallback(cycle=1, dirpath=f"tmp/log")
    learner.learn(
        total_iter=10,
        agent_learning_steps=0,
        n_episodes=len(expert),
        max_agent_iter=1,
        min_agent_iter=1,
        max_gradient_steps=1,
        min_gradient_steps=1,
        early_stop=True,
        callback=save_reward_callback.net_save,
        callback_period=1,
    )


def test_validity(learner, expert):
    learner.learn(
        total_iter=300,
        agent_learning_steps=0,
        n_episodes=len(expert),
        max_agent_iter=1,
        min_agent_iter=1,
        max_gradient_steps=1,
        min_gradient_steps=1,
        early_stop=True,
    )


def test_GCL(env, expert, eval_env):
    from imitation.util import logger
    logger.configure("tmp/log", format_strs=["stdout"])

    def feature_fn(x):
        # return x
        return th.cat([x, x**2], dim=1)

    agent = FiniteSoftQiter(env, device='cuda:1', verbose=True)

    learner = GuidedCostLearning(
        env,
        agent=agent,
        feature_fn=feature_fn,
        expert_trajectories=expert,
        use_action_as_input=False,
        rew_arch=[],
        device='cpu',
        eval_env=eval_env,
        env_kwargs={'reward_wrapper': ActionNormalizeRewardWrapper, "num_envs": 10},
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
    policy = FiniteSoftQiter(env, gamma=1, alpha=0.01, device='cpu')
    policy.learn(0)
    learner.agent = policy
    Ds = learner.cal_state_visitation()
    learner.get_whole_states_from_env()
    r1 = th.sum(Ds * env.get_reward_mat())
    r2 = learner.cal_expert_mean_reward()
    assert th.abs((r2 - r1) / r1).item() < 0.1
    print(th.abs((r2 - r1) / r1).item())


def test_state_visit_difference_according_to_init(learner):
    uniform_init_D = th.ones_like(learner.init_D) / learner.init_D.numel()
    Ds = learner.cal_state_visitation()
    learner.init_D = uniform_init_D
    uni_Ds = learner.cal_state_visitation()
    assert th.max(th.abs(Ds - uni_Ds)).item() < 0.1
    print(f"{th.max(th.abs(Ds - uni_Ds)).item()} < 0.1")
    print(f"{th.sum(th.abs(Ds - uni_Ds)).item()}")
