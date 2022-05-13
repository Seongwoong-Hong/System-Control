import os
import pickle

import torch as th
from scipy import io
import pytest

from algos.torch.MaxEntIRL.algorithm import MaxEntIRL, GuidedCostLearning, APIRL
from algos.torch.sac import MlpPolicy, SAC
from algos.tabular.viter import FiniteSoftQiter, FiniteViter, SoftQiter
from common.callbacks import SaveCallback
from common.util import make_env
from common.wrappers import *

subj = "sub05"
env_name = "2DWorld"
env_id = f"{env_name}_disc"


@pytest.fixture
def demo_dir():
    proj_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    return os.path.join(proj_path, "IRL", "demos")


@pytest.fixture
def expert(demo_dir):
    # expert_dir = os.path.join(demo_dir, env_name, "19191919", f"{subj}_1.pkl")
    expert_dir = os.path.join(demo_dir, env_name, "20", "2*alpha_nobias.pkl")
    with open(expert_dir, "rb") as f:
        expert_trajs = pickle.load(f)
    return expert_trajs


@pytest.fixture
def env(expert, demo_dir):
    subpath = os.path.join(demo_dir, "HPC", subj, subj)
    bsp = io.loadmat(subpath + f"i1.mat")['bsp']
    init_states = []
    for traj in expert:
        init_states += [traj.obs[0]]
    # return make_env(f"{env_id}-v2", bsp=bsp, N=[19, 19, 19, 19], NT=[11, 11])
    return make_env(f"{env_id}-v2")


@pytest.fixture
def eval_env(expert, demo_dir):
    subpath = os.path.join(demo_dir, "HPC", subj, subj)
    bsp = io.loadmat(subpath + f"i1.mat")['bsp']
    init_states = []
    for traj in expert:
        init_states += [traj.obs[0]]
    # return make_env(f"{env_id}-v0", bsp=bsp, init_states=init_states, N=[19, 19, 19, 19], NT=[11, 11])
    return make_env(f"{env_id}-v0", init_states=init_states)

@pytest.fixture
def learner(env, expert, eval_env):
    from imitation.util import logger
    logger.configure("tmp/log", format_strs=["stdout"])

    def feature_fn(x):
        # x1, x2, x3, x4, a1, a2 = th.split(x, 1, dim=-1)
        # return th.cat([x, x ** 2, x1 * x2, x3 * x4, a1 * a2], dim=1)
        # return x ** 2
        # return th.cat([x, x**2, x**3, x**4], dim=1)
        # x1, x2, a1, a2 = th.split(x, 1, dim=-1)
        # out = x ** 2
        # ob_sec, act_sec = 4, 3
        # for i in range(1, ob_sec):
        #     out = th.cat([out, (x1 - i / ob_sec) ** 2, (x2 - i / ob_sec) ** 2, (x3 - i /ob_sec) ** 2, (x4 - i / ob_sec) ** 2,
        #                   (x1 + i / ob_sec) ** 2, (x2 + i / ob_sec) ** 2, (x3 + i / ob_sec) ** 2, (x4 + i / ob_sec) ** 2], dim=1)
        # for i in range(1, act_sec):
        #     out = th.cat([out, (a1 - i / act_sec) ** 2, (a2 - i / act_sec) ** 2, (a1 + i / act_sec) ** 2, (a2 + i / act_sec) ** 2], dim=1)
        # return out
        return th.cat([x, x**2], dim=1)

    agent = FiniteSoftQiter(env, gamma=1, alpha=0.01, device='cpu')
    # agent = SoftQiter(env, gamma=0.99, alpha=0.01, device='cuda:2')

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
                    'optim_kwargs': {'weight_decay': 0.0, 'lr': 1e-2, 'betas': (0.9, 0.999)},
                    'lr_scheduler_cls': th.optim.lr_scheduler.StepLR,
                    'lr_scheduler_kwargs': {'step_size': 10, 'gamma': 0.95},
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
        return x ** 2
        # return th.cat([x, x**2], dim=1)

    # agent = FiniteSoftQiter(env, device='cuda:1', verbose=True)
    agent = SAC(
        MlpPolicy,
        env=env,
        gamma=0.99,
        ent_coef='auto',
        target_entropy=-0.1,
        tau=0.01,
        buffer_size=int(1e5),
        learning_starts=10000,
        train_freq=1,
        gradient_steps=1,
        device='cpu',
        verbose=1,
        policy_kwargs={'net_arch': [32, 32]}
    )

    learner = GuidedCostLearning(
        env,
        eval_env=eval_env,
        feature_fn=feature_fn,
        agent=agent,
        expert_trajectories=expert,
        use_action_as_input=True,
        rew_arch=[],
        device='cpu',
        env_kwargs={'vec_normalizer': None, 'reward_wrapper': RewardWrapper},
        rew_kwargs={'type': 'ann', 'scale': 1,
                    'optim_kwargs': {'weight_decay': 0.01, 'lr': 1e-2, 'betas': (0.9, 0.999)}
                    },
    )

    learner.learn(
        total_iter=50,
        agent_learning_steps=1.5e5,
        n_episodes=len(expert),
        max_agent_iter=1,
        min_agent_iter=1,
        max_gradient_steps=1,
        min_gradient_steps=1,
        early_stop=True,
    )


def test_state_visitation(env):
    from copy import deepcopy
    agent1 = FiniteSoftQiter(env, gamma=1, alpha=0.01, device='cpu')
    agent1.learn(0)
    agent2 = SoftQiter(env, gamma=0.99, alpha=0.01, device='cpu')
    agent2.learn(1e3)

    D_prev1 = th.ones_like(agent1.policy.v_table[0]) / (agent1.policy.act_size * agent1.policy.obs_size)
    D_prev2 = th.ones_like(agent2.policy.v_table) / (agent2.policy.act_size * agent2.policy.obs_size)
    Dc1 = D_prev1[None, :] * agent1.policy.policy_table[0]
    Dc2 = D_prev2[None, :] * agent2.policy.policy_table
    for t in range(1, 50):
        D1 = th.zeros_like(agent1.policy.v_table[0]).to(agent1.device)
        D2 = th.zeros_like(agent2.policy.v_table).to(agent2.device)
        for a in range(agent1.policy.act_size):
            D1 += agent1.transition_mat[a] @ (D_prev1 * agent1.policy.policy_table[t - 1, a])
            D2 += agent2.transition_mat[a] @ (D_prev2 * agent2.policy.policy_table[a])
        Dc1 += agent1.policy.policy_table[t] * D1[None, :] * agent1.gamma ** t
        Dc2 += agent2.policy.policy_table * D2[None, :] * agent2.gamma ** t
        D_prev1 = deepcopy(D1)
        D_prev2 = deepcopy(D2)

    print(Dc1.mean(), Dc2.mean())
    print(Dc1.max(), Dc2.max())
    print((Dc1.cpu() - Dc2.cpu()).abs().mean())
    print((Dc1.cpu() - Dc2.cpu()).abs().max())


def test_state_visit_difference_according_to_init(learner):
    uniform_init_D = th.ones_like(learner.init_D) / learner.init_D.numel()
    Ds = learner.cal_state_visitation()
    learner.init_D = uniform_init_D
    uni_Ds = learner.cal_state_visitation()
    assert th.max(th.abs(Ds - uni_Ds)).item() < 0.1
    print(f"{th.max(th.abs(Ds - uni_Ds)).item()} < 0.1")
    print(f"{th.sum(th.abs(Ds - uni_Ds)).item()}")
