import os
import pickle

import torch as th
from scipy import io, signal
import pytest

from gym_envs.envs import FaissDiscretizationInfo
from algos.torch.MaxEntIRL.algorithm import MaxEntIRL, GuidedCostLearning, APIRL
from algos.torch.sac import MlpPolicy, SAC
from algos.torch.OptCont import FiniteLQRPolicy
from algos.tabular.viter import FiniteSoftQiter, FiniteViter, SoftQiter
from common.callbacks import SaveCallback
from common.util import make_env
from common.wrappers import *

subj = "sub05"
env_name = "IDP"
env_id = f"{env_name}_custom"
proj_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
demo_dir = os.path.join(proj_path, "IRL", "demos", env_name)
bsp = io.loadmat(demo_dir + f"/{subj}/{subj}i1.mat")['bsp']


@pytest.fixture
def expert_trajs():
    expert_dir = os.path.join(demo_dir, f"{subj}_1.pkl")
    # expert_dir = os.path.join(demo_dir, env_name, "20", "2*alpha_nobias.pkl")
    with open(expert_dir, "rb") as f:
        expert_trajs = pickle.load(f)
    return expert_trajs


@pytest.fixture
def env(expert_trajs):
    return make_env(f"{env_id}-v2", bsp=bsp)


@pytest.fixture
def eval_env(expert, demo_dir):
    init_states = []
    for traj in expert:
        init_states += [traj.obs[0]]
    return make_env(f"{env_id}-v0", init_states=init_states, bsp=bsp)


class IDPLQRPolicy(FiniteLQRPolicy):
    def _build_env(self):
        I1, I2 = self.env.envs[0].model.body_inertia[1:, 0]
        l1 = self.env.envs[0].model.body_pos[2, 2]
        lc1, lc2 = self.env.envs[0].model.body_ipos[1:, 2]
        m1, m2 = self.env.envs[0].model.body_mass[1:]
        g = 9.81
        M = np.array([[I1 + m1*lc1**2 + I2 + m2*l1**2 + 2*m2*l1*lc2 + m2*lc2**2, I2 + m2*l1*lc2 + m2*lc2**2],
                      [I2 + m2*l1*lc2 + m2*lc2**2, I2 + m2*lc2**2]])
        C = np.array([[m1*lc1*g + m2*l1*g + m2*g*lc2, m2*g*lc2],
                      [m2*g*lc2, m2*g*lc2]])
        self.A, self.B = np.zeros([4, 4]), np.zeros([4, 2])
        self.A[:2, 2:] = np.eye(2, 2)
        self.A[2:, :2] = np.linalg.inv(M) @ C
        self.B[2:, :] = np.linalg.inv(M) @ np.eye(2, 2)
        self.A, self.B, _, _, dt = signal.cont2discrete((self.A, self.B, np.array([1, 1, 1, 1]), 0), self.env.envs[0].dt)
        self.Q = self.env.envs[0].Q*100
        self.R = self.env.envs[0].R*100
        # self.gear = th.tensor([60, 50])
        self.gear = 1

def test_maxentirl_scripts(expert_trajs):
    from IRL.scripts.MaxEntIRL import main
    def feature_fn(x):
        return th.cat([x, x**2], dim=1)
    kwargs = {
        'log_dir': ".",
        'env': env,
        'eval_env': eval_env,
        'feature_fn': feature_fn,
        'expert_trajs': expert_trajs,
        'use_action_as_input': True,
        'rew_arch': [],
        'env_kwargs': {'vec_normalizer': None, 'num_envs': 1, 'reward_wrapper': RewardInputNormalizeWrapper},
        'rew_kwargs': {'type': 'ann', 'scale': 1,
                       'optim_kwargs': {'weight_decay': 1e-3, 'lr': 1e-1, 'betas': (0.9, 0.999)},
                       # 'lr_scheduler_cls': th.optim.lr_scheduler.StepLR,
                       # 'lr_scheduler_kwargs': {'step_size': 1, 'gamma': 0.95}
                       },
        'agent_cls': FiniteSoftQiter,
        'agent_kwargs': {'env': env, 'gamma': 1, 'alpha': 0.01},
        'device': 'cpu',
        'callback_fn': None,
    }
    main(**kwargs)



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


def test_state_visitation_from_trajs(env):
    agent = SoftQiter(env, gamma=0.99, alpha=0.01, device='cuda:3')


def test_state_visit_difference_according_to_init(learner):
    uniform_init_D = th.ones_like(learner.init_D) / learner.init_D.numel()
    Ds = learner.cal_state_visitation()
    learner.init_D = uniform_init_D
    uni_Ds = learner.cal_state_visitation()
    assert th.max(th.abs(Ds - uni_Ds)).item() < 0.1
    print(f"{th.max(th.abs(Ds - uni_Ds)).item()} < 0.1")
    print(f"{th.sum(th.abs(Ds - uni_Ds)).item()}")
