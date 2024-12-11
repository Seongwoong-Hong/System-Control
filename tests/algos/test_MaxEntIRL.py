import os
import pickle
import pytest
from scipy import io
from imitation.util import logger

from algos.torch.MaxEntIRL.algorithm import ContMaxEntIRL
from algos.tabular.viter import FiniteSoftQiter, SoftQiter
from common.sb3.util import make_env
from common.wrappers import *
from IRL.scripts.MaxEntIRL import main
from IRL.src import *

subj = "sub05"
env_name = "HPC"
env_id = f"{env_name}_custom"
proj_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
demo_dir = os.path.join(proj_path, "IRL", "demos")
bsp = io.loadmat(demo_dir + f"/HPC/{subj}_full/{subj}i1.mat")['bsp']


@pytest.fixture
def expert_trajs():
    expert_dir = os.path.join(demo_dir, "HPC", "full", f"{subj}_1.pkl")
    # expert_dir = os.path.join(demo_dir, env_name, "20", "2*alpha_nobias.pkl")
    with open(expert_dir, "rb") as f:
        expert_trajs = pickle.load(f)
    return expert_trajs


@pytest.fixture
def env(expert_trajs):
    pltqs = []
    init_states = []
    for traj in expert_trajs:
        pltqs += [traj.pltq]
        init_states += [traj.obs[0]]
    return make_env(f"{env_id}-v2", init_states=init_states, bsp=bsp, pltqs=pltqs)


@pytest.fixture
def eval_env(expert_trajs):
    pltqs = []
    init_states = []
    for traj in expert_trajs:
        pltqs += [traj.pltq]
        init_states += [traj.obs[0]]
    return make_env(f"{env_id}-v0", init_states=init_states, bsp=bsp, pltqs=pltqs)


def test_maxentirl_scripts(env, eval_env, expert_trajs):
    logger.configure("./test_log", format_strs=['stdout'])

    def feature_fn(x):
        t, dt, u = th.split(x, 2, 1)
        prev_u = th.cat([th.zeros(1, 2), u], dim=0)
        u_diff = u - prev_u[:-1]
        return th.cat([t ** 2, dt ** 2, u ** 2, u_diff ** 2], dim=-1)
        # return x ** 2
        # return th.cat([x, x ** 2], dim=-1)
    kwargs = {
        'log_dir': "./test_log",
        'env': env,
        'eval_env': eval_env,
        'feature_fn': feature_fn,
        'expert_trajs': expert_trajs,
        'use_action_as_input': True,
        'rew_arch': [],
        'env_kwargs': {'vec_normalizer': None, 'num_envs': 1, 'reward_wrapper': RewardWrapper},
        'rew_kwargs': {'type': 'xx',
                       'optim_kwargs': {'weight_decay': 0.0, 'lr': 3e-2, 'betas': (0.9, 0.999)},
                       },
        'agent_cls': IDPDiffLQRPolicy,
        'agent_kwargs': {'env': env, 'gamma': 1, 'alpha': 0.01},
        'device': 'cpu',
        'callback_fns': None,
    }
    main(**kwargs)


def test_callbacks(env, eval_env, expert_trajs):
    logger.configure("./test_log", format_strs=['stdout', 'tensorboard'])
    def feature_fn(x):
        return th.cat([x ** 2, x ** 4], dim=-1)

    kwargs = {
        'log_dir': "./test_log",
        'env': env,
        'eval_env': eval_env,
        'feature_fn': feature_fn,
        'expert_trajs': expert_trajs,
        'use_action_as_input': True,
        'rew_arch': [],
        'env_kwargs': {'vec_normalizer': None, 'num_envs': 1, 'reward_wrapper': RewardWrapper},
        'rew_kwargs': {'type': 'sq',
                       'optim_kwargs': {'weight_decay': 0.01, 'lr': 3e-2, 'betas': (0.9, 0.999)},
                       },
        'agent_cls': IDPiterLQRPolicy,
        'agent_kwargs': {'env': env, 'gamma': 1, 'alpha': 0.02},
        'device': 'cpu',
        'callback_fns': [callback.log_figure],
    }
    main(**kwargs)


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


def test_state_visitation_from_trajs(env, eval_env, expert_trajs):
    logger.configure(".", format_strs=["stdout"])
    def feature_fn(x):
        return x ** 2
    agent = IDPLQRPolicy(env, gamma=1, alpha=0.002, device='cpu')
    learner = ContMaxEntIRL(
        env,
        eval_env=eval_env,
        feature_fn=feature_fn,
        agent=agent,
        expert_trajectories=expert_trajs,
        use_action_as_input=True,
        rew_arch=[],
        device='cpu',
        env_kwargs={'vec_normalizer': None, 'num_envs': 1, 'reward_wrapper': RewardWrapper},
        rew_kwargs={'type': 'ann', 'scale': 1, 'optim_kwargs': {'weight_decay': 0.0, 'lr': 1e-1}},
    )
    learner.reward_net.w_th = th.tensor([ 3.5593e+00, -3.3656e-17], requires_grad=True)
    learner.reward_net.w_tq = th.tensor([-4.8090e-09, -6.0656e-05], requires_grad=True)

    learner._reset_agent(**{'vec_normalizer': None, 'num_envs': 1, 'reward_wrapper': RewardWrapper})
    er = learner.cal_expert_mean_reward()
    import time
    t1 = time.time()
    learner.collect_rollouts(75)
    r1 = learner.cal_agent_mean_reward()
    # t2 = time.time()
    # learner.collect_rollouts(75)
    # r2 = learner.cal_agent_mean_reward()
    # t3 = time.time()
    # learner.collect_rollouts(150)
    # r3 = learner.cal_agent_mean_reward()
    # t4 = time.time()
    # learner.collect_rollouts(300)
    # r4 = learner.cal_agent_mean_reward()
    # t5 = time.time()
    print(er, r1)
    # print(r1, r2, r3, r4)
    # print(t2 - t1, t3 - t2, t4 - t3, t5 - t4)


def test_state_visit_difference_according_to_init(learner):
    uniform_init_D = th.ones_like(learner.init_D) / learner.init_D.numel()
    Ds = learner.cal_state_visitation()
    learner.init_D = uniform_init_D
    uni_Ds = learner.cal_state_visitation()
    assert th.max(th.abs(Ds - uni_Ds)).item() < 0.1
    print(f"{th.max(th.abs(Ds - uni_Ds)).item()} < 0.1")
    print(f"{th.sum(th.abs(Ds - uni_Ds)).item()}")
