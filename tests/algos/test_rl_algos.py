import os
import time
import pytest
import numpy as np
import torch as th
from scipy import io
from common.util import make_env
from matplotlib import pyplot as plt


def test_ppo_log():
    env = make_env("IDP_custom-v1", use_vec_env=False, num_envs=8)
    algo = def_policy("sac", env, log_dir="tmp/log/rl")
    algo.learn(total_timesteps=1e5)


def test_learn_ppo():
    env = make_env("IDP_pybullet-v2")
    algo = def_policy("ppo", env, device='cpu', log_dir=None, verbose=1)
    algo.learn(total_timesteps=1e5)


def test_iter_predict():
    env = make_env("2DTarget_disc-v2", use_vec_env=True, num_envs=1)
    algo = def_policy("viter", env, device='cpu')
    algo.learn(10000)
    obs = []
    ob = env.reset()
    done = False
    obs.append(ob)
    while not done:
        act, _ = algo.predict(ob, deterministic=False)
        ob, r, done, _ = env.step(act)
        obs.append(ob)
    print('end')


def test_finite_softqiter_at_discretized_env():
    import pickle
    from algos.tabular.viter import SoftQiter
    subj = "sub06"
    subpath = os.path.join("..", "..", "IRL", "demos", "HPC", subj, subj)
    # with open(f"../../IRL/demos/DiscretizedHuman/19171717/{subj}.pkl", "rb") as f:
    #     expt = pickle.load(f)
    # init_states = []
    # for traj in expt:
    #     init_states += [traj.obs[0]]
    init_states = np.array([-0.14329, 0.32951, -0.52652, -0.11540])
    bsp = io.loadmat(subpath + f"i1.mat")['bsp']
    env = make_env(f"DiscretizedHuman-v2", num_envs=1, N=[21, 21, 21, 21], NT=[19, 19], bsp=bsp)
    eval_env = make_env(f"DiscretizedHuman-v0", num_envs=1, N=[21, 21, 21, 21], NT=[19, 19], bsp=bsp,
                        init_states=init_states)
    # t1 = time.time()
    # algo = def_policy("finitesoftqiter", env, device='cuda:2')
    # algo.learn(0)
    eval_algo = SoftQiter(env, gamma=0.95, alpha=0.001, device='cuda:2')
    eval_algo.learn(2000)
    # assert th.abs(algo.policy.policy_table[0] - eval_algo.policy.policy_table).mean().item() <= 1e-4
    # eval_algo.policy.policy_table = algo.policy.policy_table[0]
    # print(time.time() - t1)
    whole_acts, whole_obs = [], []
    for _ in range(0, 15, 3):
        acts, obs = [], []
        ob = eval_env.reset()
        done = False
        while not done:
            a, _ = eval_algo.predict(ob, deterministic=False)
            next_obs, r, done, _ = eval_env.step(a)
            acts.append(a)
            obs.append(ob)
            ob = next_obs
            time.sleep(env.get_attr("dt")[0])
            eval_env.render("None")
        whole_acts.append(np.vstack(acts))
        whole_obs.append(np.vstack(obs))
    eval_env.close()

    x_value = range(1, 51)
    obs_fig = plt.figure(figsize=[18, 12], dpi=150.0)
    acts_fig = plt.figure(figsize=[18, 6], dpi=150.0)
    for ob_idx in range(4):
        ax = obs_fig.add_subplot(2, 2, ob_idx + 1)
        for traj_idx in range(5):
            ax.plot(x_value, whole_obs[traj_idx][:, ob_idx], color='k')
    for act_idx in range(2):
        ax = acts_fig.add_subplot(1, 2, act_idx + 1)
        for traj_idx in range(5):
            ax.plot(x_value, whole_acts[traj_idx][:, act_idx], color='k')
    obs_fig.tight_layout()
    acts_fig.tight_layout()
    plt.show()
