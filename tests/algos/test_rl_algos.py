import os
import time
import pytest
import numpy as np
from scipy import io
from common.util import make_env
from RL.project_policies import def_policy


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


def test_finite_rl():
    subpath = os.path.join("..", "..", "IRL", "demos", "HPC", f"sub03_cropped", "sub03")
    init_states = []
    for i in range(5):
        for j in range(6):
            init_states += [io.loadmat(subpath + f"i{i + 1}_{j}.mat")['state'][0, :4]]
    bsp = io.loadmat(subpath + f"i1_0.mat")['bsp']
    env = make_env(f"DiscretizedHuman-v2", num_envs=1, N=[11, 21, 21, 21], bsp=bsp)
    eval_env = make_env(f"DiscretizedHuman-v0", num_envs=1, N=[11, 21, 21, 21], bsp=bsp, init_states=init_states)
    t1 = time.time()
    algo = def_policy("finitesoftqiter", env, device='cuda:0')
    algo.learn(2000)
    algo2 = def_policy("softqiter", env, device='cuda:0')
    # algo2.learn(2000)
    algo2.policy.policy_table = algo.policy.policy_table[0]
    print(time.time() - t1)
    for _ in range(1):
        obs = eval_env.reset()
        done = False
        while not done:
            a, _ = algo2.predict(obs, deterministic=True)
            next_obs, r, done, _ = eval_env.step(a)
            obs = next_obs
            time.sleep(env.envs[0].env.dt)
            eval_env.render()
    eval_env.close()
    # assert np.abs(algo.policy.policy_table[0] - algo2.policy.policy_table).mean() <= 1e-4
