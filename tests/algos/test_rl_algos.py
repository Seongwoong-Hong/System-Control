import time
import pytest
import numpy as np
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
    env = make_env(f"DiscretizedPendulum-v0", num_envs=1, h=[0.03, 0.15])
    algo = def_policy("finitesoftqiter", env)
    algo.learn(2000)
    algo2 = def_policy("softqiter", env)
    algo2.learn(2000)
    for _ in range(5):
        obs = env.reset()
        for t_ind in range(200):
            a, _ = algo2.predict(obs, deterministic=False)
            next_obs, r, _, _ = env.step(a)
            obs = next_obs
            env.render()
            time.sleep(0.01)
    env.close()
    assert np.abs(algo.policy.policy_table[0] - algo2.policy.policy_table).mean() <= 1e-4
