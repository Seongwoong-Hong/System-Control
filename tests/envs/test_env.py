import time
import numpy as np
from common.util import make_env


def test_hpc_env_reset(hpc_env):
    hpc_env.reset()
    done = False
    while not done:
        act = hpc_env.action_space.sample()
        ob, rew, done, info = hpc_env.step(act)


def test_init():
    env = make_env("IDP_custom-v1", subpath="../../IRL/demos/HPC/sub01/sub01")
    env.render()
    env.reset()
    # env.set_state(np.array([0.0, 0.3]), np.array([0.0, 0.0]))
    done = False
    for _ in range(1000):
        env.set_state(np.array([0.3, 0.3]), np.array([0.0, 0.0]))
        # _, _, _, _ = env.step(np.array([1., 1.]))
        env.render()
        print(env.current_obs)
        time.sleep(env.dt)


def test_2d_env():
    env = make_env("SpringBall-v2")
    for i in range(10):
        env.reset()
        done = False
        env.render()
        while not done:
            action = env.action_space.sample()
            st, r, done, _ = env.step(np.array([0]))
            env.render()
            time.sleep(env.dt)
    env.close()
