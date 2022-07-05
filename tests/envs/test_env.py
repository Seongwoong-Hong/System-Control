import gym
import time
import numpy as np

from common.verification import verify_policy
from common.util import make_env

from scipy import io


def test_pybullet_envs():
    env = make_env("HPC_pybullet-v0", subpath="../../IRL/demos/HPC/sub01/sub01")
    env.render(mode="human")
    env.reset()
    env.set_state([0.25, 0.25], [0.0, 0.0])
    env.camera_adjust()
    done = False
    while not done:
        act = env.action_space.sample()
        ob, rew, done, info = env.step(act)
        print(ob[-1])
        time.sleep(0.01)
    assert isinstance(env, gym.Env)


def test_hpc_obs_reset():
    init_states = []
    for i in range(35):
        file = f"../../IRL/demos/HPC/sub01/sub01i{i+1}.mat"
        init_states += [io.loadmat(file)['state'][0, :4]]
    env = make_env("HPC_pybullet-v0", subpath="../../IRL/demos/HPC/sub01/sub01")
    for i in range(35):
        env.render(mode="None")
        init = env.reset()
        assert (init[:4] == init_states[i]).all()
        done = False
        while not done:
            act = env.action_space.sample()
            ob, rew, done, info = env.step(act)


def test_init():
    env = make_env("IDP_custom-v1", subpath="../../IRL/demos/HPC/sub01/sub01")
    env.render()
    env.reset()
    env.set_state(np.array([0.3, 0.3]), np.array([1.0, 1.0]))
    done = False
    for _ in range(1000):
        _, _, _, _ = env.step(np.array([1., 1.]))
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
