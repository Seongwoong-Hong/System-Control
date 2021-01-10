import numpy as np
import math, csv
from gym import utils, spaces
from gym.envs.mujoco import mujoco_env
import os

class IP_custom_PD(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self, n_steps):
        self.traj_len = 0
        self.n_steps = n_steps
        utils.EzPickle.__init__(self)
        mujoco_env.MujocoEnv.__init__(self, os.path.join(os.path.dirname(__file__), "assets", "IP_custom.xml"), 1)
        self.action_space = spaces.Box(low=-np.inf, high=np.inf, shape=(1,), dtype='float64')
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(2,), dtype='float64')


    def step(self, a):
        self.do_simulation(a/100, self.frame_skip)
        ob = self._get_obs()
        rew = - (ob[0] ** 2 + 0.01 * self.sim.data.qfrc_actuator ** 2)
        done = False
        info = {}
        if self.traj_len == self.n_steps:
            done = True
            info = {"terminal observation":ob}
            self.traj_len = 0
        self.traj_len += 1
        return ob, rew, done, info

    def reset_model(self):
        q = np.array([[1, -0.5],
                     [0.5, 0.5],
                     [0.7, 0.0],
                     [-0.5, 1]])
        idx = int(np.floor(np.random.uniform(size=1, low=0, high=4)))
        if idx==4: idx = 3
        q = q[idx]
        self.set_state(np.array([q[1]]), np.array([q[0]]))
        return self._get_obs()

    def _get_obs(self):
        return np.concatenate([self.sim.data.qvel, self.sim.data.qpos]).ravel()

    def viewer_setup(self):
        v = self.viewer
        v.cam.trackbodyid = 0
        v.cam.distance = self.model.stat.extent
