import numpy as np
from gym import utils, spaces
from gym.envs.mujoco import mujoco_env
import os, random

class IP_custom_PD(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self, n_steps=None):
        self.traj_len = 0
        self.n_steps = n_steps
        utils.EzPickle.__init__(self)
        mujoco_env.MujocoEnv.__init__(self, os.path.join(os.path.dirname(__file__), "assets", "IP_custom.xml"), 25)
        self.action_space = spaces.Box(low=-np.inf, high=np.inf, shape=(1,), dtype='float64')
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(2,), dtype='float64')

    def step(self, a):
        self.do_simulation(a/1000, self.frame_skip)
        ob = self._get_obs()
        rew = - (ob[0] ** 2 + 0.01 * self.sim.data.qfrc_actuator ** 2)
        done = False
        info = {}
        if self.n_steps is None:
            pass
        elif self.traj_len == self.n_steps:
            done = True
            info = {"terminal observation":ob}
            self.traj_len = 0
        self.traj_len += 1
        return ob, rew, done, info

    def reset_model(self):
        # init_group = np.array([[0.1, -0.2],
        #                        [-0.15, 0.1],
        #                        [0.05, 0.01],
        #                        [0.085, -0.15],
        #                        [0.12, 0.06],
        #                        [0.20, -0.1],
        #                        [-0.12, -0.15],
        #                        [-0.05, 0.15]])
        # q = random.choice(init_group)
        q = np.random.uniform(size=2, low=-0.15, high=0.15)
        self.set_state(np.array([q[1]]), np.array([q[0]]))
        return self._get_obs()

    def _get_obs(self):
        return np.concatenate([self.sim.data.qvel, self.sim.data.qpos]).ravel()

    def viewer_setup(self):
        v = self.viewer
        v.cam.trackbodyid = 0
        v.cam.distance = 2 * self.model.stat.extent