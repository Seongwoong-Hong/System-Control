import numpy as np
import math, csv
from gym import utils, spaces
from gym.envs.mujoco import mujoco_env

class IP_custom_PD(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self):
        utils.EzPickle.__init__(self)
        mujoco_env.MujocoEnv.__init__(self, 'IP_uni.xml', 1)
        #self.action_space = spaces.Box(low=-np.inf, high=np.inf, shape=(1,), dtype='float64')
        #self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(2,), dtype='float64')

    def step(self, a):
        self.do_simulation(a/100, self.frame_skip)
        ob = self._get_obs()
        rew = - (ob[0] ** 2 + 0.01 * self.sim.data.qfrc_actuator ** 2)
        done = False
        return ob, rew, done, {}

    def reset_model(self):
        q = np.array([0.5, 3.14])
       # q = np.concatenate([self.np_random.uniform(size=1, low=-1, high=1), self.np_random.uniform(size=1, low=-1, high=1)])
        self.set_state(np.array([q[1]]), np.array([q[0]]))
        return self._get_obs()

    def _get_obs(self):
        return np.concatenate([self.sim.data.qvel, self.sim.data.qpos]).ravel()

    def viewer_setup(self):
        v = self.viewer
        v.cam.trackbodyid = 0
        v.cam.distance = self.model.stat.extent
