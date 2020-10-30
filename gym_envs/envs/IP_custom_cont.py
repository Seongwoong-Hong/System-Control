import numpy as np
import math, csv, os, gym, mujoco_py
from gym import utils, spaces, error
from gym.envs.mujoco import mujoco_env
from gym.utils import seeding
from collections import OrderedDict
from os import path

class IP_custom_cont(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self):
        utils.EzPickle.__init__(self)
        ####mujoco_env.MujocoEnv.__init__(self, "IP_custom.xml", 1)
        fullpath = os.path.join(os.path.dirname(__file__), "assets", "IP_custom.xml")
        frame_skip = 1
        self.frame_skip = frame_skip
        self.model = mujoco_py.load_model_from_path(fullpath)
        self.sim = mujoco_py.MjSim(self.model)
        self.data = self.sim.data
        self.viewer = None
        self._viewers = {}

        self.metadata = {
            'render.modes': ['human', 'rgb_array', 'depth_array'],
            'video.frames_per_second': int(np.round(1.0 / self.dt))
        }

        self.init_qpos = self.sim.data.qpos.ravel().copy()
        self.init_qvel = self.sim.data.qvel.ravel().copy()

        self.action_space = spaces.Box(low=-10, high=10, shape=(2,), dtype='float64')
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(2,), dtype='float64')
        self.seed()

    def step(self, a):
        prev_ob = self._get_obs()
        ob = self._get_obs()
        act_ = []
        work = 0
        actu = []
        # ob_ = []
        idx = 0
        notdone = True
        while notdone:
            # self.render("human")
            action = 10 * a[0] * (0 - ob[0]) + 10 * a[1] * (0 - ob[1])
            # ob_.append(ob)
            self.do_simulation(action, self.frame_skip)
            actu.append(self.sim.data.qfrc_actuator[0])
            ob = self._get_obs()
            work += self.sim.data.qfrc_actuator[0] * ob[1] * 0.001
            idx += 1
            notdone = (idx < 3000) and (ob[0] >= -0.1) and (abs(ob[0]) - abs(prev_ob[0]) <= 0.2) and (self.sim.data.qfrc_actuator[0] < 100)

        # idx_reward = 0.2 * idx/3000
        # pos_reward = 0.1 * math.exp(-20 * (ob[0] - 0) ** 2)
        act_reward = 1.0 * math.exp(-0.01 * (sum(actu)/3000 - 0) ** 2)
        done = (idx < 3000) or (abs(ob[0]) > 0.01) or (self.sim.data.qfrc_actuator[0] >= 100)
        reward = act_reward# + idx_reward + pos_reward
        if not done: ob = self.reset_model()
        # info = dict(idx_reward=idx_reward, act_reward=act_reward, pos_reward=pos_reward)
        info = dict(action=actu)
        return ob, reward, done, info

    def reset_model(self):
        qpos = self.np_random.uniform(size=1, low=0, high=0.2)
        qvel = self.np_random.uniform(size=1, low=-0.3, high=0.3)
        self.set_state(qpos, qvel)
        return self._get_obs()

    def _get_obs(self):
        return np.concatenate([self.sim.data.qpos, self.sim.data.qvel]).ravel()

    def viewer_setup(self):
        v = self.viewer
        v.cam.trackbodyid = 0
        v.cam.distance = self.model.stat.extent
