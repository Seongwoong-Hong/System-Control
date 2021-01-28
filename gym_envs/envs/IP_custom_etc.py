import csv
import math
import mujoco_py
import os

import numpy as np
from gym import utils, spaces
from gym.envs.mujoco import mujoco_env


class IPCustom_EX(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self):
        data_path = ["trajectory1.csv", "trajectory2.csv", "trajectory3.csv",
                     "trajectory4.csv", "trajectory5.csv", "trajectory6.csv", "trajectory7.csv"]
        self.trajdata = TrajData(data_path, 3000)
        utils.EzPickle.__init__(self)
        fullpath = os.path.join(os.path.dirname(__file__), "assets", "IP_custom.xml")
        self.frame_skip = 1
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

        self.action_space = spaces.Box(low=-5, high=5, shape=(1,), dtype='float64')
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(3,), dtype='float64')
        self.seed()

    def step(self, a):
        ob = self._get_obs()
        done = bool(self.trajdata.ep_len >= 3000) or bool(abs(ob[0]) >= .3)
        if self.trajdata.frame >= 3000:
            tar = self.trajdata.reset_frame(3000)
        else:
            tar = self.trajdata.target()
        pos_reward = 0.6 * math.exp(-2100 * (ob[0] - tar[0]) ** 2)
        vel_reward = 0.3 * math.exp(-2800 * (ob[1] - tar[1]) ** 2)
        act_reward = 0.1 * math.exp(-10 * (a - tar[2]) ** 2)
        reward = pos_reward + vel_reward + act_reward
        self.trajdata.update_frame()
        self.do_simulation(a, self.frame_skip)
        ob = self._get_obs()
        return ob, reward, done, {}

    def reset_model(self):
        frame = math.floor(10 * self.np_random.uniform(size=1, low=0, high=1))
        self.trajdata.open_file(math.floor(self.np_random.uniform(size=1, low=0, high=7)))
        qpos, qvel, _ = self.trajdata.reset_frame(frame)
        # self.trajdata.open_file(0)
        # qpos, qvel, _ = self.trajdata.reset_frame(0)
        self.set_state(np.array([qpos]), np.array([qvel]))
        self.trajdata.ep_len = 1
        return self._get_obs()

    def _get_obs(self):
        return np.concatenate([self.sim.data.qpos, self.sim.data.qvel, np.array([self.trajdata.frame/3000])]).ravel()

    def viewer_setup(self):
        v = self.viewer
        v.cam.trackbodyid = 0
        v.cam.distance = self.model.stat.extent


class TrajData:
    def __init__(self, path, n_steps):
        self.frame = 0
        self.ep_len = 0
        self.path = path
        self.data = np.array((3, n_steps))

    def reset_frame(self, frame=0):
        self.frame = frame
        return self.data[frame]

    def update_frame(self):
        self.ep_len += 1
        self.frame += 1

    def target(self):
        return self.data[self.frame]

    def open_file(self, nb):
        with open(self.path[nb], 'r') as f:
            reader = csv.reader(f)
            for i, txt in enumerate(reader):
                self.data[i] = np.array([float(txt[0]), float(txt[1]), float(txt[2])])
