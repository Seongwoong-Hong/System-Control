import os
import numpy as np
from gym import utils, spaces
from gym.envs.mujoco import mujoco_env


class IDPCustom(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self, n_steps=None):
        self.__timesteps__ = 0
        self.n_steps = n_steps
        mujoco_env.MujocoEnv.__init__(self, os.path.join(os.path.dirname(__file__), "assets", "IDP_custom.xml"), 8)
        utils.EzPickle.__init__(self)
        self.action_space = spaces.Box(low=-1, high=1, shape=(2, ))
        self.init_qpos = np.array([0.0, 0.0])
        self.init_qvel = np.array([0.0, 0.0])

    def step(self, action):
        self.do_simulation(action, self.frame_skip)
        ob = self._get_obs()
        r = -0.01 * (ob[0] ** 2 + ob[1] ** 2 + 0.0001 * self.data.qfrc_actuator @ np.eye(2, 2) @ self.data.qfrc_actuator.T)
        done = False
        info = {}
        if self.n_steps is None:
            pass
        elif self.__timesteps__ == self.n_steps:
            done = True
            info = {"terminal observation": ob}
            self.__timesteps__ = 0
        if abs(ob[0]) > 2 or abs(ob[1]) > 3:
            done = True
        self.__timesteps__ += 1
        return ob, r, done, info

    @property
    def timesteps(self):
        return self.__timesteps__

    def _get_obs(self):
        return np.concatenate([
            self.sim.data.qpos,  # link angles
            self.sim.data.qvel   # link angular velocities
        ]).ravel()

    def reset_model(self):
        self.set_state(
            self.init_qpos + self.np_random.uniform(low=-.2, high=.2, size=self.model.nq),
            self.init_qvel + self.np_random.uniform(low=-.2, high=.2, size=self.model.nv)
        )
        return self._get_obs()

    def viewer_setup(self):
        v = self.viewer
        v.cam.trackbodyid = 0
        v.cam.distance = self.model.stat.extent * 0.5
        v.cam.lookat[2] = 0.5  # 0.12250000000000005  # v.model.stat.center[2]


class IDPCustomExp(IDPCustom):
    def __init__(self, n_steps=None):
        super().__init__(n_steps=n_steps)
        self.init_group = np.array([[[+0.10, +0.10], [+0.05, -0.05]],
                                    [[+0.15, +0.10], [-0.05, +0.05]],
                                    [[-0.16, +0.20], [+0.10, -0.10]],
                                    [[-0.10, +0.06], [+0.05, -0.10]],
                                    [[+0.05, +0.15], [-0.20, -0.20]],
                                    [[-0.05, +0.05], [+0.15, +0.15]],
                                    [[+0.12, +0.05], [-0.10, -0.15]],
                                    [[-0.08, +0.15], [+0.05, -0.15]],
                                    [[-0.15, +0.20], [-0.10, +0.05]],
                                    [[+0.20, +0.01], [+0.09, -0.15]]])
        self.i = 0

    def reset_model(self):
        q = self.init_group[self.i % len(self.init_group)]
        self.set_state(
            q[0].reshape(self.model.nq),
            q[1].reshape(self.model.nv)
        )
        self.i += 1
        return self._get_obs()
