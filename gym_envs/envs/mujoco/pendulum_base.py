import os
import time
from typing import List, Union

import mujoco_py
import numpy as np
from copy import deepcopy
from gym.envs.mujoco import mujoco_env


class BasePendulum(mujoco_env.MujocoEnv):
    def __init__(
            self,
            filepath: str,
            bsp=None,
            humanStates: List = None,
            ankle_limit: str = "satu",
            ankle_torque_max: Union[int, float] = None,
            stiffness: List = 0.,
            damping: List = 0.,
            ptb_act_time: float = 1 / 3,
            torque_rate_limit: bool = False,
            use_seg_ang: bool = False,
            delay: bool = False,
            delayed_time: float = 0.1,
    ):
        self.timesteps = 0
        self._epi_len = 600
        self._ptb_idx = 0
        self._next_ptb_idx = self._ptb_idx
        self.ankle_limit = ankle_limit
        self.ankle_torque_max = ankle_torque_max
        self.torque_rate_limit = torque_rate_limit
        self.use_seg_ang = use_seg_ang
        self.delay = delay
        self._jnt_stiffness = stiffness
        self._jnt_damping = damping
        if bsp is not None:
            filepath = self._set_body_config(filepath, bsp)
        self._ptb_acc = np.zeros(self._epi_len)
        self._ptb_data_range = np.array([0.03, 0.045, 0.06, 0.075, 0.09, 0.12, 0.15])
        self._ptb_act_time = ptb_act_time
        self._humanStates = humanStates
        for self._humanData in self._humanStates:
            if self._humanData is not None:
                break
        if not isinstance(self._humanData, np.ndarray):
            raise Exception("잘못된 실험 데이터 입력")
        try:
            mujoco_env.MujocoEnv.__init__(self, filepath, 1)
        except AssertionError:
            if not os.path.isfile(filepath):
                time.sleep(1)
            assert os.path.isfile(filepath), f"Is file exist? {os.path.isfile(filepath)}"
            mujoco_env.MujocoEnv.__init__(self, filepath, 1)
        if bsp is not None:
            os.remove(filepath)
        if self.delay:
            self.delayed_act = np.zeros([round(delayed_time // self.dt), self.action_space.shape[0]])

    def __deepcopy__(self, memo):
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        result.__dict__.update(self.__dict__)
        for k, v in self.__dict__.items():
            if k not in ['model', 'sim', 'data']:
                setattr(result, k, deepcopy(v, memo))
        result.sim = mujoco_py.MjSim(result.model)
        result.data = result.sim.data
        result.seed()
        return result

    def step(self, action: np.ndarray):
        raise NotImplementedError

    def reset_model(self):
        self.timesteps = 0
        self.reset_ptb()
        init_state = self._humanData[0]
        self.set_state(init_state[:self.model.nq], init_state[self.model.nq:])
        return self._get_obs()

    def reset_ptb(self):
        self._ptb_idx = self._next_ptb_idx % len(self._humanStates)
        self._humanData = self._humanStates[self._ptb_idx]
        while self._humanData is None:
            self._ptb_idx += 1
            self._ptb_idx %= len(self._humanStates)
            self._humanData = self._humanStates[self._ptb_idx]
        self._set_ptb_acc()
        self._next_ptb_idx = self._ptb_idx + 1

    def _set_ptb_acc(self):
        st_time_idx = 40
        x_max = -self._ptb_data_range[self._ptb_idx // 5]  # Backward direction(-)
        # x_max = 0
        fddx = self._cal_ptb_acc(x_max)
        self._ptb_acc = np.append(np.zeros(st_time_idx), fddx)
        self._ptb_acc = np.append(self._ptb_acc, np.zeros(self._epi_len - st_time_idx - len(fddx)))

    def _cal_ptb_acc(self, x_max):
        t = self._ptb_act_time * np.linspace(0, 1, round(self._ptb_act_time / self.dt))

        A = np.array([[self._ptb_act_time ** 3, self._ptb_act_time ** 4, self._ptb_act_time ** 5],
                      [3, 4 * self._ptb_act_time, 5 * self._ptb_act_time ** 2],
                      [6, 12 * self._ptb_act_time, 20 * self._ptb_act_time ** 2]])
        b = np.array([x_max, 0, 0]).T
        a = np.linalg.inv(A)@b

        fddx = t*(np.concatenate([np.ones([round(self._ptb_act_time/self.dt), 1]), t.reshape(-1, 1), (t**2).reshape(-1, 1)], axis=1) @ np.array([6 * a[0], 12*a[1], 20*a[2]]).T)  # 가속도 5차 regression
        # fddx = 6 * x_max / ptb_act_t ** 2 * (1 - 2 * np.linspace(0, 1, int(ptb_act_t / self.dt)))  # 가속도 3차 regression
        return fddx

    @property
    def obs(self):
        return self._get_obs()

    @property
    def ptb_acc(self):
        return self._ptb_acc

    @property
    def ptb_idx(self):
        return self._ptb_idx % len(self._humanStates)

    def viewer_setup(self):
        v = self.viewer
        v.cam.trackbodyid = 0
        v.cam.distance = self.model.stat.extent

    def _get_viewer(self, mode):
        self.viewer = self._viewers.get(mode)
        if self.viewer is None:
            if mode == 'human':
                self.viewer = mujoco_py.MjViewer(self.sim)
            elif mode == 'rgb_array' or mode == 'depth_array':
                self.viewer = mujoco_py.MjRenderContextOffscreen(self.sim, -1, opengl_backend='glfw')

            self.viewer_setup()
            self._viewers[mode] = self.viewer
        return self.viewer

    def _get_obs(self):
        return np.concatenate([self.sim.data.qpos, self.sim.data.qvel]).astype(np.float32).ravel()

    def _set_body_config(self, filepath, bsp):
        raise NotImplementedError
