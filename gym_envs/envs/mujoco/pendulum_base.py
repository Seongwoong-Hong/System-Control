import mujoco_py
import numpy as np
from copy import deepcopy
from gym.envs.mujoco import mujoco_env


class BasePendulum(mujoco_env.MujocoEnv):
    def __init__(self, filepath, bsp=None, humanStates=None, PDgain=None, ankle_max=None):
        self.timesteps = 0
        self._epi_len = 360
        self._ptb_idx = 0
        self._action_frame_skip = 1
        self.ankle_max = ankle_max
        self.PDgain = PDgain
        if PDgain is None:
            self.PDgain = np.array([1000, 100])
        if bsp is not None:
            self._set_body_config(filepath, bsp)
        self._ptb_acc = np.zeros(self._epi_len)
        self._ptb_range = np.array([0.03, 0.045, 0.06, 0.075, 0.09, 0.12, 0.15])
        self._maxT = 3
        self._ptbT = 1/3
        self._humanStates = humanStates
        for self._humanData in self._humanStates:
            if self._humanData is not None:
                break
        if not isinstance(self._humanData, np.ndarray):
            raise Exception("잘못된 실험 데이터 입력")
        mujoco_env.MujocoEnv.__init__(self, filepath, 1)

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
        self._ptb_idx %= len(self._humanStates)
        self._humanData = self._humanStates[self._ptb_idx]
        while self._humanData is None:
            self._ptb_idx += 1
            self._ptb_idx %= len(self._humanStates)
            self._humanData = self._humanStates[self._ptb_idx]
        x_max = self._ptb_range[self._ptb_idx // 5]
        ptb_act_t = 1/3
        self._ptb_acc = np.zeros(int(self._ptbT / self.dt))
        self._ptb_acc = np.append(self._ptb_acc, 6*x_max/ptb_act_t**2 * (1 - 2*np.linspace(1, 0, int(ptb_act_t/self.dt))))
        self._ptb_acc = np.append(self._ptb_acc, np.zeros(int((self._maxT - self._ptbT - ptb_act_t)/self.dt)))
        self._ptb_idx += 1

    @property
    def obs(self):
        return self._get_obs()

    @property
    def ptb_acc(self):
        return self._ptb_acc

    def viewer_setup(self):
        v = self.viewer
        v.cam.trackbodyid = 0
        v.cam.distance = self.model.stat.extent

    def _get_obs(self):
        return np.concatenate([self.sim.data.qpos, self.sim.data.qvel]).ravel()

    @staticmethod
    def _set_body_config(filepath, bsp):
        raise NotImplementedError
