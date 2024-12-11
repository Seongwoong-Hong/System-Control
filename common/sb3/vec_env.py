from collections import OrderedDict
from copy import deepcopy
from typing import Any, Callable, List, Optional, Sequence, Type, Union, Dict

import gym
import numpy as np
import torch as th
from stable_baselines3.common.vec_env import VecNormalize

from stable_baselines3.common.vec_env.base_vec_env import VecEnv, VecEnvIndices, VecEnvObs, VecEnvStepReturn
from stable_baselines3.common.vec_env.util import copy_obs_dict, obs_space_info


def dict_to_obs(space: gym.spaces.Space, obs_dict: Dict[Any, np.ndarray]) -> VecEnvObs:
    """
    Convert an internal representation raw_obs into the appropriate type
    specified by space.

    :param space: an observation space.
    :param obs_dict: a dict of numpy arrays.
    :return: returns an observation of the same type as space.
        If space is Dict, function is identity; if space is Tuple, converts dict to Tuple;
        otherwise, space is unstructured and returns the value raw_obs[None].
    """
    if isinstance(space, gym.spaces.Dict):
        return obs_dict
    elif isinstance(space, gym.spaces.Tuple):
        assert len(obs_dict) == len(space.spaces), "size of observation does not match size of observation space"
        return tuple((obs_dict[i] for i in range(len(space.spaces))))
    else:
        assert set(obs_dict.keys()) == {None}, "multiple observation keys for unstructured observation space"
        return obs_dict[None]


class IsaacDummyVecEnv(VecEnv):
    """
    Creates a simple vectorized wrapper for multiple environments, calling each environment in sequence on the current
    Python process. This is useful for computationally simple environment such as ``cartpole-v1``,
    as the overhead of multiprocess or multithread outweighs the environment computation time.
    This can also be used for RL methods that
    require a vectorized environment, but that you want a single environments to train with.

    :param envs: a list of functions
        that return environments to vectorize
    """

    def __init__(self, envs):
        self.envs = envs
        VecEnv.__init__(self, envs.num_envs, envs.observation_space, envs.action_space)
        obs_space = self.envs.observation_space
        self.keys, shapes, dtypes = obs_space_info(obs_space)

        self.buf_obs = OrderedDict([(k, th.zeros((self.num_envs,) + tuple(shapes[k]), dtype=th.float32)) for k in self.keys])
        self.buf_dones = th.zeros((self.num_envs,), dtype=th.bool)
        self.buf_rews = th.zeros((self.num_envs,), dtype=th.float32)
        self.buf_infos = [{} for _ in range(self.num_envs)]
        self.actions = None
        self.metadata = envs.metadata

    def step_async(self, actions: np.ndarray) -> None:
        self.actions = actions

    def step_wait(self):
        obs_dict, rew_buf, reset_buf, extras = self.envs.step(th.tensor(self.actions).to(self.envs.device))
        self.buf_rews = rew_buf
        self.buf_dones = reset_buf
        # for key in extras.keys():
        #     for idx in range(self.num_envs):
        #         self.buf_infos[idx] = {key: extras[key].cpu().numpy()[idx]}
        self._save_obs(obs_dict['obs'])
        return (self._obs_from_buf(), self.buf_rews.copy(), self.buf_dones.copy(), deepcopy(self.buf_infos))

    def seed(self, seed: Optional[int] = None) -> List[Union[None, int]]:
        seeds = list()
        for idx, env in enumerate(self.envs):
            seeds.append(env.seed(seed + idx))
        return seeds

    def reset(self) -> VecEnvObs:
        obs_dict = self.envs.reset()
        self._save_obs(obs_dict['obs'].cpu().numpy())
        return self._obs_from_buf()

    def close(self) -> None:
        for env in self.envs:
            env.close()

    def get_images(self) -> Sequence[np.ndarray]:
        return [env.render(mode="rgb_array") for env in self.envs]

    def render(self, mode: str = "human") -> Optional[np.ndarray]:
        """
        Gym environment rendering. If there are multiple environments then
        they are tiled together in one image via ``BaseVecEnv.render()``.
        Otherwise (if ``self.num_envs == 1``), we pass the render call directly to the
        underlying environment.

        Therefore, some arguments such as ``mode`` will have values that are valid
        only when ``num_envs == 1``.

        :param mode: The rendering type.
        """
        if self.num_envs == 1:
            return self.envs[0].render(mode=mode)
        else:
            return super().render(mode=mode)

    def _save_obs(self, obs: VecEnvObs) -> None:
        for key in self.keys:
            if key is None:
                self.buf_obs[key] = obs
            else:
                self.buf_obs[key] = obs[key]

    def _obs_from_buf(self) -> VecEnvObs:
        return dict_to_obs(self.observation_space, copy_obs_dict(self.buf_obs))

    def get_attr(self, attr_name: str, indices: VecEnvIndices = None) -> List[Any]:
        attr = getattr(self.envs, attr_name)
        return list(attr(indices))

    def set_attr(self, attr_name: str, value: Any, indices: VecEnvIndices = None) -> None:
        attr = getattr(self.envs, attr_name)
        attr[indices] = value
        setattr(self.envs, attr_name, attr)

    def env_method(self, method_name: str, *method_args, indices: VecEnvIndices = None, **method_kwargs) -> List[Any]:
        return getattr(self.envs, method_name)(*method_args, **method_kwargs)

    def env_is_wrapped(self, wrapper_class: Type[gym.Wrapper], indices: VecEnvIndices = None) -> List[bool]:
        """Check if worker environments are wrapped with a given wrapper"""
        target_envs = self._get_target_envs(indices)
        # Import here to avoid a circular import
        from stable_baselines3.common import env_util

        return [env_util.is_wrapped(env_i, wrapper_class) for env_i in target_envs]

    def _get_target_envs(self, indices: VecEnvIndices) -> List[gym.Env]:
        indices = self._get_indices(indices)
        return [self.envs[i] for i in indices]

class IsaacVecNormalize(VecNormalize):
    def step_wait(self) -> VecEnvStepReturn:
        """
        Apply sequence of actions to sequence of environments
        actions -> (observations, rewards, dones)

        where ``dones`` is a boolean vector indicating whether each element is new.
        """
        obs, rewards, dones, infos = self.venv.step_wait()
        self.old_obs = obs
        self.old_reward = rewards

        if self.training:
            if isinstance(obs, dict) and isinstance(self.obs_rms, dict):
                for key in self.obs_rms.keys():
                    self.obs_rms[key].update(obs[key])
            else:
                self.obs_rms.update(obs)

        obs = self.normalize_obs(obs)

        if self.training:
            self._update_reward(rewards)
        rewards = self.normalize_reward(rewards)

        self.ret[dones] = 0
        return obs, rewards, dones, infos