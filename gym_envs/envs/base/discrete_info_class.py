import numpy as np

from gym_envs.envs import DiscretizationInfo
from gym_envs.envs import InformationTree


class UncorrDiscreteInfo(DiscretizationInfo):
    @property
    def disc_info(self):
        return [(info[1:] + info[:-1]) / 2 for info in self.info_list]

    def set_info(self, high, low):
        self.high = high
        self.low = low
        self.info_list = []
        for high, low, n in zip(high, low, self.dims):
            self.info_list.append(np.linspace(low, high, n + 1))

    def get_vectorized(self):
        info_vec = np.stack(np.meshgrid(*self.disc_info, indexing='ij'), -1).reshape(-1, len(self.disc_info))
        return info_vec

    def get_idx_from_info(self, info:np.ndarray):
        assert (np.max(info, axis=0) <= self.high + 1e-6).all() or (np.min(info, axis=0) >= self.low - 1e-6).all()
        dims = self.dims
        idx = []
        for i, whole_candi in enumerate(self.info_list):
            idx.append((info[:, [i]] - whole_candi[:-1] >= 0).sum(axis=-1) - 1)
        tot_idx = np.ravel_multi_index(np.array(idx), dims, order='C')
        return tot_idx.flatten()

    def get_info_from_idx(self, idx:np.ndarray):
        info_vec = self.get_vectorized()
        return info_vec[idx]


class DataBasedDiscretizationInfo(DiscretizationInfo):
    def __init__(self, dims):
        self.done_adaptation = False
        self.info_tree = None
        super(DataBasedDiscretizationInfo, self).__init__(dims=dims)

    def set_info(self, high, low, info_tree=None):
        self.high = high
        self.low = low
        self.info_tree = info_tree

    def get_vectorized(self):
        all_states = []
        for node in self.info_tree.data:
            all_states.append(node.value)
        return np.stack(all_states)

    def get_idx_from_info(self, info:np.ndarray):
        assert len(info.shape) == 2
        indices = []
        for state in info:
            _, idx = self.info_tree.find_target_node(state)
            indices.append(idx)
        return np.stack(indices)

    def get_info_from_idx(self, idx:np.ndarray):
        return self.get_vectorized()[idx]