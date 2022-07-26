import json
import pickle
import numpy as np

from scipy import io
from imitation.data import types

from gym_envs.envs import FaissDiscretizationInfo, DataBasedDiscretizationInfo
from common.rollouts import generate_trajectories_from_data
from common.util import make_env


if __name__ == '__main__':
    env_type = "DiscretizedPendulum"
    env_id = f"{env_type}-v2"
    with open(f"bound_info.json", "r") as f:
        bound_info = json.load(f)
    with open(f"{env_type}/databased_contlqr/obs_info_tree_1500.pkl", "rb") as f:
        obs_info_tree = pickle.load(f)
    with open(f"{env_type}/databased_contlqr/acts_info_tree_30.pkl", "rb") as f:
        acts_info_tree = pickle.load(f)
    obs_info = FaissDiscretizationInfo([0.05,0.3,], [-0.05, -0.08], obs_info_tree)
    acts_info = FaissDiscretizationInfo([40], [-30], acts_info_tree)
    env = make_env(env_id, obs_info=obs_info, acts_info=acts_info)
    act_coeff = 1
    # act_coeff = env.model.actuator_gear[0, 0]
    for subi in [5]:
        # subi = 3
        subj = f"sub{subi:02d}"
        for actuation in range(1, 2):
            # max_states = bound_info[sub][actuation - 1]["max_states"]
            # min_states = bound_info[sub][actuation - 1]["min_states"]
            # max_torques = bound_info[sub][actuation - 1]["max_torques"]
            # min_torques = bound_info[sub][actuation - 1]["min_torques"]
            # env.set_bounds(max_states, min_states, max_torques, min_torques)
            trajectories = []
            target_data = f"{env_type}/quadcost_lqr.pkl"
            with open(target_data, "rb") as f:
                trajs = pickle.load(f)
            for traj in trajs:
                obs = env.get_obs_from_idx(env.get_idx_from_obs(traj.obs))
                acts = env.get_acts_from_idx(env.get_idx_from_acts(traj.acts))
                data_dict = {'obs': obs, 'acts': acts, 'rews': traj.rews.flatten(), 'infos': None}
                disc_traj = types.TrajectoryWithRew(**data_dict)
                trajectories.append(disc_traj)
            save_name = f"{env_type}/databased_faiss_contlqr/quadcost_150030_from_contlqr.pkl"
            types.save(save_name, trajectories)
            print(f"Expert Trajectory {save_name} is saved")
