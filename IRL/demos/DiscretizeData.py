from imitation.data import types
from common.rollouts import generate_trajectories_from_data
from common.util import make_env
from scipy import io
import numpy as np
import json
import pickle

if __name__ == '__main__':
    env_type = "DiscretizedPendulum"
    env_id = f"{env_type}_DataBased-v2"
    with open(f"bound_info.json", "r") as f:
        bound_info = json.load(f)
    env = make_env(env_id, N=[29, 29], NT=[51])
    with open("../../tests/envs/obs_test.pkl", "rb") as f:
        obs_info_tree = pickle.load(f)
    with open("../../tests/envs/acts_test.pkl", "rb") as f:
        acts_info_tree = pickle.load(f)
    env.obs_info.info_tree = obs_info_tree
    env.acts_info.info_tree = acts_info_tree
    act_coeff = 1
    # act_coeff = env.model.actuator_gear[0, 0]
    for subi in [5]:
        # subi = 3
        subj = f"sub{subi:02d}"
        for actuation in range(4, 5):
            # max_states = bound_info[sub][actuation - 1]["max_states"]
            # min_states = bound_info[sub][actuation - 1]["min_states"]
            # max_torques = bound_info[sub][actuation - 1]["max_torques"]
            # min_torques = bound_info[sub][actuation - 1]["min_torques"]
            # env.set_bounds(max_states, min_states, max_torques, min_torques)
            trajectories = []
            target_data = f"{env_type}/quadcost_lqr_many.pkl"
            with open(target_data, "rb") as f:
                trajs = pickle.load(f)
            for traj in trajs:
                obs = env.get_obs_from_idx(env.get_idx_from_obs(traj.obs))
                acts = env.get_acts_from_idx(env.get_idx_from_acts(traj.acts))
                data_dict = {'obs': obs, 'acts': acts, 'rews': traj.rews.flatten(), 'infos': None}
                disc_traj = types.TrajectoryWithRew(**data_dict)
                trajectories.append(disc_traj)
            save_name = f"{env_type}/databased_lqr/quadcost_from_contlqr_many.pkl"
            types.save(save_name, trajectories)
            print(f"Expert Trajectory {save_name} is saved")
