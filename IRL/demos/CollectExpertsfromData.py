from imitation.data import types
from common.rollouts import generate_trajectories_from_data
from common.util import make_env
from scipy import io
import numpy as np

if __name__ == '__main__':
    env_type = "DiscretizedHuman"
    env_id = f"{env_type}-v2"
    env = make_env(env_id, h=[0.03, 0.03, 0.05, 0.08])
    act_coeff = 1
    # act_coeff = env.model.actuator_gear[0, 0]
    trajectories = []
    subi = 7
    actuation = 3
    sub = f"sub{subi:02d}"
    for i in range(5 * (actuation - 1), 5 * actuation):
        for j in range(5):
            file = f"HPC/{sub}_cropped/{sub}i{i + 1}_{j}.mat"
            state = io.loadmat(file)['state'][:, :4]
            # print(io.loadmat(file)['tq'].max(axis=0), io.loadmat(file)['tq'].min(axis=0))
            state = env.get_obs_from_idx(env.get_idx_from_obs(state))
            data = {'state': state,
                    'T': io.loadmat(file)['tq'] / act_coeff,
                    'pltq': io.loadmat(file)['pltq'] / act_coeff,
                    'bsp': io.loadmat(file)['bsp'],
                    }
            trajectories += generate_trajectories_from_data(data, env)
    save_name = f"{env_type}/{sub}_{actuation}.pkl"
    types.save(save_name, trajectories)
    print("Expert Trajectories are saved")
